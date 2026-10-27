"""
CachedLLM -- wraps ANY llm backend (OpenAI, Ollama, ...) so a slow / flaky / small local
model cannot stall or break an experiment.

Why it exists
-------------
run_experiment.py asks the LLM for the SAME day-by-day skeleton once per system per query
(7 systems x 225 queries + 45 case-library queries ~ 1,600 calls). This wrapper:

  1. CACHES   one call per distinct query (kept on disk, so re-runs are instant),
  2. TIMES OUT  every call after `timeout_s` seconds (a hung server never blocks the run),
  3. VALIDATES the answer (right number of days, legs chain, returns to origin, right number
               of cities) and REJECTS bad output from small models,
  4. FALLS BACK to the rule-based skeleton (MockLLM) when the call fails or is invalid,
  5. BREAKS THE CIRCUIT after `max_failures` consecutive failures: the LLM is switched off for
               the rest of the run, with one clear warning (no more 2-minute waits).

Usage (two lines in run_experiment.py, right after `llm = get_llm(...)`):

    from llm.cached import CachedLLM
    llm = CachedLLM(llm, cache_path="llm_cache.json", timeout_s=60)
    ...
    print(llm.stats)          # calls / cache hits / fallbacks / timeouts
"""

from __future__ import annotations
import concurrent.futures as cf
import json
import os
import threading
from typing import Optional

from data_model import Query
from llm.interface import LLMInterface, MockLLM


def parse_leg(entry: str):
    """'from A to B' -> ('A', 'B');  'B' -> None (a stay day)."""
    e = entry.strip()
    if e.lower().startswith("from ") and " to " in e:
        o, d = e[5:].split(" to ", 1)
        return o.strip(), d.strip()
    return None


def valid_skeleton(sk, q: Query, valid_cities: Optional[set] = None) -> bool:
    """A skeleton is valid iff: it is a list of `days` strings; travel legs chain from the origin
    and the trip returns to the origin; stay days name the current city; exactly
    `visiting_city_number` distinct cities are visited (and exist, if `valid_cities` is given)."""
    if not isinstance(sk, list) or len(sk) != q.days or not all(isinstance(x, str) for x in sk):
        return False
    cur, visited = q.org, []
    for entry in sk:
        leg = parse_leg(entry)
        if leg:
            o, d = leg
            if o != cur:
                return False
            cur = d
            if d != q.org and d not in visited:
                visited.append(d)
        elif entry.strip() != cur:
            return False
    if cur != q.org or len(visited) != max(1, q.visiting_city_number):
        return False
    if valid_cities is not None and any(c not in valid_cities for c in visited):
        return False
    return True


class CachedLLM(LLMInterface):
    def __init__(self, inner: LLMInterface, cache_path: Optional[str] = "llm_cache.json",
                 timeout_s: float = 60.0, max_failures: int = 3, fallback: Optional[LLMInterface] = None,
                 only_multi_city: bool = False, valid_cities: Optional[set] = None, verbose: bool = True):
        """only_multi_city=True: skip the LLM for single-city queries (their skeleton is trivial:
        'from A to B', 'B', ..., 'from B to A') and use it only where it makes a real decision."""
        self.inner = inner
        self.is_real = getattr(inner, "is_real", False)
        self.fallback = fallback or MockLLM()
        self.cache_path = cache_path
        self.timeout_s = timeout_s
        self.max_failures = max_failures
        self.only_multi_city = only_multi_city
        self.valid_cities = valid_cities
        self.verbose = verbose
        self.consecutive_failures = 0
        self.disabled = False
        self._lock = threading.Lock()
        self.stats = {"llm_calls": 0, "cache_hits": 0, "invalid_output": 0, "timeouts_or_errors": 0,
                      "fallbacks": 0, "skipped_single_city": 0}
        self.cache: dict = {}
        if cache_path and os.path.exists(cache_path):
            try:
                with open(cache_path, encoding="utf-8") as f:
                    self.cache = json.load(f)
            except (OSError, ValueError):
                self.cache = {}

    # ------------------------------------------------------------------ #
    @staticmethod
    def _key(q: Query) -> str:
        return f"{q.org}|{q.dest}|{q.days}|{q.visiting_city_number}|{q.people_number}|{q.budget}"

    def _save(self):
        if not self.cache_path:
            return
        tmp = self.cache_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.cache, f, indent=0)
        os.replace(tmp, self.cache_path)

    def _call_with_timeout(self, fn, *args):
        ex = cf.ThreadPoolExecutor(max_workers=1)
        fut = ex.submit(fn, *args)
        try:
            return fut.result(timeout=self.timeout_s)
        finally:
            ex.shutdown(wait=False, cancel_futures=True)      # never block on a hung server

    def _warn(self, msg: str):
        if self.verbose:
            print(f"[CachedLLM] {msg}", flush=True)

    # ------------------------------------------------------------------ #
    def generate_initial_skeleton(self, query: Query) -> list:
        key = self._key(query)
        with self._lock:
            if key in self.cache:
                self.stats["cache_hits"] += 1
                return list(self.cache[key])

        if self.only_multi_city and query.visiting_city_number <= 1:
            self.stats["skipped_single_city"] += 1
            return self.fallback.generate_initial_skeleton(query)

        skeleton = None
        if not self.disabled:
            try:
                self.stats["llm_calls"] += 1
                cand = self._call_with_timeout(self.inner.generate_initial_skeleton, query)
                if valid_skeleton(cand, query, self.valid_cities):
                    skeleton = cand
                    self.consecutive_failures = 0
                else:
                    self.stats["invalid_output"] += 1
                    self._warn(f"LLM skeleton rejected (invalid) for {query.org}->{query.dest}: {str(cand)[:80]}")
            except Exception as e:                                   # noqa: BLE001  (timeout, HTTP, JSON ...)
                self.stats["timeouts_or_errors"] += 1
                self.consecutive_failures += 1
                self._warn(f"LLM call failed ({type(e).__name__}: {str(e)[:80] or 'timeout'}) "
                           f"[{self.consecutive_failures}/{self.max_failures}]")
                if self.consecutive_failures >= self.max_failures:
                    self.disabled = True
                    self._warn("Too many consecutive failures -> LLM DISABLED for the rest of this run; "
                               "using the rule-based skeleton. Check that the model server is running "
                               "and responds (e.g. `ollama run <model> hi`).")

        if skeleton is None:
            self.stats["fallbacks"] += 1
            skeleton = self.fallback.generate_initial_skeleton(query)
        with self._lock:
            self.cache[key] = list(skeleton)
            self._save()
        return list(skeleton)

    # everything else is delegated to the wrapped backend (or the fallback if it lacks it)
    def extract_constraints(self, query_text: str) -> dict:
        return self.inner.extract_constraints(query_text)

    def render_final_plan(self, plan):
        try:
            return self._call_with_timeout(self.inner.render_final_plan, plan)
        except Exception:                                            # noqa: BLE001
            return self.fallback.render_final_plan(plan)

    def replan_with_backprompt(self, plan, violations):
        if self.disabled:
            return None
        try:
            return self._call_with_timeout(self.inner.replan_with_backprompt, plan, violations)
        except Exception:                                            # noqa: BLE001
            return None

    def __getattr__(self, name):          # any other backend-specific attribute / method
        return getattr(self.inner, name)