"""
LLM integration layer -- kept thin and OUTSIDE the solving logic on purpose.
The LLM's three jobs anywhere in this system:
  1. extract_constraints(query_text) -> structured Query dict
  2. generate_initial_plan(query)    -> a rough day-by-day skeleton (city per day)
  3. render_final_plan(plan_state)   -> natural-language itinerary text

The actual constraint satisfaction (agents + DCSPSolver) never calls the LLM.

Two implementations:
  - MockLLM: rule-based, no API key needed. Good enough to exercise the full
    pipeline end-to-end and get your project running today.
  - RealLLM: thin wrapper over the OpenAI-compatible chat completions API.
    Swap in by setting LLM_BACKEND=real and OPENAI_API_KEY in your environment.
"""

from __future__ import annotations
import json
import os
import re
from data_model import Query, PlanState, DayPlan


class LLMInterface:
    is_real = False   # True for backends that make paid API calls
    def extract_constraints(self, query_text: str) -> dict:
        raise NotImplementedError

    def generate_initial_skeleton(self, query: Query) -> list:
        """Return a list of day->city strings, e.g. ['Seattle to LA', 'LA', 'LA to Seattle']."""
        raise NotImplementedError

    def render_final_plan(self, plan: PlanState) -> str:
        raise NotImplementedError

    def replan_with_backprompt(self, plan: PlanState, violations: list):
        """ESCALATION (architecture arrow "(6) Back Prompt"): called at most once,
        only when every negotiation restart failed. Returns a revised PlanState
        skeleton (new city sequence) or None if the LLM cannot improve it."""
        return None


# --------------------------------------------------------------------------- #
# Mock implementation -- no API key required
# --------------------------------------------------------------------------- #

class MockLLM(LLMInterface):
    """
    Rule-based stand-in for an LLM. Good enough to validate the whole
    pipeline (extraction -> agents -> solver -> rendering) without incurring
    API cost during development. Swap for RealLLM once the architecture is
    working end to end.
    """

    _BUDGET_RE = re.compile(r"\$?([\d,]+(?:\.\d+)?)")
    _DAYS_RE = re.compile(r"(\d+)[\s-]*day")
    _PEOPLE_RE = re.compile(r"(\d+)\s+(?:people|person|traveler|travelers)")

    # def extract_constraints(self, query_text: str) -> dict:
    #     """Very rough regex/keyword extraction, standing in for an LLM call.
    #     Replace with a real LLM prompt (see RealLLM) for production use."""
    #     budget_match = self._BUDGET_RE.findall(query_text)
    #     budget = float(budget_match[-1].replace(",", "")) if budget_match else 1500.0

    #     days_match = self._DAYS_RE.search(query_text)
    #     days = int(days_match.group(1)) if days_match else 3

    #     people_match = self._PEOPLE_RE.search(query_text)
    #     people = int(people_match.group(1)) if people_match else 1

    #     local_constraint = {}
    #     for cuisine in ["Chinese", "American", "Italian", "Mexican", "Indian",
    #                      "Mediterranean", "French", "Korean", "Asian"]:
    #         if cuisine.lower() in query_text.lower():
    #             local_constraint.setdefault("cuisine", []).append(cuisine)
    #     if "pet" in query_text.lower():
    #         local_constraint["room rule"] = "No pets"  # user wants pet-friendly -> avoid "No pets" listings... (see note)
    #     if "entire room" in query_text.lower():
    #         local_constraint["room type"] = "Entire Room"
    #     if "no flight" in query_text.lower() or "self-driving" in query_text.lower():
    #         local_constraint["transportation"] = "No flight"
    #     if "no self-driving" in query_text.lower() or "no self driving" in query_text.lower():
    #         local_constraint["transportation"] = "No self-driving"

    #     return {
    #         "days": days,
    #         "people_number": people,
    #         "budget": budget,
    #         "local_constraint": local_constraint,
    #     }
    def extract_constraints(self, query_text: str) -> dict:
        """Offline regex/keyword extraction (MockLLM has no model behind it)."""
        budget_match = self._BUDGET_RE.findall(query_text)
        budget = float(budget_match[-1].replace(",", "")) if budget_match else 1500.0
        days_match = self._DAYS_RE.search(query_text)
        days = int(days_match.group(1)) if days_match else 3
        people_match = self._PEOPLE_RE.search(query_text)
        people = int(people_match.group(1)) if people_match else 1
        local_constraint = {}
        for cuisine in ["Chinese", "American", "Italian", "Mexican", "Indian",
                        "Mediterranean", "French", "Korean", "Asian"]:
            if cuisine.lower() in query_text.lower():
                local_constraint.setdefault("cuisine", []).append(cuisine)
        if "entire room" in query_text.lower():
            local_constraint["room type"] = "Entire Room"
        if "no flight" in query_text.lower():
            local_constraint["transportation"] = "No flight"
        if "no self-driving" in query_text.lower() or "no self driving" in query_text.lower():
            local_constraint["transportation"] = "No self-driving"
        return {"days": days, "people_number": people, "budget": budget,
                "local_constraint": local_constraint}

    def generate_initial_skeleton(self, query: Query) -> list:
        from data import sandbox            # looked up at call time: synthetic OR real database
        k = max(1, query.visiting_city_number)
        cities = cities_for(query)
        k = len(cities)
        if query.days == 2 * k + 1:
            # TravelPlanner pattern: day 1 travel, stay, travel to next city, ..., travel home
            skeleton, prev = [], query.org
            for c in cities:
                skeleton.append(f"from {prev} to {c}")   # travel day
                skeleton.append(c)                       # stay day
                prev = c
            skeleton.append(f"from {prev} to {query.org}")   # return leg
            return skeleton
        skeleton = [f"from {query.org} to {cities[0]}"]
        for c in cities[1:]:
            skeleton.append(f"from {skeleton[-1].split(' to ')[-1]} to {c}")
        while len(skeleton) < query.days:
            skeleton.append(cities[-1])
        skeleton = skeleton[: query.days]
        if len(skeleton) == query.days:
            skeleton[-1] = f"from {cities[-1]} to {query.org}"
        return skeleton

    def render_final_plan(self, plan: PlanState) -> str:
        lines = [f"Travel Plan: {plan.query.org} -> {plan.query.dest} "
                 f"({plan.query.days} days, budget ${plan.query.budget:.0f})"]
        for day in sorted(plan.days_plan):
            dp = plan.days_plan[day]
            lines.append(f"\nDay {day}: {dp.current_city}")
            d = dp.to_travelplanner_dict()
            for field_name in ("transportation", "breakfast", "attraction", "lunch", "dinner", "accommodation"):
                lines.append(f"  {field_name.capitalize()}: {d[field_name]}")
        lines.append(f"\nTotal estimated cost: ${plan.total_cost():.2f} (budget ${plan.query.budget:.2f})")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Real LLM implementation (OpenAI-compatible). Requires `pip install openai`.
# --------------------------------------------------------------------------- #

class RealLLM(LLMInterface):
    is_real = True
    # NOTE: replan_with_backprompt() below is the single optional LLM call made
    # when negotiation + restarts fail.
    # def __init__(self, model: str = "gpt-4-turbo", api_key: str | None = None):
    #     try:
    #         from openai import OpenAI
    #     except ImportError as e:
    #         raise ImportError("pip install openai to use RealLLM") from e
    #     self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
    #     self.model = model

   
    def __init__(
        self,
        model: str = "gpt-4-turbo",
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 300.0,
        max_retries: int = 2,
    ):
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "pip install openai to use RealLLM"
            ) from e

        if base_url:
            # self.client = OpenAI(
            #     base_url=base_url,
            #     api_key=api_key or "ollama",
            # )
            self.client = OpenAI(
                    api_key=api_key,
                    base_url=base_url,
                    timeout=timeout,
                    max_retries=max_retries,
            )
        else:
            self.client = OpenAI(
                api_key=api_key or os.environ.get("OPENAI_API_KEY")
            )

        self.model = model



    # def _chat(self, system: str, user: str) -> str:
    #     resp = self.client.chat.completions.create(
    #         model=self.model,
    #         messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    #         temperature=0.0,
    #     )
    #     return resp.choices[0].message.content
    # def _chat(self, system: str, user: str, json_mode: bool = False) -> str:
    #     messages = [
    #     {"role": "system", "content": os.system},
    #     {"role": "user", "content": user},
    # ]

    #     kwargs = {
    #     "model": self.model,
    #     "messages": messages,
    #     "temperature": 0.0,
    # }

    # # Ask the model to return valid JSON when required.
    #     if json_mode:
    #         kwargs["response_format"] = {"type": "json_object"}

    #     resp = self.client.chat.completions.create(**kwargs)

    #     content = resp.choices[0].message.content

    #     if not content:
    #         raise ValueError("The LLM returned an empty response.")

    #     return content.strip()
    
    def _chat(self, system: str, user: str, json_mode: bool = False,
              max_tokens: int = 400) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": max_tokens,      # hard cap: never let the model run on forever
        }

        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        resp = self.client.chat.completions.create(**kwargs)

        content = resp.choices[0].message.content

        if not content:
            raise ValueError("The LLM returned an empty response.")

        return content.strip()


    # def extract_constraints(self, query_text: str) -> dict:
    #     system = (
    #         "Extract structured travel constraints from the user's query. "
    #         "Return ONLY valid JSON with keys: days (int), people_number (int), "
    #         "budget (float), local_constraint (object with optional keys "
    #         "'room rule', 'cuisine' (list), 'room type', 'transportation')."
    #     )
    #     raw = self._chat(system, query_text)
    #     raw = raw.strip().strip("```json").strip("```").strip()
    #     print("\nDEBUG: Raw LLM response:")
    #     print(repr(raw))
    #     return json.loads(raw)
    
    def extract_constraints(self, query_text: str) -> dict:
        system = """
    Extract travel constraints from the user's query.

    Return ONLY one valid JSON object with these keys:
    {
        "days": 3,
        "people_number": 2,
        "budget": 1500.0,
        "local_constraint": {}
    }

    Rules:
    - days must be an integer.
    - people_number must be an integer.
    - budget must be a number.
    - local_constraint must be an object.
    - Optional local_constraint keys are "room rule", "cuisine",
    "room type", and "transportation".
    - Do not invent traveler names, ages, dates, or activities.
    - Do not generate a travel itinerary.
    - Use the values from the user's query.
    - If a value is missing, use days=3, people_number=1,
    budget=0.0, and local_constraint={}.
    - Return JSON only.
    """

        raw = self._chat(system, query_text, json_mode=True)

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON returned by Ollama: {raw!r}"
            ) from exc

        if not isinstance(data, dict):
            raise ValueError("Expected a JSON object for travel constraints.")

        for key in ("days", "people_number", "budget", "local_constraint"):
            if key not in data:
                raise ValueError(f"Missing required constraint key: {key}")

        data["days"] = int(data["days"])
        data["people_number"] = int(data["people_number"])
        data["budget"] = float(data["budget"])

        if not isinstance(data["local_constraint"], dict):
            raise ValueError("'local_constraint' must be a JSON object.")

        return data



    # def generate_initial_skeleton(self, query: Query) -> list:
    #     system = (
    #         "Given a travel query, output ONLY a JSON list of strings, one per day, "
    #         "each describing the city or city-to-city leg for that day "
    #         "(e.g. [\"from Seattle to LA\", \"LA\", \"from LA to Seattle\"])."
    #     )
    #     user = f"org={query.org}, dest={query.dest}, days={query.days}, " \
    #            f"visiting_city_number={query.visiting_city_number}"
    #     # raw = self._chat(system, user)
    #     # raw = raw.strip().strip("```json").strip("```").strip()
    #     # return json.loads(raw)
    #     raw = self._chat(system, user, json_mode=True)

    #     try:
    #         skeleton = json.loads(raw)
    #     except json.JSONDecodeError as exc:
    #         raise ValueError(
    #             f"LLM returned invalid skeleton JSON: {raw!r}"
    #         ) from exc

    #     if not isinstance(skeleton, list):
    #         raise ValueError("The generated travel skeleton must be a JSON list.")

    #     return skeleton
    
    def generate_initial_skeleton(self, query: Query) -> list:
        """
        Generate a travel skeleton compatible with plan_builder.py.

        Returns a list of strings, with one entry per day.
        Ollama's JSON mode returns an object containing the list.
        """

        system = """
    You are a travel route skeleton generator.

    Return a JSON object in exactly this format:
    {
        "days": [
            "from Seattle to Yakima",
            "Yakima",
            "Yakima"
        ]
    }

    Rules:
    - The "days" value must be a list of strings.
    - Generate exactly the requested number of entries.
    - Use the provided origin and destination.
    - Each entry must describe a city or a city-to-city journey.
    - Do not invent additional cities.
    - Return JSON only, without Markdown or explanations.
    """

        user = (
            f"Origin: {query.org}\n"
            f"Destination: {query.dest}\n"
            f"Number of days: {query.days}\n"
            f"Number of visiting cities: {query.visiting_city_number}"
        )

        raw = self._chat(system, user, json_mode=True)

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Ollama returned invalid skeleton JSON: {raw!r}"
            ) from exc

        # Accept either a JSON object containing the list or a direct list.
        if isinstance(data, dict):
            skeleton = data.get("days", data.get("skeleton"))
        elif isinstance(data, list):
            skeleton = data
        else:
            skeleton = None

        if not isinstance(skeleton, list):
            raise ValueError(
                "Expected a list of daily route descriptions. "
                f"Ollama returned: {raw!r}"
            )

        if not all(isinstance(item, str) for item in skeleton):
            raise ValueError(
                f"Every skeleton entry must be a string: {skeleton!r}"
            )

        # expected_days = int(query.days)

        # if len(skeleton) != expected_days:
        #     raise ValueError(
        #         f"Ollama generated {len(skeleton)} days, "
        #         f"but the query requires {expected_days}. "
        #         f"Skeleton: {skeleton!r}"
        #     )

        # return skeleton
        expected_days = int(query.days)

                # Normalize the skeleton to contain exactly one entry per day.
        if len(skeleton) > expected_days:
                    skeleton = skeleton[:expected_days]

        elif len(skeleton) < expected_days:
                    missing_days = expected_days - len(skeleton)

                    for _ in range(missing_days):
                        skeleton.append(str(query.dest))

        return skeleton



    def render_final_plan(self, plan: PlanState) -> str:
        system = "Render the following structured travel plan JSON as a clear, natural-language itinerary."
        raw = self._chat(system, plan.to_json(), max_tokens=1500)
        return raw

    def replan_with_backprompt(self, plan: PlanState, violations: list):
        from plan_builder import build_initial_plan  # local import avoids a cycle
        system = ("You are a travel-plan skeleton generator. Given the unresolved constraint "
                  "violations, return ONLY a JSON list with one string per day, each either "
                  "'CityName' or 'from A to B'. The list length must equal the number of days.")
        user = f"Days: {plan.query.days}. Origin: {plan.query.org}. Destination: {plan.query.dest}.\n" \
               f"Unresolved violations:\n" + "\n".join(violations[:10])
        try:
            raw = self._chat(system, user).strip().strip("```json").strip("```").strip()
            skeleton = json.loads(raw)
            if not isinstance(skeleton, list) or len(skeleton) != plan.query.days:
                return None
        except Exception:
            return None
        new = PlanState(query=plan.query)
        for day, city in enumerate(skeleton, start=1):
            new.days_plan[day] = DayPlan(day=day, current_city=city)
        return new
def cities_for(query) -> list:
    """The cities of the trip: the ones the user picked in the UI (query.chosen_cities), otherwise the
    database-driven choice of the first k usable cities (TravelPlanner convention)."""
    chosen = getattr(query, "chosen_cities", None)
    if chosen:
        return list(chosen)
    from data import sandbox
    return list(sandbox.plan_cities(query.org, query.dest, max(1, query.visiting_city_number)))


def render_skeleton(query, cities: list) -> list:
    """Turn an ordered city list into the day-by-day skeleton plan_builder expects."""
    k = len(cities)
    if query.days == 2 * k + 1:       # TravelPlanner pattern: travel, stay, travel, stay, ..., return
        sk, prev = [], query.org
        for c in cities:
            sk += [f"from {prev} to {c}", c]
            prev = c
        sk.append(f"from {prev} to {query.org}")
        return sk
    sk = [f"from {query.org} to {cities[0]}"]
    for c in cities[1:]:
        sk.append(f"from {sk[-1].split(' to ')[-1]} to {c}")
    while len(sk) < query.days:
        sk.append(cities[-1])
    sk = sk[: query.days]
    sk[-1] = f"from {cities[-1]} to {query.org}"
    return sk


class OllamaLLM(RealLLM):
    """Local LLM powered by Ollama -- the ONLY model used when --llm ollama (no MockLLM anywhere).

    Job of the LLM here: decide the ORDER in which the (database-valid) cities are visited and
    write the route. Code only normalises the answer into the exact day-by-day format, so a
    small 1B model's formatting slips (wrong day count, markdown, extra keys) do not break a run.
    Results are cached on disk (re-running resumes); a hung server raises instead of silently
    switching to a rule-based model."""

    is_real = False
    HOST = "http://localhost:11434"

    def __init__(self, model: str = "llama3.2:1b", timeout: float = 120.0, attempts: int = 2,
                 max_failures: int = 3, cache_path: str | None = "auto"):
        self._check_server(model)
        super().__init__(model=model, api_key="ollama", base_url=self.HOST + "/v1",
                         timeout=timeout, max_retries=0)
        self.attempts, self.max_failures = attempts, max_failures
        self.consecutive_errors = 0
        self.stats = {"llm_calls": 0, "cache_hits": 0, "format_repaired": 0,
                      "unusable_answer": 0, "errors": 0}
        self.cache_path = (f"llm_cache_ollama_{model.replace(':', '_')}.json"
                           if cache_path == "auto" else cache_path)
        self.cache = {}
        if self.cache_path and os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, encoding="utf-8") as f:
                    self.cache = json.load(f)
            except (OSError, ValueError):
                self.cache = {}

    @classmethod
    def _check_server(cls, model: str) -> None:
        import urllib.request
        try:
            with urllib.request.urlopen(cls.HOST + "/api/tags", timeout=3) as r:
                names = [m.get("name", "") for m in json.load(r).get("models", [])]
        except Exception as e:                                       # noqa: BLE001
            raise RuntimeError(
                f"Cannot reach Ollama at {cls.HOST} ({e}). Start it (Ollama app / `ollama serve`) "
                f"and run `ollama pull {model}`.") from e
        if not any(n == model or n.split(":")[0] == model for n in names):
            raise RuntimeError(f"Model '{model}' is not installed. Run:  ollama pull {model}\n"
                               f"Installed: {names}")

    def _save(self):
        """Best-effort cache write. On Windows the file can be locked (editor, OneDrive, antivirus),
        so write in place, retry briefly, and NEVER let a cache problem stop the experiment."""
        if not self.cache_path:
            return
        import time
        for _ in range(3):
            try:
                with open(self.cache_path, "w", encoding="utf-8") as f:
                    json.dump(self.cache, f)
                return
            except OSError:
                time.sleep(0.2)
        if not getattr(self, "_warned_cache", False):
            self._warned_cache = True
            print(f"[OllamaLLM] cannot write {self.cache_path} (file locked?). Continuing without saving; "
                  f"close any program that has it open.", flush=True)

    @staticmethod
    def _order_from_answer(raw: str, candidates: list):
        """Cities in the order the model mentioned them; missing ones appended. None if it named none."""
        low = raw.lower()
        pos = {c: low.find(c.lower()) for c in candidates}
        named = sorted((c for c in candidates if pos[c] >= 0), key=lambda c: pos[c])
        if not named:
            return None
        return named + [c for c in candidates if c not in named]

    prompt_generator = None          # set by the UI to add a RAG retriever; default built lazily
    last_prompt = None               # {"parts", "system", "user"} of the most recent call (for the UI)
    last_raw = None                  # raw text the model answered with
    last_cached = False

    def generate_initial_skeleton(self, query: Query) -> list:
        from llm.prompt_generator import PromptGenerator
        candidates = cities_for(query)
        k = max(1, query.visiting_city_number)
        key = f"{query.org}|{query.dest}|{query.days}|{k}"
        if getattr(query, "chosen_cities", None):
            key += "|" + ",".join(candidates)
        pg = self.prompt_generator or PromptGenerator()
        built = pg.build(query, candidates)                       # (2) Initial Prompt
        self.last_prompt, self.last_raw, self.last_cached = built, None, False
        if key in self.cache:
            self.stats["cache_hits"] += 1
            hit = self.cache[key]
            self.last_cached = True
            if isinstance(hit, dict):
                self.last_raw = hit.get("raw")
                return list(hit["skeleton"])
            return list(hit)

        system, user = built["system"], built["user"]
        order = None
        for attempt in range(self.attempts):
            try:
                self.stats["llm_calls"] += 1
                raw = self._chat(system, user, json_mode=True, max_tokens=120)
                self.consecutive_errors = 0
            except Exception as e:                                   # noqa: BLE001 (timeout / HTTP)
                self.stats["errors"] += 1
                self.consecutive_errors += 1
                if self.consecutive_errors >= self.max_failures:
                    raise RuntimeError(
                        f"Ollama stopped responding ({type(e).__name__}) {self.max_failures} times in a row. "
                        f"Check `ollama ps` / restart Ollama, then re-run: finished queries are cached "
                        f"in {self.cache_path} and will be skipped.") from e
                continue
            self.last_raw = raw                                       # (3) LLM response
            order = self._order_from_answer(raw, candidates)          # Reformatter: text -> order
            if order is not None:
                if raw.strip().startswith("{") and '"order"' in raw and all(c in raw for c in candidates):
                    pass                                                  # perfect answer
                else:
                    self.stats["format_repaired"] += 1
                break
        if order is None:
            self.stats["unusable_answer"] += 1       # model never named a city: keep database order
            order = candidates
        skeleton = render_skeleton(query, order)                      # Reformatter: order -> JSON skeleton
        self.cache[key] = {"skeleton": skeleton, "raw": self.last_raw}
        self._save()
        return list(skeleton)

# def get_llm(backend: str | None = None, model: str = "gpt-4-turbo") -> LLMInterface:
#     backend = backend or os.environ.get("LLM_BACKEND", "mock")
#     if backend == "real":
#         return RealLLM(model=model)
#     return MockLLM()

def get_llm(
    backend: str | None = None,
    model: str = "gpt-4-turbo",
) -> LLMInterface:
    backend = backend or os.environ.get("LLM_BACKEND", "mock")

    if backend == "ollama":
        return OllamaLLM(model=model)          # Ollama only: no MockLLM fallback

    if backend == "real":
        return RealLLM(model=model)

    if backend == "mock":
        return MockLLM()

    raise ValueError(
        f"Unknown LLM backend: {backend}. "
        "Choose from 'mock', 'real', or 'ollama'."
    )