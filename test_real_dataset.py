"""
Test the ABT solver (and the negotiation solver) on the REAL TravelPlanner data.

    python test_real_dataset.py --data_dir data/travelplanner --db data/travelplanner/database
    python test_real_dataset.py --data_dir data/travelplanner --db data/travelplanner/database --split validation
    python test_real_dataset.py ... --limit 20                # quick smoke test
    python test_real_dataset.py ... --bruteforce 15           # also cross-check ABT vs exhaustive search

What it does
  1. PREFLIGHT   loads the six databases + the queries, prints sizes, and flags queries whose
                 origin / destination do not exist in the database (they cannot be solved).
  2. RUN         for every query: ABT (System G) and negotiation+restarts (System F).
                 One crashing query never stops the run (the error is recorded).
  3. VERIFY      every plan ABT calls "solved" is re-checked INDEPENDENTLY with the agents'
                 own constraint checks + the budget. Any disagreement is a bug and is listed.
  4. REPORT      pass rates overall / by difficulty / by split, ABT failure reasons,
                 messages & nogoods, F-vs-G agreement, runtime. Per-query CSV is saved.

Exit code is 1 if any verification failed (so you can use it in CI), else 0.
"""

from __future__ import annotations
import argparse
import csv
import itertools
import sys
import time
import traceback
from collections import Counter, defaultdict

from data.query_loader import load_travelplanner_dir
from data.real_sandbox import use_real_database, valid_places
from data import real_sandbox as RS
from llm.interface import MockLLM
from plan_builder import build_initial_plan
from run_experiment import fresh_agents
from solver.abt_solver import solve_abt, transport_domain, hotel_domain, food_domain
from solver.dcsp_solver import solve_with_restarts
from eval.metrics import score_plan


# --------------------------------------------------------------------------- #

def verify(plan):
    """Independent check: every agent's own constraint checks + TravelPlanner-style scoring."""
    hard = []
    for a in fresh_agents():
        hard += a.check(plan)
    return score_plan(plan, True, hard), hard


def brute_force_solvable(base, tiers=12):
    """Exhaustive search over all bundle combinations: is there ANY plan within budget?"""
    d = [transport_domain(base, tiers), hotel_domain(base, tiers), food_domain(base, tiers)]
    if any(not x for x in d):
        return None
    return any(t.cost + h.cost + f.cost <= base.query.budget + 1e-9 for t, h, f in itertools.product(*d))


def preflight(queries):
    cities, states = valid_places()
    print(f"  database: {len(cities)} cities, {len(states)} states")
    print(f"  accommodations: {sum(len(v) for v in RS._db.accommodations().values()):,} | "
          f"restaurants: {sum(len(v) for v in RS._db.restaurants().values()):,} | "
          f"attractions: {sum(len(v) for v in RS._db.attractions().values()):,} | "
          f"distance pairs: {len(RS._db.distances()):,} | flights: {len(RS._db.flights()):,}")
    by_split = Counter(getattr(q, "_split", "?") for q in queries)
    by_level = Counter(q._difficulty for q in queries)
    print(f"  queries: {len(queries)}  by split {dict(by_split)}  by level {dict(by_level)}")
    cons = Counter()
    for q in queries:
        for k in q.local_constraint:
            cons[k] += 1
    print(f"  constraint types used: {dict(cons)}")
    bad = []
    for i, q in enumerate(queries):
        k = max(1, q.visiting_city_number)
        problems = []
        if q.org not in cities:
            problems.append(f"origin '{q.org}' not in database")
        if k == 1 and q.dest not in cities:
            problems.append(f"destination city '{q.dest}' not in database")
        if k > 1 and q.dest not in states:
            problems.append(f"destination state '{q.dest}' not in database")
        if problems:
            bad.append((i, q, problems))
    if bad:
        print(f"  WARNING: {len(bad)} queries reference places missing from the database "
              f"(they will fail):")
        for i, q, p in bad[:10]:
            print(f"     query {i}: {'; '.join(p)}")
        if len(bad) > 10:
            print(f"     ... and {len(bad) - 10} more")
    else:
        print("  all query origins/destinations exist in the database")


# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", required=True, help="folder with train.csv / validation.csv")
    ap.add_argument("--db", required=True, help="folder with the six-source database")
    ap.add_argument("--split", default="all", choices=["train", "validation", "all"])
    ap.add_argument("--limit", type=int, default=None, help="only the first N queries")
    ap.add_argument("--seeds", type=int, default=1, help="ABT runs per query with different delivery orders")
    ap.add_argument("--tiers", type=int, default=12, help="ABT cost tiers per agent")
    ap.add_argument("--bruteforce", type=int, default=0,
                    help="cross-check ABT against exhaustive search on N queries (slower)")
    ap.add_argument("--out", default="real_dataset_results.csv")
    a = ap.parse_args()

    print("=" * 78 + "\nPREFLIGHT\n" + "=" * 78)
    t0 = time.time()
    try:
        use_real_database(a.db)
    except (FileNotFoundError, KeyError) as e:
        print(f"\nCould not load the database: {e}\n"
              f"Download it first:  python scripts/download_data.py")
        return 2
    print(f"  database loaded in {time.time() - t0:.1f}s")
    try:
        queries = load_travelplanner_dir(a.data_dir, a.split, a.limit)
    except FileNotFoundError as e:
        print(f"\nCould not load the queries: {e}")
        return 2
    preflight(queries)

    print("\n" + "=" * 78 + "\nRUN\n" + "=" * 78)
    llm = MockLLM()
    rows, errors, inconsistencies, bf_mismatch = [], [], [], []
    bf_left = a.bruteforce
    t_start = time.time()
    for i, q in enumerate(queries):
        row = {"idx": i, "split": getattr(q, "_split", "?"), "level": q._difficulty,
               "days": q.days, "cities": q.visiting_city_number, "people": q.people_number,
               "budget": q.budget, "constraints": "|".join(sorted(q.local_constraint))}
        try:
            base = build_initial_plan(q, llm)

            # ---- ABT (System G) ----
            results = [solve_abt(base.clone(), tiers=a.tiers, seed=s) for s in range(a.seeds)]
            r = results[0]
            if a.seeds > 1 and len({x.converged for x in results}) > 1:
                inconsistencies.append((i, "ABT outcome depends on message-delivery order",
                                        [x.converged for x in results]))
            sc, hard = verify(r.plan)
            if r.converged and not sc.final_pass:
                inconsistencies.append((i, "ABT said solved but independent check fails",
                                        [v.as_backprompt() for v in hard][:3]))
            row.update(abt_converged=r.converged, abt_final_pass=sc.final_pass,
                       abt_reason=r.failure_reason or "", abt_messages=r.messages,
                       abt_nogoods=r.nogoods_generated, abt_cost=round(r.plan.total_cost(), 2),
                       domain_sizes=str(r.domain_sizes))

            # ---- negotiation + restarts (System F) ----
            neg, _, _ = solve_with_restarts(base.clone, fresh_agents, 4, 10)
            nsc, _ = verify(neg.plan)
            row.update(neg_final_pass=nsc.final_pass, neg_cost=round(neg.plan.total_cost(), 2))

            # ---- optional exhaustive cross-check ----
            if bf_left > 0 and (not r.failure_reason or r.failure_reason == "proved_infeasible"):
                truth = brute_force_solvable(base, a.tiers)
                bf_left -= 1
                row["bruteforce_solvable"] = truth
                if truth is not None and truth != r.converged:
                    bf_mismatch.append((i, r.converged, truth))
        except Exception as e:                                   # noqa: BLE001
            errors.append((i, f"{type(e).__name__}: {e}", traceback.format_exc(limit=3)))
            row["error"] = f"{type(e).__name__}: {e}"
        rows.append(row)
        if (i + 1) % 25 == 0 or i + 1 == len(queries):
            print(f"  {i + 1}/{len(queries)} queries done ({time.time() - t_start:.1f}s)")

    # ----------------------------------------------------------------------- #
    ok = [r for r in rows if "error" not in r]
    n = max(1, len(ok))
    pct = lambda x, d=None: f"{100 * x / (d or n):5.1f}%"        # noqa: E731

    print("\n" + "=" * 78 + "\nRESULTS\n" + "=" * 78)
    abt_pass = sum(r["abt_final_pass"] for r in ok)
    neg_pass = sum(r["neg_final_pass"] for r in ok)
    print(f"  queries run without error : {len(ok)}/{len(rows)}")
    print(f"  ABT  (System G) final pass: {abt_pass}/{len(ok)} = {pct(abt_pass)}")
    print(f"  Negotiation+restarts (F)  : {neg_pass}/{len(ok)} = {pct(neg_pass)}")
    both = sum(r["abt_final_pass"] and r["neg_final_pass"] for r in ok)
    print(f"  both pass {both} | only ABT {sum(r['abt_final_pass'] and not r['neg_final_pass'] for r in ok)}"
          f" | only negotiation {sum(r['neg_final_pass'] and not r['abt_final_pass'] for r in ok)}"
          f" | neither {sum(not r['abt_final_pass'] and not r['neg_final_pass'] for r in ok)}")

    print("\n  By difficulty (final pass):")
    print(f"    {'level':<8}{'n':>5}{'ABT':>9}{'Negotiation':>14}")
    for lv in ("easy", "medium", "hard"):
        sub = [r for r in ok if r["level"] == lv]
        if sub:
            print(f"    {lv:<8}{len(sub):>5}{pct(sum(r['abt_final_pass'] for r in sub), len(sub)):>9}"
                  f"{pct(sum(r['neg_final_pass'] for r in sub), len(sub)):>14}")
    splits = sorted({r["split"] for r in ok})
    if len(splits) > 1:
        print("\n  By split (final pass):")
        for sp in splits:
            sub = [r for r in ok if r["split"] == sp]
            print(f"    {sp:<12}{len(sub):>5}{pct(sum(r['abt_final_pass'] for r in sub), len(sub)):>9}"
                  f"{pct(sum(r['neg_final_pass'] for r in sub), len(sub)):>14}")

    reasons = Counter((r["abt_reason"] or "solved") for r in ok)
    print("\n  ABT outcome / failure reasons:")
    for k, v in reasons.most_common():
        print(f"    {k:<28}{v:>5}  {pct(v)}")
    print("    (proved_infeasible = ABT PROVED no combination fits the budget;"
          " empty_domain = an agent has no option satisfying its own constraints)")

    msgs = [r["abt_messages"] for r in ok]
    ngs = [r["abt_nogoods"] for r in ok]
    print(f"\n  ABT cost: messages avg {sum(msgs)/n:.1f} / max {max(msgs, default=0)}"
          f" | nogoods avg {sum(ngs)/n:.2f} / max {max(ngs, default=0)}")
    print(f"  total time {time.time() - t_start:.1f}s ({(time.time() - t_start) / max(1, len(rows)):.2f}s/query)")

    # failure breakdown by constraint type (where does it break?)
    fails = defaultdict(lambda: [0, 0])
    for r in ok:
        for c in (r["constraints"].split("|") if r["constraints"] else ["(budget only)"]):
            fails[c][0] += 1
            fails[c][1] += (not r["abt_final_pass"])
    print("\n  ABT failure rate by constraint present in the query:")
    for c, (tot, f) in sorted(fails.items(), key=lambda kv: -kv[1][1] / max(1, kv[1][0])):
        print(f"    {c:<18} {f:>3}/{tot:<3} failed ({pct(f, tot)})")

    if a.bruteforce:
        print(f"\n  Exhaustive cross-check: {a.bruteforce - bf_left} queries, "
              f"mismatches: {len(bf_mismatch)} {bf_mismatch[:5] if bf_mismatch else ''}")

    with open(a.out, "w", newline="") as f:
        keys = sorted({k for r in rows for k in r})
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"\n  per-query results saved to {a.out}")

    problems = 0
    if errors:
        problems += len(errors)
        print("\n" + "=" * 78 + f"\nERRORS ({len(errors)} queries crashed)\n" + "=" * 78)
        for i, msg, tb in errors[:5]:
            print(f"  query {i}: {msg}\n{tb}")
        print("  Common cause: a column name or value in your download differs from what the loader expects."
              "\n  Paste the first error here and it can be fixed.")
    if inconsistencies:
        problems += len(inconsistencies)
        print("\n" + "=" * 78 + f"\nVERIFICATION FAILURES ({len(inconsistencies)})  <- these are bugs\n" + "=" * 78)
        for i, why, detail in inconsistencies[:10]:
            print(f"  query {i}: {why}: {detail}")
    if bf_mismatch:
        problems += len(bf_mismatch)
        print(f"\n  BRUTE-FORCE MISMATCHES: {bf_mismatch}")
    print("\n" + ("PASS: all checks consistent" if not problems else f"ATTENTION: {problems} problem(s) above"))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
