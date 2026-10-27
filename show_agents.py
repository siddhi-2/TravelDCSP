"""
Watch the four agents (+ the Budget agent) work, step by step.

  python show_agents.py --data_dir data/travelplanner --db data/travelplanner/database --split validation --idx 5
  python show_agents.py --db data/travelplanner/database --source Seattle --dest Yakima --days 3 --people 2 --budget 1500
  python show_agents.py ...  --restart 2        # start from a different initial assignment

For one query it prints
  1. every agent's FIRST proposal (what it picks, at what cost)
  2. each negotiation round: which agent complains about which constraint
  3. every concession: the responsible agent, the slot it changes and old -> new item (+cost)
  4. the final itinerary, total cost vs budget, and an independent pass/fail check
Agents: transport, hotel, food, attraction (they own the variables) and budget (owns no variable;
it only checks the total and tells the agent that owns the most expensive item to concede).
"""
from __future__ import annotations
import argparse
import math

from data_model import Query, transport_cost, lodging_cost, meal_cost
from data import sandbox
from llm.interface import get_llm
from plan_builder import build_initial_plan
from agents.concrete import BudgetAgent, HotelAgent, TransportAgent, FoodAgent, AttractionAgent
from solver.dcsp_solver import DCSPSolver
from eval.metrics import score_plan

KINDS = ("transportation", "accommodation", "breakfast", "lunch", "dinner", "attraction")


def label(kind, item, people):
    if not item:
        return "-"
    if kind == "transportation":
        ref = item.get("Flight Number") or f"{item.get('distance_km', '')} km"
        return f"{item.get('mode')} {ref} (${transport_cost(item, people):,.0f})"
    if kind == "accommodation":
        return f"{item.get('NAME')} [{item.get('room type')}] (${lodging_cost(item, people):,.0f}/night)"
    if kind == "attraction":
        return "; ".join(a["Name"] for a in item) or "-"
    return f"{item.get('Name')} [{item.get('Cuisines')}] (${meal_cost(item, people):,.0f})"


def snapshot(agent, plan):
    out = {}
    for d, dp in plan.days_plan.items():
        for k in KINDS:
            if agent.owns(d, k):
                out[(d, k)] = getattr(dp, k)
    return out


def ident(v):
    return repr(v) if v is None else repr(sorted((str(x) for x in (v if isinstance(v, list) else [v]))))


def attach_tracing(agents, plan, people):
    def wrap(agent):
        propose, check, concede = agent.propose, agent.check, agent.concede

        def t_propose(pl, sbx, hint=None, attempt=0):
            before = snapshot(agent, pl)
            r = propose(pl, sbx, hint=hint, attempt=attempt)
            print(f"\n  [{agent.name.upper()}] first proposal")
            for (d, k), v in sorted(snapshot(agent, pl).items()):
                if v != before.get((d, k)):
                    print(f"      day {d} {k:<14} -> {label(k, v, people)}")
            return r

        def t_check(pl):
            vs = check(pl)
            for v in vs:
                print(f"      [{agent.name.upper()}] flags ({v.severity}/{v.critic_name}): {v.message}"
                      f"   -> asks {v.responsible_agent.upper()}")
            return vs

        def t_concede(pl, sbx, violation):
            owner_before = snapshot(agent, pl)
            ok = concede(pl, sbx, violation)
            after = snapshot(agent, pl)
            changes = [(s, owner_before[s], after[s]) for s in after if ident(owner_before.get(s)) != ident(after[s])]
            if not ok and not changes:
                print(f"      [{agent.name.upper()}] cannot concede (no cheaper/other option left)")
            for (d, k), old, new in changes:
                print(f"      [{agent.name.upper()}] concedes day {d} {k}: {label(k, old, people)}  =>  {label(k, new, people)}")
            return ok

        agent.propose, agent.check, agent.concede = t_propose, t_check, t_concede
    for a in agents:
        wrap(a)


def make_query(a):
    if a.data_dir:
        from data.query_loader import load_travelplanner_dir
        qs = load_travelplanner_dir(a.data_dir, a.split)
        return qs[a.idx]
    lc = {}
    if a.cuisine:
        lc["cuisine"] = a.cuisine
    if a.room_type:
        lc["room type"] = a.room_type
    if a.transport:
        lc["transportation"] = a.transport
    return Query(org=a.source, dest=a.dest, days=a.days, visiting_city_number=a.cities,
                 date=[f"2022-03-{d:02d}" for d in range(1, a.days + 1)], people_number=a.people,
                 budget=a.budget, query_text=f"{a.days}-day trip {a.source}->{a.dest}", local_constraint=lc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir"); ap.add_argument("--split", default="validation")
    ap.add_argument("--idx", type=int, default=0, help="which query of the split")
    ap.add_argument("--db", help="REAL TravelPlanner database folder")
    ap.add_argument("--source"); ap.add_argument("--dest")
    ap.add_argument("--days", type=int, default=3); ap.add_argument("--people", type=int, default=2)
    ap.add_argument("--budget", type=float, default=1500); ap.add_argument("--cities", type=int, default=1)
    ap.add_argument("--cuisine", nargs="*"); ap.add_argument("--room_type")
    ap.add_argument("--transport", choices=["No flight", "No self-driving"])
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--restart", type=int, default=0, help="initial-assignment offset (multi-start)")
    ap.add_argument("--llm", default="mock", choices=["mock", "real", "ollama"])
    ap.add_argument("--model", default="llama3.2:1b")
    a = ap.parse_args()
    if a.db:
        from data.real_sandbox import use_real_database
        use_real_database(a.db)
    if not a.data_dir and not (a.source and a.dest):
        ap.error("give --data_dir (+ --idx) or --source and --dest")

    q = make_query(a)
    people = q.people_number
    print("=" * 78)
    print(f"QUERY: {q.org} -> {q.dest} | {q.days} days | {people} people | budget ${q.budget:,.0f} "
          f"| cities {q.visiting_city_number}")
    print(f"constraints: {q.local_constraint or 'none'}")
    print("=" * 78)

    llm = get_llm(a.llm, a.model)
    plan = build_initial_plan(q, llm)
    print("LLM skeleton (day -> city):", [dp.current_city for _, dp in sorted(plan.days_plan.items())])

    agents = [BudgetAgent(), HotelAgent(), TransportAgent(), FoodAgent(), AttractionAgent()]
    attach_tracing(agents, plan, people)
    solver = DCSPSolver(agents, negotiate=True, max_rounds=a.rounds)

    orig_check_all = solver._check_all
    state = {"rnd": 0}

    def traced_check_all(pl):
        state["rnd"] += 1
        print(f"\n--- ROUND {state['rnd']}: every agent checks the plan "
              f"(current total ${pl.total_cost():,.0f} / budget ${q.budget:,.0f}) ---")
        vs = orig_check_all(pl)
        if not vs:
            print("      all agents satisfied")
        else:
            print(f"   -> negotiation: responsible agents concede ({len(vs)} violation(s))")
        return vs

    solver._check_all = traced_check_all
    print("\n=== STEP 1: initial proposals ===")
    res = solver.run(plan, attempt=a.restart)

    print("\n" + "=" * 78 + "\nFINAL ITINERARY\n" + "=" * 78)
    for d in sorted(res.plan.days_plan):
        dp = res.plan.days_plan[d]
        print(f"Day {d}: {dp.current_city}")
        for k in KINDS:
            v = getattr(dp, k)
            if v:
                print(f"    {k:<14}: {label(k, v, people)}")
    hard = []
    for ag in [BudgetAgent(), HotelAgent(), TransportAgent(), FoodAgent(), AttractionAgent()]:
        hard += ag.check(res.plan)
    sc = score_plan(res.plan, True, hard)
    cost = res.plan.total_cost()
    print("\n" + "=" * 78)
    print(f"converged: {res.converged} in {res.rounds_used} round(s)"
          f"{'' if res.converged else ' | reason: ' + str(res.failure_reason)}")
    print(f"total cost ${cost:,.0f} / budget ${q.budget:,.0f} ({'within' if cost <= q.budget else 'OVER'})")
    print(f"independent check -> commonsense {sc.commonsense_passed}/{sc.commonsense_total}, "
          f"hard {sc.hard_passed}/{sc.hard_total}, FINAL PASS: {sc.final_pass}")
    if not res.converged:
        print("still violated:", [v.as_backprompt() for v in hard][:5])


if __name__ == "__main__":
    main()