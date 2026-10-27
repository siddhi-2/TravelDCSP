"""
Interactive demo: give your own constraints, see the negotiated itinerary.

Usage (interactive -- it asks you questions):
    python demo.py

Usage (one line):
    python demo.py --source Springfield --dest Fairview --days 3 --people 2 --budget 3000
    python demo.py --source Springfield --dest Fairview --days 5 --people 4 --budget 6000 \
                   --cuisine Italian Indian --room_type "Entire Room" --transport "No flight"

Shows: the parsed query, every negotiation round (who violated what, who
conceded), the final day-by-day plan, the cost vs budget, and the
TravelPlanner-style verdict.

Note: with the built-in synthetic sandbox every city name works (data is
generated from the name). With the real TravelPlanner database, use cities
that exist in it.
"""

from __future__ import annotations
import argparse

from data_model import Query, transport_cost, lodging_cost
from llm.interface import get_llm
from plan_builder import build_initial_plan
from agents.concrete import BudgetAgent, HotelAgent, TransportAgent, FoodAgent, AttractionAgent
from solver.dcsp_solver import solve_with_restarts
from eval.metrics import score_plan

ROOM_TYPES = ["Entire Room", "Private Room", "Shared Room", "Not Shared Room"]
ROOM_RULES = ["No parties", "No smoking", "No children under 10", "No pets", "No visitors"]


INR_SCALE = 12   # synthetic sandbox units -> illustrative rupees (flight ~Rs 1,000-6,000, meal ~Rs 100-700)
CUR = {"sym": "$", "scale": 1}


def use_currency(code: str):
    """Show everything in INR: scale the sandbox's synthetic prices by INR_SCALE.
    Prices stay illustrative -- real fares need the real database."""
    if code != "INR":
        return
    from data import sandbox
    CUR["sym"], CUR["scale"] = "Rs ", INR_SCALE

    def wrap(fn):
        def inner(*a, **k):
            out = fn(*a, **k)
            for item in out:
                for key in ("cost", "price"):
                    if key in item:
                        item[key] = round(item[key] * INR_SCALE, 2)
            return out
        return inner

    for name in ("search_flights", "search_self_driving", "search_restaurants", "search_accommodations"):
        setattr(sandbox, name, wrap(getattr(sandbox, name)))


def money(x):
    return f"{CUR['sym']}{x:,.2f}"


def fresh_agents():
    return [BudgetAgent(), HotelAgent(), TransportAgent(), FoodAgent(), AttractionAgent()]


def ask(prompt, default=None, cast=str):
    suffix = f" [{default}]" if default is not None else ""
    while True:
        raw = input(f"{prompt}{suffix}: ").strip()
        if not raw and default is not None:
            return default
        if not raw:
            continue
        try:
            return cast(raw)
        except ValueError:
            print("  Please enter a valid value.")


def interactive_args() -> argparse.Namespace:
    print("\n=== Travel planner: enter your constraints ===")
    a = argparse.Namespace()
    a.source = ask("Source city")
    a.dest = ask("Destination city")
    a.days = ask("Number of days", 3, int)
    a.people = ask("Number of people", 1, int)
    a.budget = ask("Total budget (in the currency chosen below)", 2000, float)
    print("Optional constraints (press Enter to skip)")
    c = input("  Cuisines wanted (comma separated, e.g. Italian,Indian): ").strip()
    a.cuisine = [x.strip() for x in c.split(",") if x.strip()] or None
    rt = input(f"  Room type {ROOM_TYPES}: ").strip()
    a.room_type = rt or None
    rr = input(f"  Room rule to avoid {ROOM_RULES}: ").strip()
    a.room_rule = rr or None
    tr = input("  Transport preference ('No flight' / 'No self-driving'): ").strip()
    a.transport = tr or None
    a.rounds, a.restarts, a.llm, a.model = 10, 4, "mock", "gpt-4-turbo"
    a.currency = ask("Currency (USD/INR)", "INR", str).upper()
    return a


def build_query(a) -> Query:
    lc = {}
    if a.cuisine:
        lc["cuisine"] = a.cuisine
    if a.room_type:
        lc["room type"] = a.room_type
    if a.room_rule:
        lc["room rule"] = a.room_rule
    if a.transport:
        lc["transportation"] = a.transport
    text = (f"Plan a {a.days}-day trip from {a.source} to {a.dest} for {a.people} "
            f"people, budget {a.budget:.0f}.")
    return Query(org=a.source, dest=a.dest, days=a.days, visiting_city_number=1,
                 date=[f"2022-03-{d:02d}" for d in range(1, a.days + 1)],
                 people_number=a.people, budget=a.budget, query_text=text,
                 local_constraint=lc)


def print_itinerary(plan):
    q = plan.query
    n = q.people_number
    print(f"Trip: {q.org} -> {q.dest}  ({q.days} days, {n} people)")
    for day in sorted(plan.days_plan):
        dp = plan.days_plan[day]
        print(f"\nDay {day}: {dp.current_city}")
        t = dp.transportation
        if t:
            ref = t.get("Flight Number", f"{t.get('distance_km', '')} km")
            if t.get("capacity"):
                import math
                veh = math.ceil(n / t["capacity"])
                print(f"  Transport    : {t['mode']} ({ref})  {money(t['cost'])} x {veh} vehicle(s) = {money(transport_cost(t, n))}")
            else:
                print(f"  Transport    : {t['mode']} ({ref})  {money(t['cost'])} x {n} = {money(transport_cost(t, n))}")
        for meal in ("breakfast", "lunch", "dinner"):
            mm = getattr(dp, meal)
            if mm:
                print(f"  {meal.capitalize():<13}: {mm['Name']} [{mm['Cuisines']}]  {money(mm['cost'])} x {n}")
        if dp.attraction:
            print(f"  Attractions  : " + "; ".join(x['Name'] for x in dp.attraction))
        h = dp.accommodation
        if h:
            rooms = lodging_cost(h, n) / h["price"] if h["price"] else 1
            print(f"  Hotel        : {h['NAME']} ({h['room type']})  {money(h['price'])}/night x {rooms:g} room(s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source"); ap.add_argument("--dest")
    ap.add_argument("--days", type=int, default=3)
    ap.add_argument("--people", type=int, default=1)
    ap.add_argument("--budget", type=float, default=2000)
    ap.add_argument("--cuisine", nargs="*", default=None)
    ap.add_argument("--room_type", default=None, choices=[None] + ROOM_TYPES)
    ap.add_argument("--room_rule", default=None, choices=[None] + ROOM_RULES)
    ap.add_argument("--transport", default=None, choices=[None, "No flight", "No self-driving"])
    ap.add_argument("--currency", default="USD", choices=["USD", "INR"])
    ap.add_argument("--db", default=None, help="path to the REAL TravelPlanner database folder")
    ap.add_argument("--solver", default="negotiate", choices=["negotiate", "abt"],
                    help="negotiate = round-based DCSP negotiation; abt = true Asynchronous Backtracking")
    ap.add_argument("--abt_trace", type=int, default=30, help="how many ABT messages to print")
    ap.add_argument("--list_cities", action="store_true",
                    help="with --db: print the valid cities and states, then exit")
    ap.add_argument("--cities", type=int, default=1,
                    help="number of cities to visit (2-3 => --dest must be a US STATE, as in TravelPlanner)")
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--restarts", type=int, default=4)
    ap.add_argument("--llm", default="mock", choices=["mock", "real","ollama"])
    ap.add_argument("--model", default="gpt-4-turbo")
    a = ap.parse_args()
    if a.db:
        from data.real_sandbox import use_real_database, valid_places
        use_real_database(a.db)
        if a.list_cities:
            cities, states = valid_places()
            print(f"{len(cities)} cities, {len(states)} states in the database\n")
            for st in sorted(states):
                print(f"{st}: {', '.join(sorted(states[st])[:12])}{' ...' if len(states[st]) > 12 else ''}")
            return
    if not a.source or not a.dest:
        a = interactive_args()
        a.db = getattr(a, "db", None) or (ap.parse_args().db)
    if a.days < 2:
        raise SystemExit("Need at least 2 days (outbound + return).")

    if getattr(a, "db", None):
        if a.currency == "INR":
            raise SystemExit("INR scaling only applies to the synthetic sandbox; the real database is in USD.")
        from data.real_sandbox import valid_places
        import difflib
        cities, states = valid_places()
        k = max(1, getattr(a, "cities", 1))
        for label, name, ok in (("source", a.source, a.source in cities),
                                ("destination", a.dest, (a.dest in cities) if k == 1 else (a.dest in states))):
            if not ok:
                pool = list(cities) if (label == "source" or k == 1) else list(states)
                close = difflib.get_close_matches(name, pool, n=5)
                raise SystemExit(
                    f"'{name}' is not a valid {label}"
                    f"{' state (use a STATE when --cities > 1)' if label == 'destination' and k > 1 else ''}."
                    f"\nDid you mean: {close or sorted(pool)[:8]}\nRun with --list_cities to see all valid names.")
    
    use_currency(a.currency)
    query = build_query(a)
    query.visiting_city_number = max(1, getattr(a, "cities", 1))
    llm = get_llm(backend=a.llm, model=a.model)

    print("\n" + "=" * 70)
    print("INPUT")
    print("=" * 70)
    print(f"  {query.org} -> {query.dest} | {query.days} days | {query.people_number} people "
          f"| budget {money(query.budget)}")
    print(f"  extra constraints: {query.local_constraint or 'none'}")

    # Step 1: LLM makes the skeleton (the ONLY LLM call)
    base = build_initial_plan(query, llm)
    # Step 2: agents solve it (no LLM)
    abt = None
    if getattr(a, "solver", "negotiate") == "abt":
        from solver.abt_solver import solve_abt
        abt = solve_abt(base, tiers=12, seed=0)
        result = abt
        restarts, escalated = 0, False
    else:
        result, restarts, escalated = solve_with_restarts(
            plan_factory=base.clone, agents_factory=fresh_agents,
            max_restarts=a.restarts, max_rounds=a.rounds,
            escalate=(lambda p, v: llm.replan_with_backprompt(p, v)))

    print("\n" + "=" * 70)
    print("ABT MESSAGE LOG" if abt else "NEGOTIATION LOG (last attempt)")
    print("=" * 70)
    if abt:
        print(f"  domain sizes (candidate bundles per agent): {abt.domain_sizes}")
        print(f"  priority order: transport > hotel > food > attraction   (budget checked by 'food')")
        for line in abt.trace[: a.abt_trace]:
            print(f"    {line}")
        if len(abt.trace) > a.abt_trace:
            print(f"    ... {len(abt.trace) - a.abt_trace} more messages")
        print(f"  messages: {abt.messages} (ok? {abt.ok_msgs}, nogood {abt.nogood_msgs}, add_link {abt.link_msgs}) | "
              f"nogoods generated: {abt.nogoods_generated} | value changes: {abt.value_changes}")
    else:
        for rnd, msgs in result.violations_log:
            if not msgs:
                print(f"  Round {rnd}: all agents satisfied")
            else:
                print(f"  Round {rnd}: {len(msgs)} conflict(s)")
                for m in msgs:
                    print(f"      - {m}")
        print(f"  restarts used: {restarts} | LLM escalation: {escalated}")

    print("\n" + "=" * 70)
    print("FINAL ITINERARY")
    print("=" * 70)
    print_itinerary(result.plan)

    hard = []
    for ag in fresh_agents():
        hard += ag.check(result.plan)
    score = score_plan(result.plan, delivered=True, hard_violations_from_agents=hard)

    cost = result.plan.total_cost()
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    print(f"  Converged            : {result.converged}")
    print(f"  Total cost / budget  : {money(cost)} / {money(query.budget)} "
          f"({'within' if cost <= query.budget else 'OVER'} budget)")
    print(f"  Commonsense passed   : {score.commonsense_passed}/{score.commonsense_total}")
    print(f"  Hard passed          : {score.hard_passed}/{score.hard_total}")
    print(f"  Final pass           : {score.final_pass}")
    if not result.converged:
        print(f"  Why it failed        : {result.failure_reason}")
        if result.failure_reason == "proved_infeasible":
            print("  ABT PROVED that no combination of options fits the budget (empty nogood). "
                  "Raise the budget or relax a constraint.")
        else:
            print("  Tip: raise the budget, or relax a constraint -- the agents ran out of cheaper options.")


if __name__ == "__main__":
    main()
