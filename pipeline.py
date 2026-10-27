"""
End-to-end pipeline with FULL TRACING, used by the UI (app.py).

Maps 1:1 to the architecture diagram:
  (1) user query -> (2) Prompt Generator -> LLM planner -> (3) LLM response -> Reformatter -> Plan Blackboard
  -> (4) constraint evaluation by the agents -> (5) negotiation / disagreement -> Coordinator
  -> (6) back prompt to the LLM (only when negotiation + restarts fail) -> (7) valid solution.

run_pipeline() returns a plain dict (JSON-friendly) so the UI -- or a test -- can render every step.
"""
from __future__ import annotations
import math

from data_model import transport_cost, lodging_cost, meal_cost
from plan_builder import build_initial_plan
from agents.concrete import BudgetAgent, HotelAgent, TransportAgent, FoodAgent, AttractionAgent
from solver.dcsp_solver import solve_with_restarts
from eval.metrics import score_plan

KINDS = ("transportation", "accommodation", "breakfast", "lunch", "dinner", "attraction")
AGENT_ORDER = ["budget", "transport", "hotel", "food", "attraction"]


def fresh_agents():
    return [BudgetAgent(), HotelAgent(), TransportAgent(), FoodAgent(), AttractionAgent()]


# --------------------------------------------------------------------------- #
# labels / snapshots
# --------------------------------------------------------------------------- #

def item_cost(kind, item, people) -> float:
    if not item:
        return 0.0
    if kind == "transportation":
        return transport_cost(item, people)
    if kind == "accommodation":
        return lodging_cost(item, people)
    if kind == "attraction":
        return 0.0
    return meal_cost(item, people)


def item_label(kind, item) -> str:
    if not item:
        return "-"
    if kind == "transportation":
        ref = item.get("Flight Number") or f"{item.get('distance_km', '')} km"
        return f"{item.get('mode')} {ref}"
    if kind == "accommodation":
        return f"{item.get('NAME')} [{item.get('room type')}]"
    if kind == "attraction":
        return "; ".join(a["Name"] for a in item) or "-"
    return f"{item.get('Name')} [{item.get('Cuisines')}]"


def _ident(v):
    if v is None:
        return None
    return tuple(sorted(str(x) for x in (v if isinstance(v, list) else [v])))


def _snapshot(agent, plan):
    out = {}
    for d, dp in plan.days_plan.items():
        for k in KINDS:
            if agent.owns(d, k):
                out[(d, k)] = getattr(dp, k)
    return out


def _entry(kind, item, people):
    return {"label": item_label(kind, item), "cost": round(item_cost(kind, item, people), 2)}


# --------------------------------------------------------------------------- #
# tracing: wrap each agent so every propose / check / concede is recorded
# --------------------------------------------------------------------------- #

class Tracer:
    def __init__(self, people: int):
        self.people = people
        self.events: list = []
        self.attempt = -1
        self.round = 0

    def new_attempt(self):
        self.attempt += 1
        self.round = 0
        self.events.append({"t": "attempt", "n": self.attempt})

    def wrap(self, agent):
        propose, check, concede = agent.propose, agent.check, agent.concede
        T, people = self, self.people

        def t_propose(pl, sbx, hint=None, attempt=0):
            before = _snapshot(agent, pl)
            r = propose(pl, sbx, hint=hint, attempt=attempt)
            slots = [{"day": d, "kind": k, **_entry(k, v, people)}
                     for (d, k), v in sorted(_snapshot(agent, pl).items()) if v != before.get((d, k))]
            T.events.append({"t": "proposal", "attempt": T.attempt, "agent": agent.name, "slots": slots,
                             "hint_used": bool(hint)})
            return r

        def t_check(pl):
            if agent.name == "budget":                  # the budget agent checks first in every round
                T.round += 1
                T.events.append({"t": "round", "attempt": T.attempt, "n": T.round,
                                 "total": round(pl.total_cost(), 2), "budget": pl.query.budget})
            vs = check(pl)
            for v in vs:
                T.events.append({"t": "flag", "attempt": T.attempt, "round": T.round, "agent": agent.name,
                                 "severity": v.severity, "critic": v.critic_name, "message": v.message,
                                 "asks": v.responsible_agent, "day": v.day})
            return vs

        def t_concede(pl, sbx, violation):
            before = _snapshot(agent, pl)
            ok = concede(pl, sbx, violation)
            after = _snapshot(agent, pl)
            changes = [(s, before.get(s), after[s]) for s in after if _ident(before.get(s)) != _ident(after[s])]
            if not changes:
                T.events.append({"t": "stuck", "attempt": T.attempt, "round": T.round, "agent": agent.name})
            for (d, k), old, new in changes:
                T.events.append({"t": "concede", "attempt": T.attempt, "round": T.round, "agent": agent.name,
                                 "day": d, "kind": k, "old": _entry(k, old, people), "new": _entry(k, new, people)})
            return ok

        agent.propose, agent.check, agent.concede = t_propose, t_check, t_concede
        return agent


# --------------------------------------------------------------------------- #
# Coordinator  (architecture box "Coordinator: consolidating concessions")
# --------------------------------------------------------------------------- #

class Coordinator:
    """Consolidates what the agents said in each round: which constraints were violated, who was asked
    to concede, what they conceded -- and tells the LLM planner (back prompt) when the agents cannot
    agree. The routing itself (violation -> responsible agent, cheapest concession first) is done by
    DCSPSolver; this class assembles the per-round report that the UI displays."""

    @staticmethod
    def consolidate(events: list) -> list:
        rounds = {}
        for e in events:
            if e["t"] == "round":
                rounds[(e["attempt"], e["n"])] = {"attempt": e["attempt"], "round": e["n"], "total": e["total"],
                                                  "budget": e["budget"], "flags": [], "concessions": [], "stuck": []}
            elif e["t"] in ("flag", "concede", "stuck"):
                r = rounds.get((e["attempt"], e["round"]))
                if r is not None:
                    {"flag": r["flags"], "concede": r["concessions"], "stuck": r["stuck"]}[e["t"]].append(e)
        out = sorted(rounds.values(), key=lambda r: (r["attempt"], r["round"]))
        for r in out:
            r["agreed"] = not r["flags"]
            r["asked"] = sorted({f["asks"] for f in r["flags"]})
        return out

    @staticmethod
    def back_prompt(violations: list) -> str:
        return "The agents could not reach an agreement. Unresolved violations:\n" + "\n".join(violations[:10])


# --------------------------------------------------------------------------- #

def plan_to_days(plan) -> list:
    n = plan.query.people_number
    days = []
    for d in sorted(plan.days_plan):
        dp = plan.days_plan[d]
        row = {"day": d, "city": dp.current_city, "items": []}
        for k in KINDS:
            v = getattr(dp, k)
            if v:
                row["items"].append({"kind": k, "label": item_label(k, v), "cost": round(item_cost(k, v, n), 2),
                                     "detail": _detail(k, v, n)})
        days.append(row)
    return days


def _detail(kind, v, n) -> str:
    """Currency-neutral extra info (amounts are shown by the UI in the user's currency)."""
    if kind == "transportation":
        if v.get("capacity"):
            return f"{v.get('distance_km', '')} km, {math.ceil(n / v['capacity'])} vehicle(s)"
        return f"{v.get('DepTime', '')}-{v.get('ArrTime', '')}, {n} traveller(s)"
    if kind == "accommodation":
        rooms = lodging_cost(v, n) / v["price"] if v.get("price") else 1
        return f"{rooms:g} room(s) per night; house rules: {v.get('house_rules') or 'none'}"
    if kind == "attraction":
        return ""
    rating = v.get("Aggregate Rating")
    return f"{n} traveller(s)" + (f"; rating {rating}" if rating not in (None, "") else "")


def cost_breakdown(plan) -> dict:
    n = plan.query.people_number
    out = {"Transport": 0.0, "Hotel": 0.0, "Food": 0.0}
    for dp in plan.days_plan.values():
        out["Transport"] += transport_cost(dp.transportation, n)
        out["Hotel"] += lodging_cost(dp.accommodation, n)
        out["Food"] += sum(meal_cost(getattr(dp, m), n) for m in ("breakfast", "lunch", "dinner"))
    return {k: round(v, 2) for k, v in out.items()}


def run_pipeline(query, llm, solver: str = "negotiate", max_rounds: int = 10, restarts: int = 4,
                 retriever=None) -> dict:
    out = {"solver": solver, "events": [], "abt": None}

    # (2)+(3) prompt generator -> LLM planner -> reformatter -> plan blackboard
    if retriever is not None and hasattr(llm, "prompt_generator"):
        from llm.prompt_generator import PromptGenerator
        llm.prompt_generator = PromptGenerator(retriever)
    base = build_initial_plan(query, llm)
    out["skeleton"] = [dp.current_city for _, dp in sorted(base.days_plan.items())]
    out["llm_prompt"] = getattr(llm, "last_prompt", None)
    out["prompt_sent"] = out["llm_prompt"] is not None
    if out["llm_prompt"] is None:                       # MockLLM sends nothing; still show what would be sent
        from llm.prompt_generator import PromptGenerator
        from llm.interface import cities_for
        out["llm_prompt"] = PromptGenerator(retriever).build(query, cities_for(query))
    out["llm_raw"] = getattr(llm, "last_raw", None)
    out["llm_cached"] = getattr(llm, "last_cached", False)
    out["llm_name"] = type(llm).__name__

    if solver == "abt":
        from solver.abt_solver import solve_abt
        res = solve_abt(base, tiers=12, seed=0)
        out["abt"] = {"messages": res.messages, "ok": res.ok_msgs, "nogood": res.nogood_msgs,
                      "add_link": res.link_msgs, "nogoods_generated": res.nogoods_generated,
                      "value_changes": res.value_changes, "domain_sizes": res.domain_sizes, "trace": res.trace[:400],
                      "trace_total": len(res.trace)}
        plan, converged, rounds, restarts_used, escalated = res.plan, res.converged, res.messages, 0, False
        reason = res.failure_reason
    else:
        T = Tracer(query.people_number)

        def agents_factory():
            T.new_attempt()
            return [T.wrap(a) for a in fresh_agents()]

        def escalate(plan, violations):               # (6) back prompt to the LLM planner
            T.events.append({"t": "backprompt", "text": Coordinator.back_prompt(violations)})
            new = llm.replan_with_backprompt(plan, violations)
            T.events.append({"t": "backprompt_result", "revised": new is not None})
            return new

        result, restarts_used, escalated = solve_with_restarts(
            plan_factory=base.clone, agents_factory=agents_factory, max_restarts=restarts,
            max_rounds=max_rounds, retriever=retriever, escalate=escalate)
        plan, converged, rounds, reason = result.plan, result.converged, result.rounds_used, result.failure_reason
        out["events"] = T.events

    hard = []
    for ag in fresh_agents():
        hard += ag.check(plan)
    sc = score_plan(plan, True, hard)
    out.update(
        rounds_view=Coordinator.consolidate(out["events"]),
        converged=converged, rounds=rounds, restarts_used=restarts_used, escalated=escalated, reason=reason,
        total=round(plan.total_cost(), 2), budget=query.budget, days=plan_to_days(plan),
        breakdown=cost_breakdown(plan), plan_json=plan.to_json(),
        score={"commonsense": [sc.commonsense_passed, sc.commonsense_total],
               "hard": [sc.hard_passed, sc.hard_total], "final_pass": sc.final_pass},
        remaining=[v.as_backprompt() for v in hard][:8])
    return out