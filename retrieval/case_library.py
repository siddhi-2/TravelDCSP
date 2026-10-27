"""
Case library: a small corpus of (query, solved PlanState) pairs, built by
running the DCSP solver on a set of "training" queries. This stands in for
TravelPlanner's real 45 human-annotated training examples -- once you have
the real dataset (see data/query_loader.py), swap `build_case_library`'s
input to load real training queries and, ideally, real human reference
plans instead of solver-generated ones.

The case library is the retrieval corpus for CaseRetriever (retrieval/retriever.py).
"""

from __future__ import annotations
from dataclasses import dataclass
from data_model import Query, PlanState, cuisines_of
from plan_builder import build_initial_plan
from agents.concrete import BudgetAgent, HotelAgent, TransportAgent, FoodAgent, AttractionAgent
from solver.dcsp_solver import DCSPSolver
from data import sandbox


@dataclass
class Case:
    query: Query
    plan: PlanState
    converged: bool


def _fresh_agents():
    return [BudgetAgent(), HotelAgent(), TransportAgent(), FoodAgent(), AttractionAgent()]


def build_case_library(training_queries: list, llm, max_rounds: int = 10) -> list:
    """
    Solve each training query with the full DCSP negotiation pipeline and
    keep the result as a "case" -- regardless of whether it fully converged,
    since even a partially-fixed plan carries useful preference signal
    (which cuisine/room-type/transport-mode choices tend to work).
    """
    cases = []
    for q in training_queries:
        plan = build_initial_plan(q, llm)
        solver = DCSPSolver(_fresh_agents(), negotiate=True, max_rounds=max_rounds)
        result = solver.run(plan)
        cases.append(Case(query=q, plan=result.plan, converged=result.converged))
    return cases


def case_to_hint(case: Case) -> dict:
    """
    Distill a solved case down into the lightweight preference `hint` dict
    that agents' propose() methods know how to consume (see agents/base.py
    docstring). We extract TYPE-level preferences (which mode, which room
    type, which cuisine ordering) rather than copying exact items, since
    exact flights/hotels won't exist for a different origin/destination.
    """
    plan = case.plan
    hint: dict = {}

    # Most common transport mode used in the solved case
    modes = [dp.transportation.get("mode") for dp in plan.days_plan.values()
             if dp.transportation]
    if modes:
        hint["transport_mode"] = max(sorted(set(modes)), key=modes.count)

    # Most common room type used
    room_types = [dp.accommodation.get("room type") for dp in plan.days_plan.values()
                  if dp.accommodation]
    if room_types:
        hint["room_type"] = max(sorted(set(room_types)), key=room_types.count)

    # Cuisine ordering: which cuisines appeared, ranked by frequency
    cuisines = []
    for dp in plan.days_plan.values():
        for meal in ("breakfast", "lunch", "dinner"):
            item = getattr(dp, meal)
            if item and item.get("Cuisines"):
                cuisines.extend(sorted(cuisines_of(item)))
    if cuisines:
        ranked = sorted(sorted(set(cuisines)), key=lambda c: -cuisines.count(c))
        hint["cuisine_order"] = ranked

    return hint
