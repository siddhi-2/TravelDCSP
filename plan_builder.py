"""
Builds the initial (empty) PlanState skeleton from a Query, using the LLM
only to decide the day-by-day city sequence. All actual assignments
(transportation, food, hotel choices) are left for the agents to fill in --
this keeps the LLM out of the constraint-solving step, per the project's
core design.
"""

from __future__ import annotations
from data_model import Query, PlanState, DayPlan
from llm.interface import LLMInterface


def build_initial_plan(query: Query, llm: LLMInterface) -> PlanState:
    skeleton = llm.generate_initial_skeleton(query)
    plan = PlanState(query=query)
    for day, city in enumerate(skeleton, start=1):
        plan.days_plan[day] = DayPlan(day=day, current_city=city)
    return plan
