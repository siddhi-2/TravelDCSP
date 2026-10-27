"""
Base class for all constraint agents in the DCSP negotiation framework.

Each agent owns a subset of the plan's variables (e.g. FoodAgent owns
breakfast/lunch/dinner variables for every day) and knows how to:
  - check(plan)      -> is the current assignment acceptable to this agent?
  - propose(...)      -> produce an initial candidate assignment
  - concede(...)       -> given a conflict, produce the *next-best* alternative
                          from its own domain (or signal exhaustion)

Agents never talk to the LLM directly during solving — they only see
structured PlanState / Query objects. This keeps the solving step
deterministic and LLM-free, per the project's core design goal.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from data_model import PlanState, Query, ConstraintViolation, Variable


class ConstraintAgent:
    name: str = "base"
    severity: str = "hard"     # "hard" | "commonsense"
    priority: float = 0.5      # higher = concedes LAST (i.e. harder to move); used for negotiation ordering

    def owns(self, day: int, kind: str) -> bool:
        """Does this agent own the variable (day, kind)?"""
        raise NotImplementedError

    def check(self, plan: PlanState) -> list[ConstraintViolation]:
        """Return a list of violations this agent detects in the current plan.
        Empty list = agent is satisfied."""
        raise NotImplementedError

    def propose(self, plan: PlanState, sandbox, hint: dict | None = None, attempt: int = 0) -> None:
        """Fill in this agent's variables in `plan` with an initial assignment.
        Mutates plan in place.

        `hint`, when provided, is a dict produced by a CaseRetriever (see
        retrieval/retriever.py) describing preferences distilled from the
        most similar previously-solved query -- e.g. {"transport_mode":
        "flight", "cuisine_order": [...], "room_type": "Entire Room"}.
        Agents are free to ignore hint entirely (default behavior); using it
        is what turns cold-start proposal into RAG-seeded proposal.

        `attempt`, when > 0, signals "this is a re-proposal after a failed
        check" (used by the LLM-Modulo baseline's whole-plan regeneration) --
        agents that support it should return a genuinely DIFFERENT candidate
        each time attempt increases, so the baseline gets a fair chance to
        explore alternatives across its iteration budget rather than
        deterministically regenerating the exact same plan every round.
        """
        raise NotImplementedError

    def concede(self, plan: PlanState, sandbox, violation: ConstraintViolation) -> bool:
        """
        Given a conflict directed at this agent (it is the `responsible_agent`
        in `violation`), try the next-best alternative from this agent's domain.

        Returns True if a new candidate was assigned (plan mutated in place),
        False if this agent has exhausted its domain and cannot concede further.
        """
        raise NotImplementedError

    def domain_exhausted(self, plan: PlanState) -> bool:
        raise NotImplementedError
