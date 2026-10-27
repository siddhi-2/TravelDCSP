"""
DCSP negotiation solver.

Implements the propose -> check -> conflict-signal -> concede loop described
in the project methodology: agents don't just independently fail a plan --
when Agent A's constraint is violated by a variable owned by Agent B, A sends
a targeted conflict signal to B, and B concedes to its next-best alternative.

This is the module you'll want to extend for the M.Tech-worthy additions:
  - swap `select_negotiation_order` for a learned-priority policy (Tier 2 idea)
  - log every round for the convergence-analysis / rounds-to-convergence metric
  - the `NoNegotiation` variant below gives you System C (DCSP, no negotiation)
    for the ablation study for free, by disabling the concede step.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
from data_model import PlanState, ConstraintViolation
from data import sandbox


@dataclass
class SolverResult:
    plan: PlanState
    converged: bool
    rounds_used: int
    violations_log: list = field(default_factory=list)   # list of (round, [violations]) for analysis
    failure_reason: Optional[str] = None                 # "domain_exhaustion" | "max_rounds" | None


class DCSPSolver:
    """
    agents: list[ConstraintAgent]
    negotiate: bool -- if False, agents are checked independently and NOT
                       allowed to concede to each other's conflict signals
                       (this reproduces "System C: DCSP, parallel critics,
                       no negotiation" from the ablation design).
    max_rounds: bounded backtracking limit.
    """

    def __init__(self, agents: list, negotiate: bool = True, max_rounds: int = 10, retriever=None):
        """
        retriever: an optional CaseRetriever (retrieval/retriever.py). When
        provided, the initial propose() call for every agent is seeded with
        a `hint` distilled from the most similar previously-solved query,
        instead of a cold-start (median-cost) guess. This is the RAG-seeded
        negotiation variant (System E in the ablation design) -- everything
        else about the negotiation loop is unchanged.
        """
        self.agents = agents
        self.negotiate = negotiate
        self.max_rounds = max_rounds
        self.retriever = retriever
        self._by_name = {a.name: a for a in agents}

    # --------------------------------------------------------------------- #

    def run(self, plan: PlanState, attempt: int = 0) -> SolverResult:
        hint = self.retriever.retrieve_hint(plan.query) if self.retriever else None

        # 1. Initial proposals from every agent (RAG-seeded if a retriever was given).
        #    `attempt` > 0 shifts every agent's starting candidate (multi-start).
        for agent in self.agents:
            agent.propose(plan, sandbox, hint=hint, attempt=attempt)

        log = []
        for rnd in range(1, self.max_rounds + 1):
            violations = self._check_all(plan)
            log.append((rnd, [v.as_backprompt() for v in violations]))

            if not violations:
                return SolverResult(plan=plan, converged=True, rounds_used=rnd, violations_log=log)

            if not self.negotiate:
                # No-negotiation ablation: just report and stop after one check pass.
                return SolverResult(
                    plan=plan, converged=False, rounds_used=rnd, violations_log=log,
                    failure_reason="no_negotiation_unresolved",
                )

            # 2. Negotiation: route each violation to its responsible agent and ask it to concede.
            progressed = False
            for v in self._order_violations(violations):
                agent = self._by_name.get(v.responsible_agent)
                if agent is None:
                    continue
                conceded = agent.concede(plan, sandbox, v)
                if conceded:
                    progressed = True

            if not progressed:
                return SolverResult(
                    plan=plan, converged=False, rounds_used=rnd, violations_log=log,
                    failure_reason="domain_exhaustion",
                )

        return SolverResult(
            plan=plan, converged=False, rounds_used=self.max_rounds, violations_log=log,
            failure_reason="max_rounds",
        )

    # --------------------------------------------------------------------- #

    def _check_all(self, plan: PlanState) -> list:
        violations = []
        for agent in self.agents:
            violations.extend(agent.check(plan))
        return violations

    def _order_violations(self, violations: list) -> list:
        """
        Negotiation order: higher-priority responsible agents get asked to
        concede FIRST is the wrong intuition -- priority here means "how hard
        is it for this agent to move" (BudgetAgent priority=1.0 => never
        concedes; FoodAgent priority=0.3 => concedes easily). We therefore
        resolve violations whose responsible agent has the LOWEST priority
        first, since those concessions are cheapest and most likely to
        resolve the conflict without cascading further violations.
        """
        def agent_priority(v):
            agent = self._by_name.get(v.responsible_agent)
            return agent.priority if agent else 0.5

        return sorted(violations, key=agent_priority)


# --------------------------------------------------------------------------- #
# Multi-start wrapper
# --------------------------------------------------------------------------- #

def solve_with_restarts(plan_factory, agents_factory, max_restarts: int = 4, max_rounds: int = 10,
                        retriever=None, escalate=None):
    """
    Local negotiation can get stuck in a local optimum: starting from the median
    candidate, every agent's remaining concessions may be exhausted although a
    solution exists from a different starting point. This wrapper restarts the
    WHOLE negotiation from a different initial assignment (attempt = 0..k) with
    fresh agents (clean `_tried` memory), and keeps the best result.

    plan_factory()   -> new empty PlanState skeleton (re-uses the LLM's skeleton;
                        no extra LLM call needed, just a deep copy in practice)
    agents_factory() -> new list of agents
    escalate         -> optional callable(plan, violations) -> bool. If every
                        restart fails, the LLM is asked ONCE to re-plan using the
                        unresolved violations as a back-prompt (the "(6) Back
                        Prompt" arrow in the architecture diagram); the solver is
                        then run again on the LLM's revised skeleton.
    Returns (SolverResult, restarts_used, escalated: bool)
    """
    best = None
    for k in range(max_restarts + 1):
        plan = plan_factory()
        agents = agents_factory()
        res = DCSPSolver(agents, True, max_rounds, retriever).run(plan, attempt=k)
        if best is None or len(res.violations_log[-1][1]) < len(best.violations_log[-1][1]):
            best = res
        if res.converged:
            return res, k, False

    if escalate is not None:
        new_plan = escalate(best.plan, best.violations_log[-1][1])
        if new_plan is not None:
            res = DCSPSolver(agents_factory(), True, max_rounds, retriever).run(new_plan)
            if res.converged:
                return res, max_restarts, True
            best = res if len(res.violations_log[-1][1]) < len(best.violations_log[-1][1]) else best
    return best, max_restarts, False
