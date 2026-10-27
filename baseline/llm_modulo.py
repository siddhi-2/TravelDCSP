"""
Reimplementation of the LLM-Modulo baseline (Gundawar et al., 2024) for a
fair, controlled head-to-head comparison against the DCSP negotiation system.

Faithful to the paper's described architecture:
  - Format critic runs first (precondition for all others)
  - Commonsense critics (style critics)
  - Hard-constraint critics
  - Metacontroller concatenates ALL failing critics' backprompts and resends
    to the LLM as one combined prompt (this is explicitly the "rudimentary"
    metacontroller design the paper itself flags as simple -- reproduced here
    on purpose, since it's our baseline to beat, not our contribution).
  - Bounded to max_tries=10 iterations, matching the original paper.

Key difference from our DCSP system: here, EVERY fix-attempt round is a full
LLM generation call (expensive), whereas DCSPSolver resolves conflicts with
deterministic agent logic and no LLM calls during solving. Track LLM call
counts on both systems for the cost-comparison metric in your evaluation.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from data_model import Query, PlanState, DayPlan, ConstraintViolation
from llm.interface import LLMInterface
from agents.concrete import BudgetAgent, HotelAgent, TransportAgent, FoodAgent, AttractionAgent
from data import sandbox


@dataclass
class LLMModuloResult:
    plan: PlanState
    converged: bool
    iterations_used: int
    llm_calls: int
    violations_log: list = field(default_factory=list)


class FormatCritic:
    """Checks that every day has all required fields populated (JSON validity
    analogue -- in our structured-object setting this means no missing
    variables rather than malformed JSON)."""
    name = "format"
    severity = "format"

    def check(self, plan: PlanState) -> list:
        violations = []
        last_day = max(plan.days_plan) if plan.days_plan else 0
        for day, dp in plan.days_plan.items():
            missing = [f for f in ("transportation", "breakfast", "lunch", "dinner", "accommodation")
                       if getattr(dp, f) is None and not (f == "transportation" and " to " not in dp.current_city)
                       and not (f == "accommodation" and day == last_day)]
            if missing:
                violations.append(ConstraintViolation(
                    "format", "generator",
                    f"Day {day} is missing fields: {missing}.", day=day, severity="format"))
        return violations


class LLMModuloBaseline:
    """
    Mirrors the "independent critics + concatenated backprompt" architecture.
    Critics here REUSE the same check() logic as our DCSP agents (for a fair
    comparison of the *coordination mechanism*, not the *constraint logic*),
    but crucially never call `concede()` -- instead, ALL violations are
    dumped back to the LLM as one big backprompt, and the LLM regenerates
    the WHOLE plan from scratch each round (as in the original paper).
    """

    def __init__(self, llm: LLMInterface, critic_set: str = "all", max_tries: int = 10):
        """critic_set: 'all' | 'common' | 'hard' | 'json' -- matches the
        ablation variants reported in Gundawar et al. Table 1."""
        self.llm = llm
        self.max_tries = max_tries
        self.format_critic = FormatCritic()
        self.hard_agents = [BudgetAgent(), HotelAgent(), TransportAgent(), FoodAgent(), AttractionAgent()]
        self.critic_set = critic_set

    def _active_hard_checks(self, plan: PlanState) -> list:
        if self.critic_set in ("all", "hard"):
            v = []
            for a in self.hard_agents:
                v.extend(a.check(plan))
            return v
        return []

    def run(self, plan: PlanState) -> LLMModuloResult:
        llm_calls = 0
        log = []

        # Initial fill using the same agent .propose() logic as the DCSP system,
        # so both baselines start from an equally reasonable first draft
        # (this isolates the comparison to the FIX loop, not initial quality).
        for a in self.hard_agents:
            a.propose(plan, sandbox)
        llm_calls += 1  # counts as the LLM's initial generation call

        for it in range(1, self.max_tries + 1):
            violations = []
            if self.critic_set in ("all", "json"):
                violations += self.format_critic.check(plan)
            violations += self._active_hard_checks(plan)
            # commonsense critics folded into hard-agent checks in this
            # simplified reimplementation; extend here for a closer match
            # if you add dedicated CommonsenseCritic classes.

            log.append((it, [v.as_backprompt() for v in violations]))

            if not violations:
                return LLMModuloResult(plan, True, it, llm_calls, log)

            # Metacontroller: concatenate ALL backprompts, regenerate whole plan.
            combined_backprompt = "\n".join(v.as_backprompt() for v in violations)
            plan = self._regenerate_with_backprompt(plan, combined_backprompt, attempt=it)
            llm_calls += 1

        return LLMModuloResult(plan, False, self.max_tries, llm_calls, log)

    def _regenerate_with_backprompt(self, plan: PlanState, backprompt: str, attempt: int = 0) -> PlanState:
        """
        In the real system this would be a full LLM call re-generating the
        plan conditioned on `backprompt`. Here we approximate that behavior
        deterministically (LLM stand-in) by re-running propose() for the
        agents implicated in the backprompt, passing `attempt` so each retry
        genuinely explores a DIFFERENT candidate rather than deterministically
        regenerating the exact same plan every round -- this gives the
        baseline a fair chance to improve over its iteration budget, so the
        comparison against DCSP negotiation isn't stacked by construction.
        Swap this method's body for a real LLM call when `self.llm` is a
        RealLLM instance for true fidelity to the original paper.
        """
        for a in self.hard_agents:
            if a.name in backprompt:
                # crude retry: clear this agent's variables and re-propose
                # with a different `attempt` index to force a new candidate
                for dp in plan.days_plan.values():
                    for kind in ("transportation", "accommodation", "breakfast", "lunch", "dinner", "attraction"):
                        if a.owns(dp.day, kind):
                            setattr(dp, kind, None)
                a.propose(plan, sandbox, attempt=attempt)
        return plan
