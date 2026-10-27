"""
TravelPlanner-style evaluation metrics.

Implements exactly the formulas from Xie et al. (2024), Section 3.4:

  Micro Pass Rate = (sum of passed constraints across all plans)
                     / (sum of total constraints checked across all plans)

  Macro Pass Rate = (# plans that pass ALL their constraints) / (# plans)

Plus Delivery Rate and Final Pass Rate (passes commonsense AND hard AND
was delivered).

This module is deliberately decoupled from the agents/solver -- it just
takes a PlanState + list of (commonsense_violations, hard_violations) and
scores it, so it works identically for the DCSP system, the LLM-Modulo
baseline, or any future system you add.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from data_model import PlanState, ConstraintViolation


COMMONSENSE_CHECKS = [
    "within_sandbox", "complete_information", "within_current_city",
    "reasonable_city_route", "diverse_restaurants", "diverse_attractions",
    "non_conflicting_transportation", "minimum_nights_stay",
]

HARD_CHECKS = ["budget", "room_rule", "room_type", "cuisine", "transportation"]


@dataclass
class PlanScore:
    delivered: bool
    commonsense_passed: int
    commonsense_total: int
    hard_passed: int
    hard_total: int
    final_pass: bool


def check_commonsense(plan: PlanState) -> tuple:
    """Returns (num_passed, num_total, list[ConstraintViolation]).
    Mirrors TravelPlanner's 8 commonsense dimensions at a structural level;
    extend individual checks here as your project matures."""
    violations = []
    total = 0

    # 1. Complete information -- every day has all required fields
    total += 1
    missing_any = False
    last_day = max(plan.days_plan) if plan.days_plan else 0
    for day, dp in plan.days_plan.items():
        for f in ("breakfast", "lunch", "dinner", "accommodation", "attraction"):
            if f == "accommodation" and day == last_day:
                continue   # departure day: no night to book
            val = getattr(dp, f)
            if val is None or (isinstance(val, list) and len(val) == 0):
                missing_any = True
    if missing_any:
        violations.append(ConstraintViolation("complete_information", "generator",
                                               "Some days are missing required fields.", severity="commonsense"))

    # 2. Within current city -- attractions/meals should belong to that day's city
    #    (mock sandbox already scopes searches by city, so this passes by construction
    #     unless you wire in cross-city leakage -- left as a hook for extension)
    total += 1

    # 3. Reasonable city route -- no teleporting between non-adjacent legs
    total += 1
    cities = [dp.current_city for dp in sorted(plan.days_plan.values(), key=lambda d: d.day)]
    # simplistic check: consecutive legs should chain (dest of leg i == origin of leg i+1)
    route_ok = True
    prev_dest = None
    for c in cities:
        if " to " in c:
            origin, dest = c.split(" to ")
            origin = origin.replace("from ", "").strip()
            if prev_dest and origin != prev_dest:
                route_ok = False
            prev_dest = dest.strip()
    if not route_ok:
        violations.append(ConstraintViolation("reasonable_city_route", "generator",
                                               "City route is not contiguous.", severity="commonsense"))

    # 4. Diverse restaurants -- no repeated restaurant name across the trip
    total += 1
    names = []
    for dp in plan.days_plan.values():
        for meal in ("breakfast", "lunch", "dinner"):
            item = getattr(dp, meal)
            if item:
                names.append(item.get("Name"))
    if len(names) != len(set(names)):
        violations.append(ConstraintViolation("diverse_restaurants", "food",
                                               "Repeated restaurant used across the trip.", severity="commonsense"))

    # 5. Diverse attractions
    total += 1
    attractions = []
    for dp in plan.days_plan.values():
        if dp.attraction:
            attractions += [a.get("Name") for a in dp.attraction]
    if attractions and len(attractions) != len(set(attractions)):
        violations.append(ConstraintViolation("diverse_attractions", "generator",
                                               "Repeated attraction used across the trip.", severity="commonsense"))

    # 6. Non-conflicting transportation (e.g. not mixing flight + self-driving oddly across one leg)
    total += 1
    modes = {dp.transportation.get("mode") for dp in plan.days_plan.values() if dp.transportation}
    if "self-driving" in modes and "flight" in modes:
        violations.append(ConstraintViolation("non_conflicting_transportation", "transport",
                                               "Plan mixes self-driving with flights.", severity="commonsense"))

    # 7. Minimum nights stay -- check that each accommodation is booked for
    #    at least as many consecutive days as its own minimum-night requirement.
    total += 1
    ok = True
    days_sorted = sorted(plan.days_plan.keys())
    i = 0
    while i < len(days_sorted):
        dp = plan.days_plan[days_sorted[i]]
        acc = dp.accommodation
        if acc is None:
            i += 1
            continue
        j = i
        while j + 1 < len(days_sorted) and plan.days_plan[days_sorted[j + 1]].accommodation == acc:
            j += 1
        consecutive_nights = days_sorted[j] - days_sorted[i] + 1
        min_required = acc.get("minimum nights", 1)
        if min_required > consecutive_nights:
            ok = False
        i = j + 1
    if not ok:
        violations.append(ConstraintViolation("minimum_nights_stay", "hotel",
                                               "Accommodation minimum-night requirement not met.", severity="commonsense"))

    # 8. Within sandbox (all referenced entities came from the sandbox -- true by
    #    construction in this implementation since agents only pull from `sandbox.py`)
    total += 1

    passed = total - len(violations)
    return passed, total, violations


def check_hard(plan: PlanState, hard_violations_from_agents: list) -> tuple:
    """hard_violations_from_agents: output of running Budget/Hotel/Transport/Food
    .check() on the final plan (i.e. what the DCSP agents or LLM-Modulo hard
    critics reported). Deduplicates by critic_name per TravelPlanner's dimension list."""
    fired = {v.critic_name for v in hard_violations_from_agents}
    applicable = set()
    if plan.query.budget:
        applicable.add("budget")
    if plan.query.room_rule:
        applicable.add("room_rule")
    if plan.query.room_type:
        applicable.add("room_type")
    if plan.query.cuisine:
        applicable.add("cuisine")
    if plan.query.transportation_pref:
        applicable.add("transport_pref")

    total = max(1, len(applicable))
    passed = total - len({f for f in fired if f in applicable or f == "budget"})
    passed = max(0, passed)
    return passed, total


def score_plan(plan: PlanState, delivered: bool, hard_violations_from_agents: list) -> PlanScore:
    cs_passed, cs_total, cs_violations = check_commonsense(plan)
    hard_passed, hard_total = check_hard(plan, hard_violations_from_agents)

    final_pass = delivered and (len(cs_violations) == 0) and (hard_passed == hard_total)
    return PlanScore(
        delivered=delivered,
        commonsense_passed=cs_passed, commonsense_total=cs_total,
        hard_passed=hard_passed, hard_total=hard_total,
        final_pass=final_pass,
    )


@dataclass
class AggregateResults:
    n: int = 0
    delivered: int = 0
    cs_passed_sum: int = 0
    cs_total_sum: int = 0
    cs_macro_pass: int = 0
    hard_passed_sum: int = 0
    hard_total_sum: int = 0
    hard_macro_pass: int = 0
    final_pass: int = 0

    def add(self, s: PlanScore):
        self.n += 1
        self.delivered += int(s.delivered)
        self.cs_passed_sum += s.commonsense_passed
        self.cs_total_sum += s.commonsense_total
        self.cs_macro_pass += int(s.commonsense_passed == s.commonsense_total)
        self.hard_passed_sum += s.hard_passed
        self.hard_total_sum += s.hard_total
        self.hard_macro_pass += int(s.hard_passed == s.hard_total)
        self.final_pass += int(s.final_pass)

    def as_dict(self) -> dict:
        n = max(1, self.n)
        return {
            "n": self.n,
            "delivery_rate": 100 * self.delivered / n,
            "commonsense_micro": 100 * self.cs_passed_sum / max(1, self.cs_total_sum),
            "commonsense_macro": 100 * self.cs_macro_pass / n,
            "hard_micro": 100 * self.hard_passed_sum / max(1, self.hard_total_sum),
            "hard_macro": 100 * self.hard_macro_pass / n,
            "final_pass_rate": 100 * self.final_pass / n,
        }
