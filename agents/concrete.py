"""
Concrete constraint agents: Budget, Hotel, Transport, Food.

Design notes:
  - HotelAgent, TransportAgent, FoodAgent each own specific per-day variables
    and try to satisfy their OWN local constraints (room type/rule, transport
    preference, cuisine) while proposing the cheapest option that fits.
  - BudgetAgent owns no variable directly -- it watches the *total* cost and,
    when over budget, signals a conflict at whichever other agent currently
    holds the single most expensive swappable item (a simple, explainable
    "who should concede" heuristic -- see solver.py for where this plugs in).
"""

from __future__ import annotations
from data_model import (PlanState, DayPlan, ConstraintViolation, transport_cost, lodging_cost,
                        meal_cost, cuisines_of, room_type_ok)
from data import sandbox
from agents.base import ConstraintAgent


# --------------------------------------------------------------------------- #
# TransportAgent
# --------------------------------------------------------------------------- #

class TransportAgent(ConstraintAgent):
    name = "transport"
    severity = "hard"
    priority = 0.6

    def __init__(self):
        self._tried: dict = {}  # day -> set of Flight Number / mode+dest already rejected

    def _cand_id(self, c):
        return c.get("Flight Number", f"{c.get('mode')}-{c.get('cost')}")

    def owns(self, day, kind):
        return kind == "transportation"

    def _candidates(self, plan: PlanState, day: int):
        dp = plan.days_plan[day]
        origin, dest = self._leg_cities(plan, day)
        if origin is None:
            return []
        pref = plan.query.transportation_pref  # "No flight" / "No self-driving" / None
        candidates = []
        if pref != "No flight":
            candidates += sandbox.search_flights(origin, dest, plan.query.date[day - 1] if plan.query.date else "2022-03-01")
        if pref != "No self-driving":
            candidates += sandbox.search_self_driving(origin, dest)
        # taxi is allowed under every preference (only exists in the real database)
        if hasattr(sandbox, "search_taxi"):
            candidates += sandbox.search_taxi(origin, dest)
        # Commonsense: a trip must not mix self-driving with flights. Stay consistent
        # with the legs already booked on the OTHER days.
        others = {d.transportation.get("mode") for k, d in plan.days_plan.items()
                  if k != day and d.transportation}
        if "self-driving" in others and "flight" not in others:
            candidates = [c for c in candidates if c.get("mode") != "flight"] or candidates
        elif "flight" in others and "self-driving" not in others:
            candidates = [c for c in candidates if c.get("mode") != "self-driving"] or candidates
        n = plan.query.people_number
        return sorted(candidates, key=lambda c: transport_cost(c, n))

    def _leg_cities(self, plan: PlanState, day: int):
        dp = plan.days_plan.get(day)
        if dp is None or " to " not in dp.current_city:
            return None, None
        origin, dest = dp.current_city.split(" to ", 1)
        origin = origin.replace("from ", "").strip()
        return origin, dest.strip()

    def propose(self, plan, sbx, hint=None, attempt=0):
        preferred_mode = hint.get("transport_mode") if hint else None
        for day, dp in plan.days_plan.items():
            if " to " not in dp.current_city:
                continue
            cands = self._candidates(plan, day)
            if not cands:
                continue
            if preferred_mode:
                matched = [c for c in cands if c.get("mode") == preferred_mode]
                pool = matched if matched else cands
                dp.transportation = pool[(len(pool) // 2 + attempt) % len(pool)]
            else:
                # Cold-start proposal picks a REASONABLE (median-cost) option rather
                # than the globally cheapest one -- this mirrors a first LLM guess,
                # which wouldn't reliably be cost-optimal, and leaves genuine room
                # for Budget-driven negotiation to demonstrate value across rounds.
                dp.transportation = cands[(len(cands) // 2 + attempt) % len(cands)]

    def check(self, plan):
        violations = []
        pref = plan.query.transportation_pref
        for day, dp in plan.days_plan.items():
            if " to " not in dp.current_city:
                continue
            if dp.transportation is None:
                violations.append(ConstraintViolation(
                    "transport_missing", self.name,
                    f"Day {day}: no transportation booked for leg '{dp.current_city}'.",
                    day=day, severity="hard"))
                continue
            mode = dp.transportation.get("mode")
            if pref == "No flight" and mode == "flight":
                violations.append(ConstraintViolation(
                    "transport_pref", self.name,
                    f"Day {day}: user disallows flights but plan uses one.",
                    day=day, severity="hard"))
            if pref == "No self-driving" and mode == "self-driving":
                violations.append(ConstraintViolation(
                    "transport_pref", self.name,
                    f"Day {day}: user disallows self-driving but plan uses it.",
                    day=day, severity="hard"))
        modes = {dp.transportation.get("mode") for dp in plan.days_plan.values() if dp.transportation}
        if "self-driving" in modes and "flight" in modes:
            first = min(d for d, dp in plan.days_plan.items()
                        if dp.transportation and dp.transportation.get("mode") == "self-driving")
            violations.append(ConstraintViolation(
                "non_conflicting_transportation", self.name,
                f"Day {first}: plan mixes self-driving with flights.", day=first, severity="commonsense"))
        return violations

    def concede(self, plan, sbx, violation: ConstraintViolation):
        day = violation.day
        dp = plan.days_plan[day]
        cands = self._candidates(plan, day)
        current = dp.transportation
        tried = self._tried.setdefault(day, set())
        if current is not None:
            tried.add(self._cand_id(current))

        n = plan.query.people_number
        if violation.critic_name == "non_conflicting_transportation":
            return self._unify_family(plan)

        if violation.critic_name == "budget":
            # Budget wants something CHEAPER than what we currently hold --
            # searching the generic "next untried, cheapest-first" list would
            # actually walk UPWARD in price once the cheapest option is used,
            # so here we specifically look for anything strictly cheaper.
            cur_cost = transport_cost(current, n) if current else float("inf")
            cheaper = [c for c in cands if self._cand_id(c) not in tried and transport_cost(c, n) < cur_cost]
            if not cheaper:
                # No cheaper option in the current mode family: try switching the
                # WHOLE trip to the other family (flight <-> self-driving) if that is cheaper.
                return self._switch_family(plan)
            cheaper.sort(key=lambda c: -transport_cost(c, n))   # nearest cheaper = minimal concession
            dp.transportation = cheaper[0]
            return True

        # Non-budget violation (e.g. transport preference mismatch): find the
        # next untried candidate satisfying the preference, cheapest first.
        remaining = [c for c in cands if self._cand_id(c) not in tried]
        if not remaining:
            return False
        remaining.sort(key=lambda c: transport_cost(c, n))
        dp.transportation = remaining[0]
        return True

    # ---- whole-trip mode-family helpers ------------------------------------ #

    def _leg_pool(self, plan, day: int, forbid: str | None = None):
        """All transport options for one leg, cheapest (group cost) first; optionally
        excluding one mode ('flight' or 'self-driving') to keep a trip in one family."""
        origin, dest = self._leg_cities(plan, day)
        pref = plan.query.transportation_pref
        pool = []
        if pref != "No flight":
            pool += sandbox.search_flights(origin, dest, plan.query.date[day - 1] if plan.query.date else "2022-03-01")
        if pref != "No self-driving":
            pool += sandbox.search_self_driving(origin, dest)
        if hasattr(sandbox, "search_taxi"):
            pool += sandbox.search_taxi(origin, dest)
        if forbid:
            pool = [c for c in pool if c.get("mode") != forbid]
        n = plan.query.people_number
        return sorted(pool, key=lambda c: transport_cost(c, n))

    def _family_plan(self, plan, forbid: str):
        """Cheapest assignment for every leg that never uses mode `forbid`.
        Returns {day: candidate} or None if some leg has no such option."""
        out = {}
        for day, dp in plan.days_plan.items():
            if " to " not in dp.current_city:
                continue
            pool = self._leg_pool(plan, day, forbid)
            if not pool:
                return None
            out[day] = pool[0]
        return out

    def _apply(self, plan, assignment) -> None:
        for day, cand in assignment.items():
            plan.days_plan[day].transportation = cand

    def _unify_family(self, plan) -> bool:
        """Resolve flight/self-driving mixing: pick the cheaper consistent family."""
        n = plan.query.people_number
        options = []
        for forbid in ("self-driving", "flight"):
            a = self._family_plan(plan, forbid)
            if a:
                options.append((sum(transport_cost(c, n) for c in a.values()), forbid, a))
        if not options:
            return False
        options.sort(key=lambda t: t[0])
        self._apply(plan, options[0][2])
        return True

    def _switch_family(self, plan) -> bool:
        """Budget pressure: move the whole trip to the other family if strictly cheaper."""
        n = plan.query.people_number
        current = sum(transport_cost(dp.transportation, n) for dp in plan.days_plan.values() if dp.transportation)
        modes = {dp.transportation.get("mode") for dp in plan.days_plan.values() if dp.transportation}
        best = None
        for forbid in ("self-driving", "flight"):
            if forbid not in modes and modes != {"taxi"} and len(modes - {"taxi"}) == 1:
                pass
            a = self._family_plan(plan, forbid)
            if a:
                total = sum(transport_cost(c, n) for c in a.values())
                if total < current - 1e-9 and (best is None or total < best[0]):
                    best = (total, a)
        if best is None:
            return False
        self._apply(plan, best[1])
        return True

    def domain_exhausted(self, plan):
        return False  # simplification: transport domain rarely exhausts in this mock


# --------------------------------------------------------------------------- #
# HotelAgent
# --------------------------------------------------------------------------- #

class HotelAgent(ConstraintAgent):
    name = "hotel"
    severity = "hard"
    priority = 0.5

    def __init__(self):
        self._tried: dict = {}  # day -> set of accommodation NAME already rejected

    def owns(self, day, kind):
        return kind == "accommodation"

    @staticmethod
    def _stay_length(plan: PlanState, city: str) -> int:
        """Number of nights the plan spends in `city` (last day needs no room)."""
        n = 0
        last = max(plan.days_plan)
        for d, dp in plan.days_plan.items():
            if d == last:
                continue
            c = dp.current_city.split(" to ")[-1].strip() if " to " in dp.current_city else dp.current_city
            if c == city:
                n += 1
        return n

    def _candidates(self, plan: PlanState, city: str):
        cands = sandbox.search_accommodations(city)
        n_people = plan.query.people_number
        wanted_type = plan.query.room_type
        wanted_rule = plan.query.room_rule
        stay = self._stay_length(plan, city)
        # Commonsense: honour the property's minimum-nights rule (soft preference
        # -- fall back to the full list if nothing satisfies it).
        cands = [c for c in cands if c.get("minimum nights", 1) <= stay] or cands
        filtered = cands
        if wanted_type:
            filtered = [c for c in filtered if room_type_ok(wanted_type, c["room type"])] or filtered
        if wanted_rule:
            filtered = [c for c in filtered if wanted_rule not in c.get("house_rules", [])] or filtered
        return sorted(filtered, key=lambda c: lodging_cost(c, n_people))

    def propose(self, plan, sbx, hint=None, attempt=0):
        preferred_type = hint.get("room_type") if hint else None
        last_day = max(plan.days_plan)   # departure day: no night to book (as in TravelPlanner)
        for day, dp in plan.days_plan.items():
            if day == last_day:
                continue
            city = dp.current_city.split(" to ")[-1].strip() if " to " in dp.current_city else dp.current_city
            if dp.accommodation is None:
                cands = self._candidates(plan, city)
                if not cands:
                    continue
                if preferred_type:
                    matched = [c for c in cands if c.get("room type") == preferred_type]
                    pool = matched if matched else cands
                    dp.accommodation = pool[(len(pool) // 2 + attempt) % len(pool)]
                else:
                    # Reasonable (median-price) cold start -- see TransportAgent
                    # for the rationale; leaves headroom for negotiation.
                    dp.accommodation = cands[(len(cands) // 2 + attempt) % len(cands)]

    def check(self, plan):
        violations = []
        wanted_type = plan.query.room_type
        wanted_rule = plan.query.room_rule
        for day, dp in plan.days_plan.items():
            if dp.accommodation is None:
                continue
            if wanted_type and not room_type_ok(wanted_type, dp.accommodation.get("room type")):
                violations.append(ConstraintViolation(
                    "room_type", self.name,
                    f"Day {day}: accommodation room type '{dp.accommodation.get('room type')}' "
                    f"does not match requested '{wanted_type}'.", day=day, severity="hard"))
            if wanted_rule and wanted_rule in dp.accommodation.get("house_rules", []):
                violations.append(ConstraintViolation(
                    "room_rule", self.name,
                    f"Day {day}: accommodation violates room rule '{wanted_rule}'.",
                    day=day, severity="hard"))
        # Commonsense: minimum-nights rule (one violation per offending stay)
        days_sorted = sorted(plan.days_plan)
        i = 0
        while i < len(days_sorted):
            acc = plan.days_plan[days_sorted[i]].accommodation
            if acc is None:
                i += 1
                continue
            j = i
            while j + 1 < len(days_sorted) and plan.days_plan[days_sorted[j + 1]].accommodation == acc:
                j += 1
            if acc.get("minimum nights", 1) > days_sorted[j] - days_sorted[i] + 1:
                violations.append(ConstraintViolation(
                    "minimum_nights_stay", self.name,
                    f"Days {days_sorted[i]}-{days_sorted[j]}: '{acc['NAME']}' needs "
                    f"{acc.get('minimum nights')} nights minimum.",
                    day=days_sorted[i], severity="commonsense"))
            i = j + 1
        return violations

    def concede(self, plan, sbx, violation: ConstraintViolation):
        day = violation.day
        dp = plan.days_plan[day]
        city = dp.current_city.split(" to ")[-1].strip() if " to " in dp.current_city else dp.current_city
        cands = self._candidates(plan, city)
        current = dp.accommodation
        tried = self._tried.setdefault(day, set())
        if current is not None:
            tried.add(current["NAME"])

        if violation.critic_name == "minimum_nights_stay":
            old = current
            stay = [d for d, x in plan.days_plan.items() if x.accommodation == old]
            ok = [c for c in cands if c["NAME"] not in tried and c.get("minimum nights", 1) <= len(stay)]
            if not ok:
                return False
            ok.sort(key=lambda c: lodging_cost(c, plan.query.people_number))
            for d in stay:
                plan.days_plan[d].accommodation = ok[len(ok) // 2]
            return True

        if violation.critic_name == "budget":
            n_people = plan.query.people_number
            cur_price = lodging_cost(current, n_people) if current else float("inf")
            stay = [d for d, x in plan.days_plan.items() if x.accommodation == current] if current else [day]
            cheaper = [c for c in cands if c["NAME"] not in tried and lodging_cost(c, n_people) < cur_price
                       and c.get("minimum nights", 1) <= len(stay)]
            if not cheaper:
                return False  # already at the cheapest available option -- genuinely exhausted
            cheaper.sort(key=lambda c: lodging_cost(c, n_people))
            # Move the WHOLE stay so one hotel is kept for the run (keeps min-nights satisfied)
            for d in stay:
                plan.days_plan[d].accommodation = cheaper[-1]  # nearest cheaper option = minimal concession
            return True

        remaining = [c for c in cands if c["NAME"] not in tried]
        if not remaining:
            return False  # exhausted -- every candidate has already been tried and rejected
        dp.accommodation = remaining[0]
        return True

    def domain_exhausted(self, plan):
        return False


# --------------------------------------------------------------------------- #
# FoodAgent
# --------------------------------------------------------------------------- #

class FoodAgent(ConstraintAgent):
    """
    Owns breakfast / lunch / dinner for every day.

    Semantics follow the TravelPlanner benchmark:
      * cuisine (hard): EACH requested cuisine must appear at least once
        somewhere in the trip (not "every meal must be that cuisine").
      * diverse_restaurants (commonsense): no restaurant is repeated.
    """
    name = "food"
    severity = "hard"
    priority = 0.3              # concedes relatively easily (cheap swaps)

    MEALS = ("breakfast", "lunch", "dinner")

    def owns(self, day, kind):
        return kind in self.MEALS

    @staticmethod
    def _city(dp):
        return dp.current_city.split(" to ")[-1].strip() if " to " in dp.current_city else dp.current_city

    def _all(self, city):
        return sorted(sandbox.search_restaurants(city), key=lambda c: c["cost"])

    @staticmethod
    def _missing_cuisines(plan, wanted):
        have = set()
        for dp in plan.days_plan.values():
            for m in FoodAgent.MEALS:
                if getattr(dp, m):
                    have |= cuisines_of(getattr(dp, m))
        return [c for c in wanted if c not in have]

    def propose(self, plan, sbx, hint=None, attempt=0):
        wanted = list(plan.query.cuisine or [])
        cuisine_order = hint.get("cuisine_order") if hint else None
        used = set()
        # Slots to be used for cuisine coverage: dinner of day 1, 2, ... (one per wanted cuisine)
        coverage = {}
        for i, cu in enumerate(wanted):
            days = sorted(plan.days_plan)
            coverage[(days[i % len(days)], "dinner")] = cu
        for day in sorted(plan.days_plan):
            dp = plan.days_plan[day]
            pool = self._all(self._city(dp))
            if not pool:
                continue
            if cuisine_order:   # RAG-seeded preference ordering
                pool = sorted(pool, key=lambda c: (
                    min((cuisine_order.index(x) for x in cuisines_of(c) if x in cuisine_order),
                        default=len(cuisine_order)),
                    c["cost"]))
            for meal in self.MEALS:
                if getattr(dp, meal) is not None:
                    continue
                cu = coverage.get((day, meal))
                cands = [c for c in pool if c["Name"] not in used and (cu is None or cu in cuisines_of(c))]
                if not cands:
                    cands = [c for c in pool if c["Name"] not in used] or pool
                # median-cost start (leaves headroom for budget negotiation); `attempt` shifts it
                pick = cands[(len(cands) // 2 + attempt) % len(cands)] if cu is None else cands[attempt % len(cands)]
                setattr(dp, meal, pick)
                used.add(pick["Name"])

    def check(self, plan):
        violations = []
        wanted = list(plan.query.cuisine or [])
        for k, cu in enumerate(self._missing_cuisines(plan, wanted)):
            day = sorted(plan.days_plan)[k % len(plan.days_plan)]
            violations.append(ConstraintViolation(
                "cuisine", self.name,
                f"Day {day} dinner: requested cuisine '{cu}' does not appear anywhere in the trip.",
                day=day, severity="hard"))
        seen = {}
        for day in sorted(plan.days_plan):
            dp = plan.days_plan[day]
            for meal in self.MEALS:
                item = getattr(dp, meal)
                if not item:
                    continue
                name = item["Name"]
                if name in seen:
                    violations.append(ConstraintViolation(
                        "diverse_restaurants", self.name,
                        f"Day {day} {meal}: restaurant '{name}' repeats a choice from day {seen[name][0]} "
                        f"{seen[name][1]}.", day=day, severity="commonsense"))
                else:
                    seen[name] = (day, meal)
        return violations

    def concede(self, plan, sbx, violation: ConstraintViolation):
        day = violation.day
        dp = plan.days_plan[day]
        pool = self._all(self._city(dp))
        meal = next((m for m in self.MEALS if m in violation.message), None)
        if meal is None:
            return False
        current = getattr(dp, meal)
        used_elsewhere = self._names_used_in_plan(plan, exclude_day=day, exclude_meal=meal)
        wanted = list(plan.query.cuisine or [])

        if violation.critic_name == "cuisine":
            missing = self._missing_cuisines(plan, wanted)
            if not missing:
                return False
            cands = [c for c in pool if missing[0] in cuisines_of(c) and c["Name"] not in used_elsewhere]
            if not cands:
                return False
            setattr(dp, meal, cands[0])
            return True

        # Never swap away the only restaurant covering a requested cuisine.
        def keeps_coverage(c):
            if not current:
                return True
            lost = (cuisines_of(current) & set(wanted)) - cuisines_of(c)
            for cu in lost:
                count = sum(1 for d in plan.days_plan.values() for m in self.MEALS
                            if getattr(d, m) and cu in cuisines_of(getattr(d, m)))
                if count <= 1:
                    return False
            return True

        cands = [c for c in pool if c != current and c["Name"] not in used_elsewhere and keeps_coverage(c)]
        if violation.critic_name == "budget" and current:
            cands = [c for c in cands if c["cost"] < current["cost"]]
            cands.sort(key=lambda c: -c["cost"])      # nearest cheaper = minimal concession
        if not cands:
            return False     # genuinely exhausted -- report instead of thrashing on duplicates
        setattr(dp, meal, cands[0])
        return True

    @staticmethod
    def _names_used_in_plan(plan, exclude_day, exclude_meal):
        """Names used anywhere in the plan except the slot being fixed (other days AND other meals)."""
        names = set()
        for day, dp in plan.days_plan.items():
            for meal in FoodAgent.MEALS:
                if day == exclude_day and meal == exclude_meal:
                    continue
                item = getattr(dp, meal)
                if item:
                    names.add(item["Name"])
        return names

    def domain_exhausted(self, plan):
        return False


# --------------------------------------------------------------------------- #
# BudgetAgent  (meta-agent: watches total cost, doesn't own a variable itself)
# --------------------------------------------------------------------------- #

class BudgetAgent(ConstraintAgent):
    name = "budget"
    severity = "hard"
    priority = 1.0  # never concedes -- budget is fixed by the user

    def owns(self, day, kind):
        return False  # owns no variable; it watches total cost across all agents

    TOP_K = 3   # how many of the most expensive items are blamed per round

    def check(self, plan):
        total = plan.total_cost()
        if total <= plan.query.budget:
            return []
        over_by = total - plan.query.budget
        violations = []
        # Blame the K most expensive swappable items (each owner agent is asked to
        # concede) -- pointing at only one item can stall if that owner is exhausted.
        for day, kind, cost in self._top_expensive_items(plan, self.TOP_K):
            violations.append(ConstraintViolation(
                "budget", self._owner_agent_name(kind),
                f"Total cost {total:.2f} exceeds budget {plan.query.budget:.2f} "
                f"by {over_by:.2f}. Expensive swappable item: day {day} "
                f"{kind} (cost {cost:.2f}). Please propose a cheaper alternative.",
                day=day, severity="hard"))
        return violations

    def propose(self, plan, sbx, hint=None, attempt=0):
        pass  # no variables to propose

    def concede(self, plan, sbx, violation):
        return False  # BudgetAgent itself never concedes -- others concede to it

    def domain_exhausted(self, plan):
        return False

    @staticmethod
    def _owner_agent_name(kind: str) -> str:
        return {
            "transportation": "transport",
            "accommodation": "hotel",
            "breakfast": "food", "lunch": "food", "dinner": "food",
        }.get(kind, "food")

    @staticmethod
    def _top_expensive_items(plan: PlanState, k: int = 3):
        """(day, kind, group_cost) of the k most expensive line items."""
        n = plan.query.people_number
        items = []
        for day, dp in plan.days_plan.items():
            if dp.transportation:
                items.append((day, "transportation", transport_cost(dp.transportation, n)))
            if dp.accommodation:
                items.append((day, "accommodation", lodging_cost(dp.accommodation, n)))
            for meal in ("breakfast", "lunch", "dinner"):
                m = getattr(dp, meal)
                if m:
                    items.append((day, meal, meal_cost(m, n)))
        items.sort(key=lambda t: -t[2])
        return items[:k]

    @staticmethod
    def _most_expensive_item(plan: PlanState):
        top = BudgetAgent._top_expensive_items(plan, 1)
        return top[0] if top else (None, None, -1)


# --------------------------------------------------------------------------- #
# AttractionAgent  (fills the `attraction` field; owns diversity, not budget)
# --------------------------------------------------------------------------- #

class AttractionAgent(ConstraintAgent):
    """Fills the attraction field for every day and enforces the diverse-
    attractions commonsense constraint. Doesn't affect total cost (mock
    attractions are free), so it never conflicts with BudgetAgent -- a good
    example of an agent that only ever negotiates with itself over time."""
    name = "attraction"
    severity = "commonsense"
    priority = 0.2  # concedes very easily -- swapping an attraction is cheap

    def owns(self, day, kind):
        return kind == "attraction"

    def _candidates(self, city: str):
        return sandbox.search_attractions(city)

    # def propose(self, plan, sbx, hint=None, attempt=0):
    #     used_names = set()
    #     for day, dp in plan.days_plan.items():
    #         city = dp.current_city.split(" to ")[-1].strip() if " to " in dp.current_city else dp.current_city
    #         if dp.attraction:
    #             continue
    #         cands = self._candidates(city)
    #         fresh = [c for c in cands if c["Name"] not in used_names]
    #         pick = fresh[:2] if fresh else cands[:2]
    #         dp.attraction = pick
    #         used_names.update(a["Name"] for a in pick)
    def propose(self, plan, sbx, hint=None, attempt=0):
        used_names = set()

        for day, dp in plan.days_plan.items():
            city = (
                dp.current_city.split(" to ")[-1].strip()
                if " to " in dp.current_city
                else dp.current_city
            )

            cands = self._candidates(city)

            print(f"\nCity: {city}")
            print(f"First attraction record: {cands[0] if cands else 'No attractions found'}")

            if dp.attraction:
                continue

            fresh = [c for c in cands if c["Name"] not in used_names]
            pick = fresh[:2] if fresh else cands[:2]

            dp.attraction = pick
            used_names.update(a["Name"] for a in pick)

    def check(self, plan):
        violations = []
        for day, dp in plan.days_plan.items():
            if not dp.attraction:
                violations.append(ConstraintViolation(
                    "complete_information", self.name,
                    f"Day {day}: no attraction planned.", day=day, severity="commonsense"))

        seen = {}
        for day, dp in plan.days_plan.items():
            for a in (dp.attraction or []):
                name = a["Name"]
                if name in seen:
                    violations.append(ConstraintViolation(
                        "diverse_attractions", self.name,
                        f"Day {day}: attraction '{name}' repeats day {seen[name]}'s choice.",
                        day=day, severity="commonsense"))
                else:
                    seen[name] = day
        return violations

    def concede(self, plan, sbx, violation: ConstraintViolation):
        day = violation.day
        dp = plan.days_plan[day]
        city = dp.current_city.split(" to ")[-1].strip() if " to " in dp.current_city else dp.current_city
        cands = self._candidates(city)
        used_elsewhere = {a["Name"] for d, ddp in plan.days_plan.items() if d != day
                           for a in (ddp.attraction or [])}
        if violation.critic_name == "complete_information":
            fresh = [c for c in cands if c["Name"] not in used_elsewhere]
            if not fresh:
                return False
            dp.attraction = fresh[:2]
            return True
        if violation.critic_name == "diverse_attractions":
            current_names = {a["Name"] for a in (dp.attraction or [])}
            fresh = [c for c in cands if c["Name"] not in used_elsewhere and c["Name"] not in current_names]
            if not fresh:
                return False
            # swap out one repeated item for a fresh one
            new_list = list(dp.attraction or [])
            new_list[0] = fresh[0]
            dp.attraction = new_list
            return True
        return False

    def domain_exhausted(self, plan):
        return False
