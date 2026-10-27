"""
LLM integration layer -- kept thin and OUTSIDE the solving logic on purpose.
The LLM's three jobs anywhere in this system:
  1. extract_constraints(query_text) -> structured Query dict
  2. generate_initial_plan(query)    -> a rough day-by-day skeleton (city per day)
  3. render_final_plan(plan_state)   -> natural-language itinerary text

The actual constraint satisfaction (agents + DCSPSolver) never calls the LLM.

Two implementations:
  - MockLLM: rule-based, no API key needed. Good enough to exercise the full
    pipeline end-to-end and get your project running today.
  - RealLLM: thin wrapper over the OpenAI-compatible chat completions API.
    Swap in by setting LLM_BACKEND=real and OPENAI_API_KEY in your environment.
"""

from __future__ import annotations
import json
import os
import re
from data_model import Query, PlanState, DayPlan


class LLMInterface:
    is_real = False   # True for backends that make paid API calls
    def extract_constraints(self, query_text: str) -> dict:
        raise NotImplementedError

    def generate_initial_skeleton(self, query: Query) -> list:
        """Return a list of day->city strings, e.g. ['Seattle to LA', 'LA', 'LA to Seattle']."""
        raise NotImplementedError

    def render_final_plan(self, plan: PlanState) -> str:
        raise NotImplementedError

    def replan_with_backprompt(self, plan: PlanState, violations: list):
        """ESCALATION (architecture arrow "(6) Back Prompt"): called at most once,
        only when every negotiation restart failed. Returns a revised PlanState
        skeleton (new city sequence) or None if the LLM cannot improve it."""
        return None


# --------------------------------------------------------------------------- #
# Mock implementation -- no API key required
# --------------------------------------------------------------------------- #

class MockLLM(LLMInterface):
    """
    Rule-based stand-in for an LLM. Good enough to validate the whole
    pipeline (extraction -> agents -> solver -> rendering) without incurring
    API cost during development. Swap for RealLLM once the architecture is
    working end to end.
    """

    _BUDGET_RE = re.compile(r"\$?([\d,]+(?:\.\d+)?)")
    _DAYS_RE = re.compile(r"(\d+)[\s-]*day")
    _PEOPLE_RE = re.compile(r"(\d+)\s+(?:people|person|traveler|travelers)")

    def extract_constraints(self, query_text: str) -> dict:
        """Very rough regex/keyword extraction, standing in for an LLM call.
        Replace with a real LLM prompt (see RealLLM) for production use."""
        budget_match = self._BUDGET_RE.findall(query_text)
        budget = float(budget_match[-1].replace(",", "")) if budget_match else 1500.0

        days_match = self._DAYS_RE.search(query_text)
        days = int(days_match.group(1)) if days_match else 3

        people_match = self._PEOPLE_RE.search(query_text)
        people = int(people_match.group(1)) if people_match else 1

        local_constraint = {}
        for cuisine in ["Chinese", "American", "Italian", "Mexican", "Indian",
                         "Mediterranean", "French", "Korean", "Asian"]:
            if cuisine.lower() in query_text.lower():
                local_constraint.setdefault("cuisine", []).append(cuisine)
        if "pet" in query_text.lower():
            local_constraint["room rule"] = "No pets"  # user wants pet-friendly -> avoid "No pets" listings... (see note)
        if "entire room" in query_text.lower():
            local_constraint["room type"] = "Entire Room"
        if "no flight" in query_text.lower() or "self-driving" in query_text.lower():
            local_constraint["transportation"] = "No flight"
        if "no self-driving" in query_text.lower() or "no self driving" in query_text.lower():
            local_constraint["transportation"] = "No self-driving"

        return {
            "days": days,
            "people_number": people,
            "budget": budget,
            "local_constraint": local_constraint,
        }

    def generate_initial_skeleton(self, query: Query) -> list:
        from data import sandbox            # looked up at call time: synthetic OR real database
        k = max(1, query.visiting_city_number)
        cities = sandbox.plan_cities(query.org, query.dest, k)
        k = len(cities)
        if query.days == 2 * k + 1:
            # TravelPlanner pattern: day 1 travel, stay, travel to next city, ..., travel home
            skeleton, prev = [], query.org
            for c in cities:
                skeleton.append(f"from {prev} to {c}")   # travel day
                skeleton.append(c)                       # stay day
                prev = c
            skeleton.append(f"from {prev} to {query.org}")   # return leg
            return skeleton
        skeleton = [f"from {query.org} to {cities[0]}"]
        for c in cities[1:]:
            skeleton.append(f"from {skeleton[-1].split(' to ')[-1]} to {c}")
        while len(skeleton) < query.days:
            skeleton.append(cities[-1])
        skeleton = skeleton[: query.days]
        if len(skeleton) == query.days:
            skeleton[-1] = f"from {cities[-1]} to {query.org}"
        return skeleton

    def render_final_plan(self, plan: PlanState) -> str:
        lines = [f"Travel Plan: {plan.query.org} -> {plan.query.dest} "
                 f"({plan.query.days} days, budget ${plan.query.budget:.0f})"]
        for day in sorted(plan.days_plan):
            dp = plan.days_plan[day]
            lines.append(f"\nDay {day}: {dp.current_city}")
            d = dp.to_travelplanner_dict()
            for field_name in ("transportation", "breakfast", "attraction", "lunch", "dinner", "accommodation"):
                lines.append(f"  {field_name.capitalize()}: {d[field_name]}")
        lines.append(f"\nTotal estimated cost: ${plan.total_cost():.2f} (budget ${plan.query.budget:.2f})")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Real LLM implementation (OpenAI-compatible). Requires `pip install openai`.
# --------------------------------------------------------------------------- #

class RealLLM(LLMInterface):
    is_real = True
    # NOTE: replan_with_backprompt() below is the single optional LLM call made
    # when negotiation + restarts fail.
    def __init__(self, model: str = "gpt-4-turbo", api_key: str | None = None):
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError("pip install openai to use RealLLM") from e
        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        self.model = model

    def _chat(self, system: str, user: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.0,
        )
        return resp.choices[0].message.content

    def extract_constraints(self, query_text: str) -> dict:
        system = (
            "Extract structured travel constraints from the user's query. "
            "Return ONLY valid JSON with keys: days (int), people_number (int), "
            "budget (float), local_constraint (object with optional keys "
            "'room rule', 'cuisine' (list), 'room type', 'transportation')."
        )
        raw = self._chat(system, query_text)
        raw = raw.strip().strip("```json").strip("```").strip()
        return json.loads(raw)

    def generate_initial_skeleton(self, query: Query) -> list:
        system = (
            "Given a travel query, output ONLY a JSON list of strings, one per day, "
            "each describing the city or city-to-city leg for that day "
            "(e.g. [\"from Seattle to LA\", \"LA\", \"from LA to Seattle\"])."
        )
        user = f"org={query.org}, dest={query.dest}, days={query.days}, " \
               f"visiting_city_number={query.visiting_city_number}"
        raw = self._chat(system, user)
        raw = raw.strip().strip("```json").strip("```").strip()
        return json.loads(raw)

    def render_final_plan(self, plan: PlanState) -> str:
        system = "Render the following structured travel plan JSON as a clear, natural-language itinerary."
        raw = self._chat(system, plan.to_json())
        return raw

    def replan_with_backprompt(self, plan: PlanState, violations: list):
        from plan_builder import build_initial_plan  # local import avoids a cycle
        system = ("You are a travel-plan skeleton generator. Given the unresolved constraint "
                  "violations, return ONLY a JSON list with one string per day, each either "
                  "'CityName' or 'from A to B'. The list length must equal the number of days.")
        user = f"Days: {plan.query.days}. Origin: {plan.query.org}. Destination: {plan.query.dest}.\n" \
               f"Unresolved violations:\n" + "\n".join(violations[:10])
        try:
            raw = self._chat(system, user).strip().strip("```json").strip("```").strip()
            skeleton = json.loads(raw)
            if not isinstance(skeleton, list) or len(skeleton) != plan.query.days:
                return None
        except Exception:
            return None
        new = PlanState(query=plan.query)
        for day, city in enumerate(skeleton, start=1):
            new.days_plan[day] = DayPlan(day=day, current_city=city)
        return new


def get_llm(backend: str | None = None, model: str = "gpt-4-turbo") -> LLMInterface:
    backend = backend or os.environ.get("LLM_BACKEND", "mock")
    if backend == "real":
        return RealLLM(model=model)
    return MockLLM()
