"""
Core data model shared across agents, solver, baseline, and eval modules.

This mirrors the TravelPlanner JSON plan schema (day-wise itinerary with
transportation / breakfast / lunch / dinner / attraction / accommodation)
so that plans produced by this system are directly scorable by a
TravelPlanner-style evaluation script.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional
import copy
import json
import math


# --------------------------------------------------------------------------- #
# Shared helpers (cost model + normalisation). Work for BOTH the synthetic
# sandbox and the real TravelPlanner database.
# --------------------------------------------------------------------------- #

def transport_cost(item: dict, people: int) -> float:
    """Cost of one transportation leg for the whole group.
    Flights are priced per person. Self-driving / taxi items carry a
    `capacity` (cars needed = ceil(people / capacity)) -- as in TravelPlanner."""
    if not item:
        return 0.0
    cap = item.get("capacity")
    n = math.ceil(people / cap) if cap else people
    return item.get("cost", 0.0) * n


def lodging_cost(item: dict, people: int) -> float:
    """Cost of one night. Synthetic hotels: price per room. Real hotels carry
    `rooms_by_occupancy` -> rooms = ceil(people / maximum occupancy)."""
    if not item:
        return 0.0
    price = item.get("price", 0.0)
    if item.get("rooms_by_occupancy"):
        occ = max(1, int(item.get("maximum occupancy", 1) or 1))
        return price * math.ceil(people / occ)
    return price


def meal_cost(item: dict, people: int) -> float:
    return item.get("cost", 0.0) * people if item else 0.0


def cuisines_of(item: dict) -> set:
    """Set of cuisines a restaurant serves. Real data stores 'Chinese, Indian,
    Bakery' in one field; the synthetic data has a single value."""
    raw = (item or {}).get("Cuisines") or ""
    return {c.strip() for c in str(raw).split(",") if c.strip()}


def room_type_ok(wanted, actual) -> bool:
    """Room-type constraint, incl. TravelPlanner's negated form 'Not Shared Room'."""
    if not wanted:
        return True
    w = str(wanted).strip().lower()
    a = str(actual or "").strip().lower()
    if w.startswith("not "):
        return a != w[4:].strip()
    return a == w


# --------------------------------------------------------------------------- #
# Query representation
# --------------------------------------------------------------------------- #

@dataclass
class Query:
    """A single travel-planning query, matching TravelPlanner's query schema."""
    org: str
    dest: str
    days: int
    visiting_city_number: int
    date: list
    people_number: int
    budget: float
    query_text: str
    local_constraint: dict = field(default_factory=dict)
    chosen_cities: Optional[list] = None   # cities picked by the user in the UI (else chosen from the database)
    # local_constraint keys we support: room rule, cuisine (list), room type, transportation

    @property
    def room_rule(self) -> Optional[str]:
        return self.local_constraint.get("room rule")

    @property
    def cuisine(self):
        return self.local_constraint.get("cuisine")

    @property
    def room_type(self) -> Optional[str]:
        return self.local_constraint.get("room type")

    @property
    def transportation_pref(self) -> Optional[str]:
        return self.local_constraint.get("transportation")

    @classmethod
    def from_dict(cls, d: dict) -> "Query":
        return cls(
            org=d["org"],
            dest=d["dest"],
            days=d["days"],
            visiting_city_number=d.get("visiting_city_number", 1),
            date=d.get("date", []),
            people_number=d.get("people_number", 1),
            budget=float(d["budget"]),
            query_text=d.get("query", ""),
            local_constraint=d.get("local_constraint", d.get("constraint", {})) or {},
        )


# --------------------------------------------------------------------------- #
# Variable / domain representation (for DCSP framing)
# --------------------------------------------------------------------------- #

@dataclass
class Variable:
    """
    A single decision variable owned by one agent for one day of the trip,
    e.g. (day=1, kind='accommodation') -> candidate list of accommodation dicts.
    """
    day: int
    kind: str                # "transportation" | "accommodation" | "breakfast" | "lunch" | "dinner" | "attraction"
    domain: list              # candidate value objects (each a dict, e.g. with a "cost"/"price" field)
    assignment: Optional[dict] = None
    exhausted_values: list = field(default_factory=list)  # values already tried & rejected

    def remaining_domain(self) -> list:
        tried_ids = {id(v) for v in self.exhausted_values}
        return [v for v in self.domain if id(v) not in tried_ids and v is not self.assignment]

    def key(self):
        return (self.day, self.kind)


# --------------------------------------------------------------------------- #
# Plan state — the evolving itinerary
# --------------------------------------------------------------------------- #

@dataclass
class DayPlan:
    day: int
    current_city: str
    transportation: Optional[dict] = None
    breakfast: Optional[dict] = None
    lunch: Optional[dict] = None
    dinner: Optional[dict] = None
    attraction: Optional[list] = None
    accommodation: Optional[dict] = None

    def to_travelplanner_dict(self) -> dict:
        """Render into the structured dict a TravelPlanner-style evaluator expects
        (strings for each field, '-' for empty)."""

        def fmt(x, label_fields=("Name",)):
            if x is None:
                return "-"
            if isinstance(x, list):
                if not x:
                    return "-"
                return "; ".join(fmt(i) for i in x)
            if isinstance(x, dict):
                for lf in label_fields:
                    if lf in x:
                        city = x.get("City", x.get("city", ""))
                        return f"{x[lf]}, {city}".strip(", ")
                return str(x)
            return str(x)

        return {
            "days": self.day,
            "current_city": self.current_city,
            "transportation": fmt(self.transportation, ("mode", "Flight Number")),
            "breakfast": fmt(self.breakfast, ("Name",)),
            "attraction": fmt(self.attraction, ("Name",)),
            "lunch": fmt(self.lunch, ("Name",)),
            "dinner": fmt(self.dinner, ("Name",)),
            "accommodation": fmt(self.accommodation, ("NAME", "Name")),
        }


@dataclass
class PlanState:
    query: Query
    days_plan: dict = field(default_factory=dict)  # int day -> DayPlan

    def total_cost(self) -> float:
        total = 0.0
        n = self.query.people_number
        for dp in self.days_plan.values():
            total += transport_cost(dp.transportation, n)
            total += meal_cost(dp.breakfast, n) + meal_cost(dp.lunch, n) + meal_cost(dp.dinner, n)
            total += lodging_cost(dp.accommodation, n)
        return total

    def to_json(self) -> str:
        out = [self.days_plan[d].to_travelplanner_dict() for d in sorted(self.days_plan)]
        return json.dumps(out, indent=2)

    def clone(self) -> "PlanState":
        return copy.deepcopy(self)


# --------------------------------------------------------------------------- #
# Constraint violation / conflict signal
# --------------------------------------------------------------------------- #

@dataclass
class ConstraintViolation:
    critic_name: str            # which check failed, e.g. "budget", "room_type"
    responsible_agent: str      # which agent owns the variable that must change
    message: str                # human-readable explanation (used as LLM-Modulo backprompt text)
    day: Optional[int] = None   # which day's variable is implicated, if applicable
    severity: str = "hard"      # "hard" | "commonsense" | "format"

    def as_backprompt(self) -> str:
        return f"[{self.severity.upper()} | {self.critic_name}] {self.message}"