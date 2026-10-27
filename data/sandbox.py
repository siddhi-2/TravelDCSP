"""
A mock sandbox database standing in for TravelPlanner's real tool-backed
databases (FlightSearch, RestaurantSearch, AttractionSearch,
AccommodationSearch). This lets the whole pipeline run end-to-end offline,
with a deterministic seed, before you wire in the real TravelPlanner CSVs.

To use the REAL TravelPlanner data instead: replace the functions in this
file with loaders over the Hugging Face dataset / the sandbox CSVs from
https://github.com/OSU-NLP-Group/TravelPlanner, keeping the same return
schema (list of dicts) so nothing else in the project needs to change.
"""

from __future__ import annotations
import hashlib
import random

IS_REAL = False   # flipped by data.real_sandbox.use_real_database()

CUISINES = ["Chinese", "American", "Italian", "Mexican", "Indian",
            "Mediterranean", "French", "Korean", "Asian", "Middle Eastern"]

ROOM_RULES = ["No parties", "No smoking", "No children under 10", "No pets", "No visitors"]
ROOM_TYPES = ["Entire Room", "Private Room", "Shared Room"]


def _rng(seed_key: str) -> random.Random:
    """Deterministic per-key RNG so repeated calls for the same city/day are stable."""
    return random.Random(int(hashlib.md5(seed_key.encode()).hexdigest(), 16) % (2**32))  # stable across runs


def search_flights(origin: str, destination: str, date: str, n: int = 5) -> list:
    r = _rng(f"flight-{origin}-{destination}-{date}")
    flights = []
    for i in range(n):
        dep_h = r.randint(5, 21)
        dur = r.randint(1, 6)
        flights.append({
            "Flight Number": f"F{r.randint(1000000, 9999999)}",
            "OriginCityName": origin,
            "DestCityName": destination,
            "FlightDate": date,
            "DepTime": f"{dep_h:02d}:{r.randint(0,59):02d}",
            "ArrTime": f"{(dep_h+dur)%24:02d}:{r.randint(0,59):02d}",
            "mode": "flight",
            "cost": round(r.uniform(80, 500), 2),
        })
    return flights


def search_self_driving(origin: str, destination: str, n: int = 1) -> list:
    r = _rng(f"drive-{origin}-{destination}")
    dist = r.uniform(50, 1500)
    return [{
        "mode": "self-driving",
        "OriginCityName": origin,
        "DestCityName": destination,
        "distance_km": round(dist, 1),
        "cost": round(dist * 0.5, 2),
    } for _ in range(n)]


def search_restaurants(city: str, n: int = 40) -> list:
    r = _rng(f"restaurant-{city}")
    names = ["Cafe Maple Street", "Olive Tree Cafe", "Birbal Ji Dhaba", "Pind Balluchi",
             "Bombay Vada Pav", "Nagaland's Kitchen", "Kylin Skybar", "The Curzon Room",
             "Books & Brew", "Sunset Grill", "Golden Dragon", "La Piazza"]
    out = []
    for i in range(n):
        out.append({
            "Name": r.choice(names) + f" #{i}",
            "City": city,
            "cost": round(r.uniform(8, 60), 2),
            "Cuisines": r.choice(CUISINES),
        })
    return out


def search_attractions(city: str, n: int = 40) -> list:
    r = _rng(f"attraction-{city}")
    names = ["History Museum", "Riverfront Park", "Art Gallery", "Botanical Garden",
             "Old Town Square", "Science Center", "City Zoo", "Heritage Trail"]
    return [{"Name": f"{city} {r.choice(names)} #{i}", "City": city} for i in range(n)]


def search_accommodations(city: str, n: int = 8) -> list:
    r = _rng(f"accommodation-{city}")
    names = ["Cozy Room for U", "Luxury Building Studio", "Sunny Chelsea Studio",
             "Peaceful Home Away", "East Side Apartment", "Downtown Loft"]
    out = []
    for i in range(n):
        out.append({
            "NAME": f"{r.choice(names)} #{i}",
            "City": city,
            "price": round(r.uniform(60, 400), 2),
            "room type": r.choice(ROOM_TYPES),
            "house_rules": r.sample(ROOM_RULES, k=r.randint(0, 2)),
            "minimum nights": r.randint(1, 4),
            "maximum occupancy": r.randint(1, 6),
        })
    return out


def cities_in_state(state: str, n: int = 3) -> list:
    r = _rng(f"cities-{state}")
    pool = ["Springfield", "Fairview", "Riverside", "Greenville", "Franklin",
            "Georgetown", "Madison", "Salem", "Bristol", "Clinton"]
    return r.sample(pool, k=min(n, len(pool)))


def search_taxi(origin: str, destination: str) -> list:
    """Synthetic world has no taxi option (the real TravelPlanner database does)."""
    return []


def plan_cities(org: str, dest: str, k: int) -> list:
    """Which `k` cities to visit. Synthetic world: destination + two filler towns."""
    return ([dest] + ["Fairview", "Riverside"])[: max(1, k)]
