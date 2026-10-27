"""
REAL TravelPlanner database backend (the "sandbox" of Xie et al., 2024).

The six tools of the benchmark map to these functions:

    FlightSearch          -> search_flights(origin, dest, date)
    GoogleDistanceMatrix  -> search_self_driving(origin, dest) / search_taxi(origin, dest)
    AccommodationSearch   -> search_accommodations(city)
    RestaurantSearch      -> search_restaurants(city)
    AttractionSearch      -> search_attractions(city)
    CitySearch            -> cities_in_state(state) / plan_cities(org, dest, k)

Usage (before running anything):

    from data.real_sandbox import use_real_database
    use_real_database("data/travelplanner/database")

After that call every agent transparently reads the real data, because the
agents call `sandbox.search_*` through the module (see agents/concrete.py).

Get the data:  python scripts/download_data.py   (see README)

Expected layout (the official database.zip). File discovery is by pattern, so
small differences in folder/file names are fine:
    database/flights/*.csv                 Flight Number, Price, DepTime, ArrTime, FlightDate,
                                           OriginCityName, DestCityName, ...
    database/accommodations/*.csv          NAME, price, room type, house_rules, minimum nights,
                                           maximum occupancy, review rate number, city
    database/restaurants/*.csv             Name, Average Cost, Cuisines, Aggregate Rating, City
    database/attractions/*.csv             Name, Latitude, Longitude, Address, ..., City
    database/googleDistanceMatrix/*.csv    origin, destination, cost, duration, distance
    database/background/citySet*.txt       city / state lists

Column names are matched case-insensitively with aliases, and a clear error
listing the real columns is raised if something is missing.
"""

from __future__ import annotations
import glob
import math
import os
import re

import pandas as pd

from data import sandbox as _synthetic

# TravelPlanner cost model for the distance matrix (per vehicle):
#   self-driving: 0.05 per km, seats 5      taxi: 1.0 per km, seats 4
SELF_DRIVE_RATE, SELF_DRIVE_SEATS = 0.05, 5
TAXI_RATE, TAXI_SEATS = 1.0, 4

US_STATES = {
    "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut", "Delaware",
    "Florida", "Georgia", "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky",
    "Louisiana", "Maine", "Maryland", "Massachusetts", "Michigan", "Minnesota", "Mississippi",
    "Missouri", "Montana", "Nebraska", "Nevada", "New Hampshire", "New Jersey", "New Mexico",
    "New York", "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon", "Pennsylvania",
    "Rhode Island", "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah", "Vermont",
    "Virginia", "Washington", "West Virginia", "Wisconsin", "Wyoming", "District of Columbia",
}

ROOM_TYPE_CANON = {
    "entire home/apt": "Entire Room", "entire room": "Entire Room",
    "private room": "Private Room", "shared room": "Shared Room",
    "hotel room": "Hotel Room",
}


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #

def _find_file(db: str, patterns: list, what: str) -> str:
    for pat in patterns:
        hits = sorted(glob.glob(os.path.join(db, "**", pat), recursive=True))
        if hits:
            return hits[0]
    raise FileNotFoundError(
        f"Could not find the {what} file under '{db}' (looked for {patterns}). "
        f"Download the official database (python scripts/download_data.py) and unzip it there.")


def _colmap(df: pd.DataFrame, want: dict, what: str) -> dict:
    """want = {canonical: [aliases...]} -> {canonical: actual column name}."""
    lower = {c.lower().strip(): c for c in df.columns}
    out = {}
    for canon, aliases in want.items():
        for a in aliases:
            if a.lower() in lower:
                out[canon] = lower[a.lower()]
                break
        else:
            raise KeyError(f"{what}: no column for '{canon}' (tried {aliases}). "
                           f"Columns present: {list(df.columns)}")
    return out


def _num(x, default=0.0) -> float:
    """Parse a number; blank / NaN / junk -> default (never NaN)."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        m = re.findall(r"[\d.]+", str(x).replace(",", ""))
        try:
            v = float(m[0]) if m else default
        except ValueError:
            v = default
    return default if v != v else v


def _has(x) -> bool:
    """True if a cell holds a real value (not NaN / blank)."""
    return x is not None and not (isinstance(x, float) and x != x) and str(x).strip() not in ("", "nan", "NaN")


class _DB:
    """Lazy, cached loaders for the six sources."""

    def __init__(self, path: str):
        self.path = path
        self._flights = None
        self._dist = None
        self._acc = None
        self._rest = None
        self._attr = None
        self._cities = None      # set of city names
        self._state_cities = None  # state -> [cities]

    # ---- flights ---------------------------------------------------------- #
    def flights(self):
        if self._flights is None:
            f = _find_file(self.path, ["*light*.csv"], "flights")
            head = pd.read_csv(f, nrows=0)
            cm = _colmap(head, {
                "num": ["Flight Number", "flight_number"], "price": ["Price", "price"],
                "dep": ["DepTime", "dep_time"], "arr": ["ArrTime", "arr_time"],
                "date": ["FlightDate", "flight_date", "date"],
                "org": ["OriginCityName", "origin"], "dst": ["DestCityName", "destination", "dest"],
            }, "flights")
            df = pd.read_csv(f, usecols=list(cm.values()))
            df = df.rename(columns={v: k for k, v in cm.items()})
            df = df.dropna(subset=["price", "org", "dst", "date"]).set_index(["org", "dst", "date"]).sort_index()
            self._flights = df
        return self._flights

    # ---- distance matrix ------------------------------------------------- #
    def distances(self):
        if self._dist is None:
            f = _find_file(self.path, ["distance*.csv"], "distance matrix")
            df = pd.read_csv(f)
            cm = _colmap(df, {"org": ["origin"], "dst": ["destination", "dest"],
                              "dist": ["distance"]}, "distance matrix")
            out = {}
            for o, d, dist in zip(df[cm["org"]], df[cm["dst"]], df[cm["dist"]]):
                if not (_has(o) and _has(d) and _has(dist)):
                    continue
                km = _num(dist, default=0.0)
                if km > 0:
                    out[(str(o).strip(), str(d).strip())] = km
            self._dist = out
        return self._dist

    # ---- accommodations ---------------------------------------------------- #
    def accommodations(self):
        if self._acc is None:
            f = _find_file(self.path, ["*ccommodation*.csv"], "accommodations")
            df = pd.read_csv(f)
            cm = _colmap(df, {
                "name": ["NAME", "name"], "price": ["price"], "room": ["room type", "room_type"],
                "rules": ["house_rules", "house rules"], "minn": ["minimum nights", "minimum_nights"],
                "maxo": ["maximum occupancy", "maximum_occupancy"], "city": ["city", "City"],
            }, "accommodations")
            by_city = {}
            for _, row in df.iterrows():
                if not (_has(row[cm["name"]]) and _has(row[cm["city"]]) and _has(row[cm["price"]])):
                    continue
                rules = [x.strip() for x in str(row[cm["rules"]]).split("&")
                         if x.strip() and x.strip().lower() != "nan"] if _has(row[cm["rules"]]) else []
                rt = str(row[cm["room"]]).strip() if _has(row[cm["room"]]) else "Entire home/apt"
                by_city.setdefault(str(row[cm["city"]]).strip(), []).append({
                    "NAME": str(row[cm["name"]]).strip(), "City": str(row[cm["city"]]).strip(),
                    "price": _num(row[cm["price"]]),
                    "room type": ROOM_TYPE_CANON.get(rt.lower(), rt.title()),
                    "house_rules": rules,
                    "minimum nights": max(1, int(_num(row[cm["minn"]], 1))),
                    "maximum occupancy": max(1, int(_num(row[cm["maxo"]], 1))),
                    "rooms_by_occupancy": True,
                })
            self._acc = by_city
        return self._acc

    # ---- restaurants ------------------------------------------------------- #
    def restaurants(self):
        if self._rest is None:
            f = _find_file(self.path, ["*estaurant*.csv"], "restaurants")
            df = pd.read_csv(f)
            cm = _colmap(df, {"name": ["Name", "name"], "cost": ["Average Cost", "average_cost", "cost"],
                              "cui": ["Cuisines", "cuisines"], "city": ["City", "city"]}, "restaurants")
            by_city = {}
            seen = set()
            for _, row in df.iterrows():
                if not (_has(row[cm["name"]]) and _has(row[cm["city"]]) and _has(row[cm["cost"]])):
                    continue
                city, name = str(row[cm["city"]]).strip(), str(row[cm["name"]]).strip()
                if (city, name) in seen:
                    continue      # keep names unique per city (diverse-restaurants check is by name)
                seen.add((city, name))
                by_city.setdefault(city, []).append({
                    "Name": name, "City": city, "cost": _num(row[cm["cost"]]),
                    "Cuisines": str(row[cm["cui"]]).strip() if _has(row[cm["cui"]]) else "",
                })
            self._rest = by_city
        return self._rest

    # ---- attractions ------------------------------------------------------- #
    def attractions(self):
        if self._attr is None:
            f = _find_file(self.path, ["*ttraction*.csv"], "attractions")
            df = pd.read_csv(f)
            cm = _colmap(df, {"name": ["Name", "name"], "city": ["City", "city"]}, "attractions")
            by_city, seen = {}, set()
            for name, city in zip(df[cm["name"]], df[cm["city"]]):
                if not (_has(name) and _has(city)):
                    continue
                city, name = str(city).strip(), str(name).strip()
                if (city, name) in seen:
                    continue
                seen.add((city, name))
                by_city.setdefault(city, []).append({"Name": name, "City": city})
            self._attr = by_city
        return self._attr

    # ---- city search ------------------------------------------------------- #
    def city_tables(self):
        if self._state_cities is None:
            state_cities: dict = {}
            files = sorted(glob.glob(os.path.join(self.path, "**", "citySet_with_states*.txt"), recursive=True))
            for f in files:
                for line in open(f, encoding="utf-8", errors="ignore"):
                    parts = [p.strip() for p in re.split(r"[\t,]", line) if p.strip()]
                    if len(parts) < 2:
                        continue
                    a, b = parts[0], parts[1]
                    if b in US_STATES:
                        city, state = a, b
                    elif a in US_STATES:
                        city, state = b, a
                    else:
                        continue
                    state_cities.setdefault(state, []).append(city)
            self._state_cities = state_cities
            plain = sorted(glob.glob(os.path.join(self.path, "**", "citySet.txt"), recursive=True))
            cities = set()
            for f in plain:
                cities |= {l.strip() for l in open(f, encoding="utf-8", errors="ignore") if l.strip()}
            for v in state_cities.values():
                cities |= set(v)
            if not cities:   # fall back to cities that appear in the main tables
                cities = set(self.accommodations()) & set(self.restaurants())
            self._cities = cities
        return self._cities, self._state_cities


_db: _DB | None = None


# --------------------------------------------------------------------------- #
# Public API -- identical signatures to data/sandbox.py
# --------------------------------------------------------------------------- #

def search_flights(origin: str, destination: str, date: str, n: int = 5) -> list:
    df = _db.flights()
    try:
        rows = df.loc[(origin, destination, date)]
    except KeyError:
        return []
    if isinstance(rows, pd.Series):
        rows = rows.to_frame().T
    return [{
        "Flight Number": str(r["num"]), "OriginCityName": origin, "DestCityName": destination,
        "FlightDate": date, "DepTime": str(r["dep"]), "ArrTime": str(r["arr"]),
        "mode": "flight", "cost": float(r["price"]),
    } for _, r in rows.iterrows()]


def _leg_km(origin: str, destination: str):
    return _db.distances().get((origin, destination))


def search_self_driving(origin: str, destination: str, n: int = 1) -> list:
    km = _leg_km(origin, destination)
    if km is None:
        return []
    return [{"mode": "self-driving", "OriginCityName": origin, "DestCityName": destination,
             "distance_km": km, "cost": round(km * SELF_DRIVE_RATE, 2), "capacity": SELF_DRIVE_SEATS}]


def search_taxi(origin: str, destination: str) -> list:
    km = _leg_km(origin, destination)
    if km is None:
        return []
    return [{"mode": "taxi", "OriginCityName": origin, "DestCityName": destination,
             "distance_km": km, "cost": round(km * TAXI_RATE, 2), "capacity": TAXI_SEATS}]


def search_accommodations(city: str, n: int = 0) -> list:
    return [dict(x, house_rules=list(x["house_rules"])) for x in _db.accommodations().get(city, [])]


def search_restaurants(city: str, n: int = 0) -> list:
    return [dict(x) for x in _db.restaurants().get(city, [])]


def search_attractions(city: str, n: int = 0) -> list:
    return [dict(x) for x in _db.attractions().get(city, [])]


def cities_in_state(state: str, n: int = 0) -> list:
    _, sc = _db.city_tables()
    return list(sc.get(state, []))


def _city_is_usable(city: str) -> bool:
    return bool(_db.accommodations().get(city) and _db.restaurants().get(city)
                and _db.attractions().get(city))


def plan_cities(org: str, dest: str, k: int) -> list:
    """Which k cities to visit. dest is a CITY when k == 1, otherwise a STATE
    (TravelPlanner convention). Picks usable cities (have hotels, restaurants and
    attractions) nearest to the origin first -- cheaper trips are easier to satisfy."""
    cities, sc = _db.city_tables()
    if dest in sc:
        pool = [c for c in sc[dest] if _city_is_usable(c) and c != org]
    elif dest in cities:
        pool = [dest]
    else:
        pool = [dest]
    k = max(1, k)
    if len(pool) > k:
        pool = sorted(pool, key=lambda c: (_leg_km(org, c) is None, _leg_km(org, c) or 0, c))
    return pool[:k] if pool else [dest]


# --------------------------------------------------------------------------- #

def valid_places():
    """(set of valid city names, {state: [cities]}) -- for input validation."""
    cities, sc = _db.city_tables()
    return cities, sc


def use_real_database(path: str) -> None:
    """Switch the whole project from the synthetic sandbox to the real database."""
    global _db
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Database folder not found: {path}")
    _db = _DB(path)
    for name in ("search_flights", "search_self_driving", "search_taxi", "search_accommodations",
                 "search_restaurants", "search_attractions", "cities_in_state", "plan_cities"):
        setattr(_synthetic, name, globals()[name])
    _synthetic.IS_REAL = True
    # fail fast with a clear message if a file/column is missing
    _db.flights(); _db.distances(); _db.accommodations()
    _db.restaurants(); _db.attractions(); _db.city_tables()
