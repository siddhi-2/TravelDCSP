"""
Query loading.

`load_synthetic_queries()` generates deterministic mock queries so you can
run and test the whole pipeline right now, with no dataset download.

`load_travelplanner_csv()` is a stub loader for the REAL TravelPlanner data
once you download it (see README.md) -- fill in the column mapping to match
whatever schema the Hugging Face CSV export uses (org/dest/days/date/
people_number/budget/constraint/query columns, per the paper's Appendix).
"""

from __future__ import annotations
import random
from data_model import Query

CITIES = ["Springfield", "Fairview", "Riverside", "Greenville", "Franklin", "Georgetown"]
CUISINES = ["Chinese", "American", "Italian", "Mexican", "Indian", "Mediterranean", "French"]
ROOM_TYPES = ["Entire Room", "Private Room", "Shared Room"]
ROOM_RULES = ["No parties", "No smoking", "No children under 10", "No pets", "No visitors"]


def load_synthetic_queries(n: int = 20, difficulty: str = "mixed", seed: int = 42) -> list:
    """
    difficulty: 'easy' | 'medium' | 'hard' | 'mixed'
      easy   -> budget constraint only
      medium -> budget + 1 more constraint, 2-8 people
      hard   -> budget + 3 constraints, incl. transportation preference
    Mirrors TravelPlanner's own difficulty tiering (Section 3.3).
    """
    r = random.Random(seed)
    queries = []
    for i in range(n):
        diff = difficulty if difficulty != "mixed" else r.choice(["easy", "medium", "hard"])
        org, dest = r.sample(CITIES, 2)
        days = r.choice([3, 5, 7])
        people = 1 if diff == "easy" else r.randint(2, 8)
        budget = r.uniform(600, 900) * days * (people ** 0.5)

        local_constraint = {}
        if diff in ("medium", "hard"):
            extra = r.choice(["cuisine", "room_type", "room_rule"])
            if extra == "cuisine":
                local_constraint["cuisine"] = r.sample(CUISINES, k=r.randint(1, 2))
            elif extra == "room_type":
                local_constraint["room type"] = r.choice(ROOM_TYPES)
            else:
                local_constraint["room rule"] = r.choice(ROOM_RULES)
        if diff == "hard":
            local_constraint["transportation"] = r.choice(["No flight", "No self-driving"])
            # add a second extra constraint for hard tier
            extra2 = r.choice(["cuisine", "room_type"])
            if extra2 == "cuisine" and "cuisine" not in local_constraint:
                local_constraint["cuisine"] = r.sample(CUISINES, k=1)
            elif "room type" not in local_constraint:
                local_constraint["room type"] = r.choice(ROOM_TYPES)

        q = Query(
            org=org, dest=dest, days=days, visiting_city_number=1,
            date=[f"2022-03-{d:02d}" for d in range(1, days + 1)],
            people_number=people, budget=round(budget, 2),
            query_text=f"Plan a {days}-day trip from {org} to {dest} for {people} people, budget ${budget:.0f}.",
            local_constraint=local_constraint,
        )
        q._difficulty = diff  # tag for later stratified analysis
        queries.append(q)
    return queries


ROOM_TYPE_QUERY = {"entire room": "Entire Room", "private room": "Private Room",
                   "shared room": "Shared Room", "not shared room": "Not Shared Room"}


def _normalise_constraints(lc: dict) -> dict:
    """Map the REAL TravelPlanner local_constraint values to this project's canonical form:
        house rule: 'smoking'        -> room rule: 'No smoking'   (stay must NOT forbid it)
        room type : 'entire room'    -> 'Entire Room'  ('not shared room' -> 'Not Shared Room')
        cuisine   : 'Chinese'|list   -> ['Chinese']
        transportation: 'no flight'  -> 'No flight'
    """
    out = {}
    for k, v in (lc or {}).items():
        if not v:
            continue
        if k == "house rule":
            v = str(v).strip()
            out["room rule"] = v if v.lower().startswith("no ") else f"No {v.lower()}"
        elif k == "room type":
            out["room type"] = ROOM_TYPE_QUERY.get(str(v).strip().lower(), str(v).strip().title())
        elif k == "cuisine":
            out["cuisine"] = [str(c).strip() for c in (v if isinstance(v, (list, tuple, set)) else [v])]
        elif k == "transportation":
            v = str(v).strip().lower()
            out["transportation"] = "No flight" if "flight" in v else "No self-driving"
        else:
            out[k] = v
    return out


def load_travelplanner_csv(path: str, limit: int | None = None) -> list:
    """
    Load REAL TravelPlanner queries (osunlp/TravelPlanner) from csv / jsonl / parquet.

    Columns: org, dest, days, visiting_city_number, date, people_number,
             local_constraint, budget, query (+ level = easy/medium/hard).
    `date` and `local_constraint` are stringified Python literals in the CSV export.
    Constraint values are normalised by `_normalise_constraints` above.
    """
    import ast
    import pandas as pd

    if path.endswith(".jsonl"):
        df = pd.read_json(path, lines=True)
    elif path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    def lit(x, default):
        if isinstance(x, (list, dict)):
            return x
        if x is None or (isinstance(x, float) and x != x) or x == "":
            return default
        try:
            return ast.literal_eval(x)
        except (ValueError, SyntaxError):
            return default

    queries = []
    for _, row in df.iterrows():
        lc = _normalise_constraints(lit(row.get("local_constraint"), {}) or {})
        q = Query.from_dict({
            "org": str(row["org"]).strip(), "dest": str(row["dest"]).strip(), "days": int(row["days"]),
            "visiting_city_number": int(row.get("visiting_city_number", 1)),
            "date": list(lit(row.get("date"), [])),
            "people_number": int(row.get("people_number", 1)),
            "budget": float(row["budget"]), "query": row.get("query", ""),
            "local_constraint": lc,
        })
        q._difficulty = str(row.get("level", "n/a")).lower()
        queries.append(q)
        if limit and len(queries) >= limit:
            break
    return queries


def _split_file(data_dir: str, split: str) -> str:
    import glob, os
    for ext in ("csv", "jsonl", "parquet"):
        hits = sorted(glob.glob(os.path.join(data_dir, f"{split}.{ext}")) +
                      glob.glob(os.path.join(data_dir, "**", f"{split}.{ext}"), recursive=True))
        if hits:
            return hits[0]
    raise FileNotFoundError(f"No {split}.csv/.jsonl/.parquet under '{data_dir}'. "
                            f"Run: python scripts/download_data.py")


def load_travelplanner_dir(data_dir: str, split: str = "validation", limit: int | None = None) -> list:
    """split: 'train' (45) | 'validation' (180) | 'all' (train + validation = 225).
    (The 1,000-query 'test' split has no reference answers; use it only via the leaderboard.)"""
    parts = ["train", "validation"] if split == "all" else [split]
    out = []
    for p in parts:
        qs = load_travelplanner_csv(_split_file(data_dir, p))
        for q in qs:
            q._split = p
        out += qs
    return out[:limit] if limit else out
