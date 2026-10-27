"""
CaseRetriever: nearest-neighbor retrieval over the case library.

Deliberately dependency-light: rather than requiring a downloaded embedding
model (sentence-transformers etc., which needs network access this sandbox
may not have), we build a similarity score from STRUCTURED query features
(days, budget-per-day-per-person, people count, which constraint types are
present) plus a simple bag-of-words overlap on constraint values (cuisine
names, room type, etc.). This is fast, fully offline, and -- since queries
in this domain are highly structured -- captures the similarity that
actually matters for seeding agent proposals.

If you later want embedding-based retrieval instead (e.g. once you have
internet access to pull a sentence-transformers model, or you're using the
OpenAI embeddings API), swap `CaseRetriever._featurize` for an embedding
call and `_similarity` for cosine similarity -- the rest of the pipeline
(retrieve() -> case_to_hint()) is unchanged.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from data_model import Query
from retrieval.case_library import Case, case_to_hint


@dataclass
class RetrievedCase:
    case: Case
    similarity: float
    hint: dict


class CaseRetriever:
    def __init__(self, cases: list):
        self.cases = cases
        self._feature_cache = {id(c): self._featurize(c.query) for c in cases}

    # --------------------------------------------------------------------- #

    @staticmethod
    def _featurize(q: Query) -> dict:
        """Turn a Query into a small structured feature dict used for similarity."""
        budget_per_day_per_person = q.budget / max(1, q.days) / max(1, q.people_number)
        constraint_types = set()
        if q.room_rule:
            constraint_types.add("room_rule")
        if q.room_type:
            constraint_types.add("room_type")
        if q.cuisine:
            constraint_types.add("cuisine")
        if q.transportation_pref:
            constraint_types.add("transportation")

        bag = set()
        if q.cuisine:
            bag.update(c.lower() for c in q.cuisine)
        if q.room_type:
            bag.add(q.room_type.lower())
        if q.room_rule:
            bag.add(q.room_rule.lower())
        if q.transportation_pref:
            bag.add(q.transportation_pref.lower())

        return {
            "days": q.days,
            "people_number": q.people_number,
            "budget_per_day_per_person": budget_per_day_per_person,
            "n_constraints": len(constraint_types),
            "constraint_types": constraint_types,
            "bag": bag,
        }

    @staticmethod
    def _similarity(fa: dict, fb: dict) -> float:
        """Blend of numeric closeness (days, budget/day/person, #constraints)
        and categorical overlap (constraint types + bag-of-words on values)."""
        def norm_diff(x, y, scale):
            return 1.0 - min(1.0, abs(x - y) / scale)

        numeric_score = (
            norm_diff(fa["days"], fb["days"], scale=7)
            + norm_diff(fa["people_number"], fb["people_number"], scale=8)
            + norm_diff(fa["budget_per_day_per_person"], fb["budget_per_day_per_person"], scale=500)
            + norm_diff(fa["n_constraints"], fb["n_constraints"], scale=4)
        ) / 4.0

        types_union = fa["constraint_types"] | fb["constraint_types"]
        types_overlap = (len(fa["constraint_types"] & fb["constraint_types"]) / len(types_union)
                          if types_union else 1.0)

        bag_union = fa["bag"] | fb["bag"]
        bag_overlap = len(fa["bag"] & fb["bag"]) / len(bag_union) if bag_union else 1.0

        return 0.5 * numeric_score + 0.3 * types_overlap + 0.2 * bag_overlap

    # --------------------------------------------------------------------- #

    def retrieve(self, query: Query, k: int = 1) -> list:
        """Return the top-k most similar cases, each with its distilled hint."""
        if not self.cases:
            return []
        qf = self._featurize(query)
        scored = []
        for c in self.cases:
            # leave-one-out: a query must never retrieve its OWN solved case
            if c.query is query or (query.query_text and c.query.query_text == query.query_text):
                continue
            sim = self._similarity(qf, self._feature_cache[id(c)])
            scored.append((sim, c))
        scored.sort(key=lambda t: -t[0])
        top = scored[:k]
        return [RetrievedCase(case=c, similarity=sim, hint=case_to_hint(c)) for sim, c in top]

    def retrieve_hint(self, query: Query) -> dict:
        """Convenience: retrieve the single best case and return just its hint,
        ready to pass straight into agent.propose(plan, sandbox, hint=...)."""
        top = self.retrieve(query, k=1)
        return top[0].hint if top else {}
