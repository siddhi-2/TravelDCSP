"""
Prompt Generator  --  box "(2) Initial Prompt" of the architecture diagram.

Appends, in this order:
  1. Instruction prompt      what the LLM planner must do
  2. Context                 facts pulled from the databases (hotels, restaurants, distances) for the
                             cities involved -- so the model reasons over data, not guesses
  3. Output format           the exact JSON the Reformatter expects
  4. Few-shot example        one worked example
  5. RAG-retrieved case      the most similar already-solved query (CaseRetriever), if a retriever is given

`build()` returns the parts separately (the UI shows them) plus the assembled system / user prompts.
"""
from __future__ import annotations
import statistics


class PromptGenerator:
    INSTRUCTION = (
        "You are the planner of a multi-agent travel system. The cities of the trip are already fixed. "
        "Decide the best ORDER in which to visit them: start with the city closest to the origin, avoid "
        "long hops, and prefer cities whose hotels and restaurants are cheaper when the budget is tight.")
    OUTPUT_FORMAT = ('Answer with JSON only, exactly like {"order": ["CityA", "CityB"]}. '
                     "Use ONLY the given city names, each exactly once.")
    FEW_SHOT = ('Worked example (different trip): origin Boston, 5 days, visit Albany and Syracuse; '
                'Albany is closer to Boston, so the answer is {"order": ["Albany", "Syracuse"]}')

    def __init__(self, retriever=None):
        self.retriever = retriever

    # ------------------------------------------------------------------ #
    @staticmethod
    def _context(query, candidates) -> str:
        from data import sandbox
        lines = []
        for c in candidates:
            try:
                hotels = sandbox.search_accommodations(c)
                rests = sandbox.search_restaurants(c)
                n = max(1, query.people_number)
                hp = statistics.median(h["price"] for h in hotels) if hotels else None
                rp = statistics.median(r["cost"] for r in rests) if rests else None
                dist = None
                try:
                    drive = sandbox.search_self_driving(query.org, c)
                    dist = drive[0].get("distance_km") if drive else None
                except Exception:                                            # noqa: BLE001
                    pass
                bits = [f"{len(hotels)} stays" + (f" (median ${hp:,.0f}/night)" if hp is not None else ""),
                        f"{len(rests)} restaurants" + (f" (median ${rp:,.0f}/person/meal)" if rp is not None else "")]
                if dist:
                    bits.append(f"{dist:,.0f} km from {query.org}")
                lines.append(f"- {c}: " + ", ".join(bits))
            except Exception:                                                # noqa: BLE001
                lines.append(f"- {c}: (no data)")
        return "\n".join(lines)

    def _rag(self, query) -> str:
        if self.retriever is None:
            return ""
        try:
            top = self.retriever.retrieve(query, k=1)
        except Exception:                                                    # noqa: BLE001
            return ""
        if not top:
            return ""
        r = top[0]
        q, hint = r.case.query, r.hint
        bits = []
        if hint.get("transport_mode"):
            bits.append(f"transport that worked: {hint['transport_mode']}")
        if hint.get("room_type"):
            bits.append(f"room type: {hint['room_type']}")
        if hint.get("cuisine_order"):
            bits.append("cuisines: " + ", ".join(hint["cuisine_order"][:3]))
        solved = "solved" if r.case.converged else "partly solved"
        return (f"Similar {solved} case (similarity {r.similarity:.2f}): {q.days}-day trip, {q.people_number} "
                f"people, budget ${q.budget:,.0f}" + (f"; {'; '.join(bits)}" if bits else "") + ".")

    # ------------------------------------------------------------------ #
    def build(self, query, candidates: list) -> dict:
        parts = {
            "instruction": self.INSTRUCTION,
            "context": self._context(query, candidates),
            "output_format": self.OUTPUT_FORMAT,
            "few_shot": self.FEW_SHOT,
            "rag_case": self._rag(query),
        }
        system = f"{parts['instruction']} {parts['output_format']}"
        user = (f"Origin: {query.org}\nTrip length: {query.days} days, {query.people_number} people, "
                f"budget ${query.budget:,.0f}\nCities to visit: {', '.join(candidates)}\n"
                f"Database context:\n{parts['context']}\n{parts['few_shot']}")
        if parts["rag_case"]:
            user += f"\n{parts['rag_case']}"
        # the last line keeps the exact 'Cities:' marker used by earlier tooling / tests
        user += f"\nCities: {', '.join(candidates)}"
        return {"parts": parts, "system": system, "user": user}