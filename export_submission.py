# """
# Export plans in TravelPlanner's official submission format (jsonl) so you can score
# them with the OFFICIAL evaluator (OSU-NLP-Group/TravelPlanner -> evaluation/eval.py).
# This gives you the numbers a thesis should report -- the metrics in eval/metrics.py
# are structural approximations.

#     python export_submission.py --data_dir data/travelplanner --db data/travelplanner/database \
#         --split validation --out plans_validation.jsonl

# BEST EFFORT: the string layout follows the official sample submissions. Check 2-3 lines
# against the official eval script before trusting the numbers.
# """
# import argparse
# import json

# from data.query_loader import load_travelplanner_dir
# from data.real_sandbox import use_real_database
# from llm.interface import get_llm
# from plan_builder import build_initial_plan
# from run_experiment import fresh_agents
# from solver.dcsp_solver import solve_with_restarts


# def fmt_transport(t, org, dst):
#     if not t:
#         return "-"
#     if t["mode"] == "flight":
#         return (f"Flight Number: {t['Flight Number']}, from {org} to {dst}, "
#                 f"Departure Time: {t['DepTime']}, Arrival Time: {t['ArrTime']}")
#     label = "Self-driving" if t["mode"] == "self-driving" else "Taxi"
#     return f"{label}, from {org} to {dst}, Distance: {t['distance_km']:.0f}km, Cost: {t['cost']:.0f}"


# def day_to_official(dp):
#     org = dst = None
#     if " to " in dp.current_city:
#         org, dst = dp.current_city.replace("from ", "", 1).split(" to ", 1)
#     meal = lambda m: f"{m['Name']}, {m['City']}" if m else "-"          # noqa: E731
#     attr = ";".join(f"{a['Name']}, {a['City']}" for a in (dp.attraction or [])) + ";" if dp.attraction else "-"
#     acc = f"{dp.accommodation['NAME']}, {dp.accommodation['City']}" if dp.accommodation else "-"
#     return {"days": dp.day, "current_city": dp.current_city,
#             "transportation": fmt_transport(dp.transportation, org, dst),
#             "breakfast": meal(dp.breakfast), "attraction": attr,
#             "lunch": meal(dp.lunch), "dinner": meal(dp.dinner), "accommodation": acc}


# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--data_dir", required=True)
#     ap.add_argument("--db", required=True)
#     ap.add_argument("--split", default="validation", choices=["train", "validation", "all"])
#     ap.add_argument("--out", default="plans.jsonl")
#     ap.add_argument("--limit", type=int, default=None)
#     a = ap.parse_args()
#     use_real_database(a.db)
#     llm = get_llm("mock")
#     qs = load_travelplanner_dir(a.data_dir, a.split, a.limit)
#     with open(a.out, "w") as f:
#         for i, q in enumerate(qs, 1):
#             base = build_initial_plan(q, llm)
#             res, _, _ = solve_with_restarts(base.clone, fresh_agents, 4, 10)
#             f.write(json.dumps({"idx": i, "query": q.query_text,
#                                 "plan": [day_to_official(res.plan.days_plan[d]) for d in sorted(res.plan.days_plan)]}) + "\n")
#     print(f"wrote {len(qs)} plans -> {a.out}")


# if __name__ == "__main__":
#     main()

"""
Export plans in TravelPlanner's official submission format (jsonl) so you can score
them with the OFFICIAL evaluator (OSU-NLP-Group/TravelPlanner -> evaluation/eval.py).
This gives you the numbers a thesis should report -- the metrics in eval/metrics.py
are structural approximations.

    python export_submission.py --llm ollama --model llama3.2:1b --data_dir data/travelplanner --db data/travelplanner/database \
        --split validation --out plans_validation.jsonl

BEST EFFORT: the string layout follows the official sample submissions. Check 2-3 lines
against the official eval script before trusting the numbers.
"""
import argparse
import json

from data.query_loader import load_travelplanner_dir
from data.real_sandbox import use_real_database
from llm.interface import get_llm
from plan_builder import build_initial_plan
from run_experiment import fresh_agents
from solver.dcsp_solver import solve_with_restarts


def fmt_transport(t, org, dst):
    if not t:
        return "-"
    if t["mode"] == "flight":
        return (f"Flight Number: {t['Flight Number']}, from {org} to {dst}, "
                f"Departure Time: {t['DepTime']}, Arrival Time: {t['ArrTime']}")
    label = "Self-driving" if t["mode"] == "self-driving" else "Taxi"
    return f"{label}, from {org} to {dst}, Distance: {t['distance_km']:.0f}km, Cost: {t['cost']:.0f}"


def day_to_official(dp):
    org = dst = None
    if " to " in dp.current_city:
        org, dst = dp.current_city.replace("from ", "", 1).split(" to ", 1)
    meal = lambda m: f"{m['Name']}, {m['City']}" if m else "-"          # noqa: E731
    attr = ";".join(f"{a['Name']}, {a['City']}" for a in (dp.attraction or [])) + ";" if dp.attraction else "-"
    acc = f"{dp.accommodation['NAME']}, {dp.accommodation['City']}" if dp.accommodation else "-"
    return {"days": dp.day, "current_city": dp.current_city,
            "transportation": fmt_transport(dp.transportation, org, dst),
            "breakfast": meal(dp.breakfast), "attraction": attr,
            "lunch": meal(dp.lunch), "dinner": meal(dp.dinner), "accommodation": acc}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--db", required=True)
    ap.add_argument("--split", default="validation", choices=["train", "validation", "all"])
    ap.add_argument("--out", default="plans.jsonl")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--llm", default="ollama", choices=["mock", "real", "ollama"])
    ap.add_argument("--model", default="llama3.2:1b")
    a = ap.parse_args()
    use_real_database(a.db)
    llm = get_llm(a.llm, a.model)
    qs = load_travelplanner_dir(a.data_dir, a.split, a.limit)
    with open(a.out, "w") as f:
        for i, q in enumerate(qs, 1):
            base = build_initial_plan(q, llm)
            res, _, _ = solve_with_restarts(base.clone, fresh_agents, 4, 10)
            print(f"  [{i}/{len(qs)}] converged={res.converged}", flush=True)
            f.write(json.dumps({"idx": i, "query": q.query_text,
                                "plan": [day_to_official(res.plan.days_plan[d]) for d in sorted(res.plan.days_plan)]}) + "\n")
    print(f"wrote {len(qs)} plans -> {a.out}")


if __name__ == "__main__":
    main()
