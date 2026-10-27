"""
Generates the key thesis figures from a run of run_experiment.py:
  1. Final Pass Rate by system (bar chart)
  2. Final Pass Rate by difficulty tier, grouped by system
  3. Rounds-to-convergence vs. budget tightness (the "negotiation earns its
     keep under tight budgets" plot demonstrated in development)

Usage:
    python plot_results.py --n 100 --difficulty mixed
Requires matplotlib (pip install matplotlib).
"""

from __future__ import annotations
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from run_experiment import main as run_main
from data_model import Query
from plan_builder import build_initial_plan
from llm.interface import get_llm
from agents.concrete import BudgetAgent, HotelAgent, TransportAgent, FoodAgent, AttractionAgent
from solver.dcsp_solver import DCSPSolver


def plot_final_pass_by_system(aggregates: dict, out="fig_final_pass_by_system.png"):
    names = list(aggregates.keys())
    values = [aggregates[n].as_dict()["final_pass_rate"] for n in names]
    plt.figure(figsize=(8, 5))
    bars = plt.bar(names, values, color=["#888", "#c99", "#8ac", "#4a7", "#48a"][: len(names)])
    plt.ylabel("Final Pass Rate (%)")
    plt.title("Final Pass Rate by System")
    plt.xticks(rotation=20, ha="right")
    for b, v in zip(bars, values):
        plt.text(b.get_x() + b.get_width() / 2, v + 0.5, f"{v:.1f}%", ha="center")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")


def plot_pass_by_difficulty(logs: list, out="fig_final_pass_by_difficulty.png"):
    systems = sorted(set(l.system for l in logs))
    diffs = ["easy", "medium", "hard"]
    width = 0.8 / len(systems)
    plt.figure(figsize=(9, 5))
    for i, sysname in enumerate(systems):
        vals = []
        for d in diffs:
            rows = [l.final_pass for l in logs if l.system == sysname and l.difficulty == d]
            vals.append(100 * sum(rows) / len(rows) if rows else 0)
        xs = [j + i * width for j in range(len(diffs))]
        plt.bar(xs, vals, width=width, label=sysname)
    plt.xticks([j + width * (len(systems) - 1) / 2 for j in range(len(diffs))], diffs)
    plt.ylabel("Final Pass Rate (%)")
    plt.title("Final Pass Rate by Difficulty Tier")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")


def plot_rounds_vs_budget(out="fig_rounds_vs_budget_tightness.png", n_points=12):
    """Sweeps budget tightness for a fixed trip and plots rounds-to-convergence,
    demonstrating negotiation's value scales with constraint difficulty."""
    llm = get_llm("mock")

    def fresh():
        return [BudgetAgent(), HotelAgent(), TransportAgent(), FoodAgent(), AttractionAgent()]

    budgets = [300 + i * 150 for i in range(n_points)]
    rounds, converged_flags = [], []
    for b in budgets:
        q = Query(org="Springfield", dest="Fairview", days=3, visiting_city_number=1,
                  date=["2022-03-01", "2022-03-02", "2022-03-03"], people_number=1,
                  budget=float(b), query_text="sweep", local_constraint={})
        plan = build_initial_plan(q, llm)
        solver = DCSPSolver(fresh(), negotiate=True, max_rounds=15)
        result = solver.run(plan)
        rounds.append(result.rounds_used)
        converged_flags.append(result.converged)

    plt.figure(figsize=(8, 5))
    colors = ["#4a7" if c else "#c33" for c in converged_flags]
    plt.bar([str(b) for b in budgets], rounds, color=colors)
    plt.xlabel("Budget ($)")
    plt.ylabel("Rounds to resolve (converged=green, exhausted=red)")
    plt.title("Negotiation Rounds vs. Budget Tightness")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")


if __name__ == "__main__":
    aggregates, logs = run_main()
    plot_final_pass_by_system(aggregates)
    plot_pass_by_difficulty(logs)
    plot_rounds_vs_budget()
