"""
Main experiment runner.

Runs each configured system across a set of queries and prints a
TravelPlanner-style results table, plus DCSP-specific metrics
(rounds-to-convergence, LLM-call counts) for your thesis tables.

Usage:
    python run_experiment.py                      # quick smoke test, 20 synthetic queries
    python run_experiment.py --n 180 --difficulty mixed
    python run_experiment.py --llm real --model gpt-4-turbo   # use a real LLM (needs OPENAI_API_KEY)

Systems run by default:
    A: Direct           -- agents propose once, no checking/fixing at all
    B: LLM-Modulo        -- reimplemented baseline (independent critics + backprompt)
    C: DCSP-NoNegotiate -- agents check independently, no concession
    D: DCSP-Negotiate   -- full negotiation loop (this project's core contribution)
"""

from __future__ import annotations
import argparse
import time
from dataclasses import dataclass

from data.query_loader import load_synthetic_queries, load_travelplanner_csv, load_travelplanner_dir
from data_model import Query
from plan_builder import build_initial_plan
from llm.interface import get_llm
from agents.concrete import BudgetAgent, HotelAgent, TransportAgent, FoodAgent, AttractionAgent
from solver.dcsp_solver import DCSPSolver, solve_with_restarts
from solver.abt_solver import solve_abt
from baseline.llm_modulo import LLMModuloBaseline
from eval.metrics import score_plan, AggregateResults
from retrieval.case_library import build_case_library
from retrieval.retriever import CaseRetriever


def fresh_agents():
    return [BudgetAgent(), HotelAgent(), TransportAgent(), FoodAgent(), AttractionAgent()]


@dataclass
class RunLog:
    system: str
    query_idx: int
    difficulty: str
    delivered: bool
    final_pass: bool
    rounds_or_iters: int
    llm_calls: int
    wall_time_s: float


def run_system_A_direct(query: Query, llm) -> tuple:
    """System A: propose once, never check or fix. Cheapest, weakest baseline."""
    t0 = time.time()
    plan = build_initial_plan(query, llm)
    agents = fresh_agents()
    for a in agents:
        a.propose(plan, __import__("data.sandbox", fromlist=["*"]))
    hard_violations = []
    for a in agents:
        hard_violations += a.check(plan)
    score = score_plan(plan, delivered=True, hard_violations_from_agents=hard_violations)
    return score, 1, 1, time.time() - t0


def run_system_B_llm_modulo(query: Query, llm, critic_set="all") -> tuple:
    """System B: LLM-Modulo reimplementation."""
    t0 = time.time()
    plan = build_initial_plan(query, llm)
    baseline = LLMModuloBaseline(llm, critic_set=critic_set, max_tries=10)
    result = baseline.run(plan)
    hard_violations = []
    for a in baseline.hard_agents:
        hard_violations += a.check(result.plan)
    score = score_plan(result.plan, delivered=result.converged or result.iterations_used > 0,
                        hard_violations_from_agents=hard_violations)
    return score, result.iterations_used, result.llm_calls, time.time() - t0


def run_system_CD_dcsp(query: Query, llm, negotiate: bool, retriever=None) -> tuple:
    """System C (negotiate=False), D (negotiate=True, no retriever),
    or E (negotiate=True, retriever=CaseRetriever -> RAG-seeded)."""
    t0 = time.time()
    plan = build_initial_plan(query, llm)
    agents = fresh_agents()
    solver = DCSPSolver(agents, negotiate=negotiate, max_rounds=10, retriever=retriever)
    result = solver.run(plan)
    hard_violations = []
    for a in agents:
        hard_violations += a.check(result.plan)
    score = score_plan(result.plan, delivered=True, hard_violations_from_agents=hard_violations)
    llm_calls = 1  # only the initial skeleton generation call -- solving itself is LLM-free
    return score, result.rounds_used, llm_calls, time.time() - t0


def run_system_F_restarts(query: Query, llm, retriever=None, max_restarts: int = 4) -> tuple:
    """System F: DCSP negotiation + multi-start restarts + (optional) one LLM escalation."""
    t0 = time.time()
    base = build_initial_plan(query, llm)
    llm_calls = 1
    escalations = {"n": 0}

    def escalate(plan, violations):
        new = llm.replan_with_backprompt(plan, violations)
        if getattr(llm, "is_real", False):      # only a real LLM actually costs a call
            escalations["n"] += 1
        return new

    result, restarts, escalated = solve_with_restarts(
        plan_factory=base.clone, agents_factory=fresh_agents, max_restarts=max_restarts,
        max_rounds=10, retriever=retriever, escalate=escalate)
    llm_calls += escalations["n"]
    agents = fresh_agents()
    hard = []
    for a in agents:
        hard += a.check(result.plan)
    score = score_plan(result.plan, delivered=True, hard_violations_from_agents=hard)
    return score, result.rounds_used + restarts, llm_calls, time.time() - t0


ABT_STATS: list = []     # one dict per ABT run (printed as an extra table)


def run_system_G_abt(query: Query, llm, seed: int = 0) -> tuple:
    """System G: TRUE Asynchronous Backtracking (ok?/nogood messages, priority order,
    nogood store, proof of infeasibility). 'rounds' reported = messages exchanged."""
    t0 = time.time()
    base = build_initial_plan(query, llm)
    res = solve_abt(base, tiers=12, seed=seed)
    hard = []
    for a in fresh_agents():
        hard += a.check(res.plan)
    score = score_plan(res.plan, delivered=True, hard_violations_from_agents=hard)
    ABT_STATS.append({"converged": res.converged, "messages": res.messages,
                      "nogoods": res.nogoods_generated, "changes": res.value_changes,
                      "reason": res.failure_reason})
    return score, res.messages, 1, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None,
                    help="number of queries (synthetic default 20; with --data_dir default = all in the split)")
    ap.add_argument("--difficulty", type=str, default="mixed", choices=["easy", "medium", "hard", "mixed"])
    ap.add_argument("--llm", type=str, default="mock", choices=["mock", "real","ollama"])
    ap.add_argument("--model", type=str, default="gpt-4-turbo")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_train", type=int, default=45, help="training queries for the case library (RAG corpus)")
    ap.add_argument("--data", type=str, default=None,
                    help="path to ONE real TravelPlanner query file (csv/jsonl/parquet); default = synthetic")
    ap.add_argument("--data_dir", type=str, default=None,
                    help="folder with train.csv / validation.csv (see scripts/download_data.py)")
    ap.add_argument("--split", type=str, default="all", choices=["train", "validation", "all"],
                    help="with --data_dir: train=45, validation=180, all=225 queries")
    ap.add_argument("--db", type=str, default=None,
                    help="path to the REAL TravelPlanner database folder (the 6 tool databases)")
    ap.add_argument("--no_rag", action="store_true", help="skip building System E (RAG-seeded)")
    args = ap.parse_args()

    if args.db:
        from data.real_sandbox import use_real_database
        print(f"Loading REAL TravelPlanner database from {args.db} ...")
        use_real_database(args.db)

    if args.data_dir:
        queries = load_travelplanner_dir(args.data_dir, split=args.split, limit=args.n)
    elif args.data:
        queries = load_travelplanner_csv(args.data, limit=args.n)
    else:
        queries = load_synthetic_queries(n=args.n or 20, difficulty=args.difficulty, seed=args.seed)
    print(f"Evaluating on {len(queries)} queries "
          f"({'real TravelPlanner' if (args.data_dir or args.data) else 'synthetic'}; "
          f"database: {'REAL' if args.db else 'synthetic'})")
    llm = get_llm(backend=args.llm, model=args.model)

    systems = {
        "A_Direct": lambda q: run_system_A_direct(q, llm),
        "B_LLM-Modulo": lambda q: run_system_B_llm_modulo(q, llm, critic_set="all"),
        "C_DCSP-NoNegotiate": lambda q: run_system_CD_dcsp(q, llm, negotiate=False),
        "D_DCSP-Negotiate": lambda q: run_system_CD_dcsp(q, llm, negotiate=True),
        "F_DCSP+Restarts": lambda q: run_system_F_restarts(q, llm),
        "G_ABT": lambda q: run_system_G_abt(q, llm),
    }

    if not args.no_rag:
        print("Building case library (RAG corpus for System E)...")
        if args.data_dir:
            train_queries = load_travelplanner_dir(args.data_dir, split="train")   # the 45 real train queries
        else:
            train_queries = load_synthetic_queries(n=args.n_train, difficulty="mixed", seed=args.seed + 1000)
        cases = build_case_library(train_queries, llm)
        retriever = CaseRetriever(cases)
        systems["E_DCSP-RAG"] = lambda q: run_system_CD_dcsp(q, llm, negotiate=True, retriever=retriever)

    aggregates = {name: AggregateResults() for name in systems}
    logs: list[RunLog] = []

    for qi, q in enumerate(queries):
        diff = getattr(q, "_difficulty", "n/a")
        for name, fn in systems.items():
            score, rounds, llm_calls, wall = fn(q)
            aggregates[name].add(score)
            logs.append(RunLog(name, qi, diff, score.delivered, score.final_pass, rounds, llm_calls, wall))

    print(f"\n{'='*90}\nResults over {len(queries)} queries (difficulty={args.difficulty}, llm={args.llm})\n{'='*90}")
    header = f"{'System':<22}{'Delivery%':>11}{'CS-micro':>10}{'CS-macro':>10}{'Hard-micro':>12}{'Hard-macro':>12}{'Final%':>9}"
    print(header)
    print("-" * len(header))
    for name, agg in aggregates.items():
        d = agg.as_dict()
        print(f"{name:<22}{d['delivery_rate']:>11.1f}{d['commonsense_micro']:>10.1f}"
              f"{d['commonsense_macro']:>10.1f}{d['hard_micro']:>12.1f}{d['hard_macro']:>12.1f}"
              f"{d['final_pass_rate']:>9.1f}")

    print(f"\n{'='*90}\nCost / efficiency (mean per query)\n{'='*90}")
    print(f"{'System':<22}{'Avg rounds/iters':>18}{'Avg LLM calls':>16}{'Avg wall time (s)':>20}")
    for name in systems:
        rows = [l for l in logs if l.system == name]
        avg_rounds = sum(l.rounds_or_iters for l in rows) / len(rows)
        avg_calls = sum(l.llm_calls for l in rows) / len(rows)
        avg_time = sum(l.wall_time_s for l in rows) / len(rows)
        print(f"{name:<22}{avg_rounds:>18.2f}{avg_calls:>16.2f}{avg_time:>20.4f}")

    if ABT_STATS:
        n_abt = len(ABT_STATS)
        reasons = {}
        for st in ABT_STATS:
            if not st["converged"]:
                reasons[st["reason"]] = reasons.get(st["reason"], 0) + 1
        print(f"\n{'='*90}\nABT statistics (System G)\n{'='*90}")
        print(f"  converged                : {100*sum(s['converged'] for s in ABT_STATS)/n_abt:.1f}%")
        print(f"  avg messages / query     : {sum(s['messages'] for s in ABT_STATS)/n_abt:.1f}")
        print(f"  avg nogoods / query      : {sum(s['nogoods'] for s in ABT_STATS)/n_abt:.2f}")
        print(f"  avg value changes        : {sum(s['changes'] for s in ABT_STATS)/n_abt:.2f}")
        print(f"  non-converged reasons    : {reasons or 'none'}   "
              f"(proved_infeasible = ABT PROVED no plan fits; empty_domain = an agent has no valid option)")

    print(f"\n{'='*90}\nFinal Pass Rate by difficulty tier\n{'='*90}")
    print(f"{'System':<22}{'Easy%':>10}{'Medium%':>10}{'Hard%':>10}")
    for name in systems:
        rows_by_diff = {"easy": [], "medium": [], "hard": []}
        for l in logs:
            if l.system == name and l.difficulty in rows_by_diff:
                rows_by_diff[l.difficulty].append(l.final_pass)
        cells = []
        for d in ("easy", "medium", "hard"):
            r = rows_by_diff[d]
            cells.append(100 * sum(r) / len(r) if r else float("nan"))
        print(f"{name:<22}{cells[0]:>10.1f}{cells[1]:>10.1f}{cells[2]:>10.1f}")

    return aggregates, logs


if __name__ == "__main__":
    main()
