"""Sanity tests for the ABT solver:  python tests/test_abt.py"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.query_loader import load_synthetic_queries
from llm.interface import MockLLM
from plan_builder import build_initial_plan
from run_experiment import fresh_agents
from solver.abt_solver import solve_abt
from solver.dcsp_solver import solve_with_restarts
from eval.metrics import score_plan

def verify(plan):
    hv = []
    for a in fresh_agents():
        hv += a.check(plan)
    return score_plan(plan, True, hv), hv

llm = MockLLM()
qs = load_synthetic_queries(n=80, difficulty="mixed", seed=42)
agree = abt_only = neg_only = 0
stats = dict(conv=0, inf=0, other=0, msgs=0, nogoods=0)
bad = 0
for q in qs:
    base = build_initial_plan(q, llm)
    r = solve_abt(base.clone(), seed=1)
    sc, hv = verify(r.plan)
    neg, _, _ = solve_with_restarts(base.clone, fresh_agents, 4, 10)
    nsc, _ = verify(neg.plan)
    if r.converged:
        stats["conv"] += 1
        if not sc.final_pass:           # ABT said solved but the independent checker disagrees
            bad += 1
            print("INCONSISTENT:", q.query_text, [v.as_backprompt() for v in hv][:3])
    elif r.failure_reason == "proved_infeasible":
        stats["inf"] += 1
    else:
        stats["other"] += 1; print("other failure:", r.failure_reason)
    stats["msgs"] += r.messages; stats["nogoods"] += r.nogoods_generated
    agree += (sc.final_pass == nsc.final_pass)
    abt_only += (sc.final_pass and not nsc.final_pass)
    neg_only += (nsc.final_pass and not sc.final_pass)
n = len(qs)
print(f"queries {n}: ABT converged {stats['conv']}, proved infeasible {stats['inf']}, other {stats['other']}")
print(f"avg messages {stats['msgs']/n:.1f}, avg nogoods {stats['nogoods']/n:.2f}")
print(f"ABT final-pass == negotiation final-pass on {agree}/{n}; ABT-only {abt_only}; negotiation-only {neg_only}")
print("INCONSISTENT converged plans:", bad)
