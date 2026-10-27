# TravelPlanner DCSP-Negotiation Framework

A working reimplementation of the **LLM-Modulo** critic-loop idea from
Kambhampati et al. (2024) and Gundawar et al. (2024), replaced with a
**Distributed Constraint Satisfaction Problem (DCSP)** solver: instead of
independent critics that check-and-fail a plan one constraint at a time,
this system models each constraint category (Budget, Hotel, Transport, Food,
Attractions) as an **agent** that negotiates directly with whichever other
agent caused a conflict, until they converge on a plan or provably exhaust
the search space.

This is a **complete, runnable** project (not a sketch) built to be extended
into an M.Tech thesis. It runs fully offline with a mock LLM and a synthetic
sandbox database, so you can start experimenting immediately, then swap in
the real TravelPlanner dataset and a real LLM when you're ready.

---

## Quick start

```bash
pip install -r requirements.txt

# Smoke test -- 20 synthetic queries, mock LLM, all 5 systems
python run_experiment.py --n 20

# Bigger run + generates the 3 thesis figures
python plot_results.py --n 100 --difficulty mixed
```

No API key needed to get started. Everything (LLM calls, flight/hotel/
restaurant data) is mocked deterministically so results are reproducible.

---

## What's actually implemented

| Component | File | What it does |
|---|---|---|
| Data model | `data_model.py` | `Query`, `PlanState`, `DayPlan`, `Variable`, `ConstraintViolation` |
| Mock sandbox | `data/sandbox.py` | Stand-in for TravelPlanner's flight/hotel/restaurant/attraction tools |
| Synthetic queries | `data/query_loader.py` | Deterministic query generator (Easy/Medium/Hard tiers, matching TravelPlanner's own difficulty scheme) |
| Agents | `agents/concrete.py` | `BudgetAgent`, `HotelAgent`, `TransportAgent`, `FoodAgent`, `AttractionAgent` |
| **DCSP solver (core contribution)** | `solver/dcsp_solver.py` | The propose → check → conflict-signal → concede negotiation loop |
| LLM-Modulo baseline | `baseline/llm_modulo.py` | Faithful reimplementation of the critics + metacontroller + backprompt loop, for fair head-to-head comparison |
| RAG / case-based seeding | `retrieval/retriever.py`, `retrieval/case_library.py` | Retrieves the nearest previously-solved query and seeds agent proposals from it, instead of a cold start |
| LLM interface | `llm/interface.py` | `MockLLM` (offline, no key needed) and `RealLLM` (OpenAI-compatible, for when you're ready) |
| Evaluation | `eval/metrics.py` | TravelPlanner-style Delivery Rate, Commonsense/Hard Pass Rate (micro + macro), Final Pass Rate |
| Experiment runner | `run_experiment.py` | Runs Systems A–E across N queries, prints the full results table |
| Plotting | `plot_results.py` | Generates the 3 key thesis figures |

## The five systems (your ablation table)

| System | Negotiation? | RAG-seeded? | Purpose |
|---|---|---|---|
| **A. Direct** | No | No | Weakest baseline — propose once, never check |
| **B. LLM-Modulo** | N/A (regenerate whole plan) | No | Reimplemented baseline to beat |
| **C. DCSP-NoNegotiate** | No | No | Isolates "does formalizing as DCSP alone help" |
| **D. DCSP-Negotiate** | Yes | No | **This project's core contribution** |
| **E. DCSP-RAG** | Yes | Yes | Case-based initialization from the 45-query training corpus |

---

## Honest results from the mock sandbox (read this before you present anything)

Running `python plot_results.py --n 80` produces:

```
System                  Delivery%  CS-micro  CS-macro  Hard-micro  Hard-macro   Final%
A_Direct                    100.0      89.1      27.5        96.3        92.5     26.2
B_LLM-Modulo                100.0      89.1      27.5        96.9        93.8     26.2
C_DCSP-NoNegotiate          100.0      89.1      27.5        96.3        92.5     26.2
D_DCSP-Negotiate            100.0      89.1      27.5        96.9        93.8     26.2
E_DCSP-RAG                  100.0      87.7      17.5        96.9        93.8     16.2
```

**Two things to understand, not hide, in your thesis:**

1. **On a random mixed-difficulty sample, A/C/D look identical.** This is
   expected, not a bug: negotiation only does work when the *initial*
   proposal actually violates a constraint. Most randomly-generated queries
   in the mock sandbox aren't near a real constraint boundary, so there's
   nothing to negotiate. The real proof that negotiation works is the
   **targeted budget-tightness sweep** (`fig_rounds_vs_budget_tightness.png`),
   which shows negotiation rounds climbing as budget tightens, and clean
   `domain_exhaustion` reporting (not thrashing) when a budget is genuinely
   infeasible. **For your thesis, report both**: aggregate metrics on random
   queries AND a stress-test sweep like this one, exactly like the
   "constraint density stress test" from the project planning discussion.

2. **System E (RAG) underperforms D on this synthetic data (16.2% vs 26.2%).**
   This is a real, reproducible finding worth investigating and reporting
   honestly rather than dropping — likely causes to explore:
   - Only 45 training cases is a very small corpus for meaningful retrieval
   - `case_to_hint()` distills a whole solved plan into one blended hint
     (dominant cuisine/room-type/mode), which can override a perfectly good
     median-cost cold start with a worse one if the "nearest" retrieved case
     isn't actually that similar
   - Possible fixes to try and report as your own ablation: only apply the
     hint when retrieval similarity is above some threshold; blend hint-guided
     and cold-start proposals rather than fully replacing one with the other;
     grow the case library size and re-test.

   This is legitimate, presentable thesis material — a clean negative result
   with a documented hypothesis for *why*, plus concrete next steps, is
   exactly what a good ablation section looks like.

---

## Bugs found and fixed during development (also worth a paragraph in your thesis)

These are genuine DCSP pitfalls, not incidental implementation slips — worth
documenting as lessons learned in your methodology chapter:

1. **Missing field coverage** — no agent originally filled the `attraction`
   field at all; added `AttractionAgent`.
2. **Domain-pool starvation** — mock restaurant/attraction pools were too
   small for multi-day trips, causing *artificial* diversity failures
   unrelated to the negotiation logic itself.
3. **Same-slot blind spot** — `concede()` checked other *days* for duplicate
   picks but not other *meals on the same day*, letting an agent swap into a
   duplicate it had just created itself.
4. **Two-cycle oscillation** — without memory of previously-rejected
   candidates, `HotelAgent`/`TransportAgent` alternated between the two
   cheapest options forever instead of making monotonic progress or failing
   cleanly. Fixed with a `_tried` set per agent instance.
5. **Wrong concession direction** — conceding to a Budget violation must mean
   "find something *cheaper*"; a generic "try any untried alternative" can
   only get *more* expensive once the cheapest option is already in use.
   Fixed by branching `concede()` logic on `violation.critic_name`.
6. **Overly-optimal cold start** — agents originally proposed the *cheapest*
   candidate immediately, leaving no room for negotiation to ever show value
   (a plan was either already-optimal-and-fits, or already-optimal-and-still-
   over-budget). Changed to a median-cost initial proposal, which is also a
   more realistic stand-in for an LLM's first guess.

---

## Extending this for a full M.Tech submission

Ranked by value-to-effort (see prior discussion for full detail):

**High priority**
- [ ] Swap `data/query_loader.py` and `data/sandbox.py` for the **real
      TravelPlanner data** (see stub `load_travelplanner_csv()` — fill in the
      column mapping once you've downloaded `osunlp/TravelPlanner` from
      Hugging Face)
- [ ] Switch `--llm real --model gpt-4-turbo` (needs `OPENAI_API_KEY`) once
      the architecture is validated on mock data
- [ ] Add McNemar's test comparing System D vs. B on paired pass/fail per query
- [ ] Investigate and fix the System E underperformance (see above)

**Good differentiators**
- [ ] Failure mode taxonomy — categorize `domain_exhaustion` vs `max_rounds`
      cases by which agent/critic was the bottleneck (the `violations_log`
      in every `SolverResult` already gives you this data — just aggregate it)
- [ ] Learned concession priority — replace the hardcoded `priority` class
      attribute on each agent with a small learned policy
- [ ] Second synthetic domain (e.g. meeting scheduling) to test generalization

**Polish**
- [ ] Statistical significance testing, cost-per-query reporting (already
      partially there via `Avg LLM calls` in the results table)
- [ ] A simple FastAPI + visualization demo for your defense presentation

---

## Project structure

```
travelplanner_dcsp/
├── data_model.py            # Query, PlanState, DayPlan, Variable, ConstraintViolation
├── plan_builder.py          # Query + LLM skeleton -> initial PlanState
├── run_experiment.py        # Main entry point -- runs Systems A-E
├── plot_results.py          # Generates the 3 thesis figures
├── requirements.txt
├── agents/
│   ├── base.py               # ConstraintAgent abstract base
│   └── concrete.py           # Budget, Hotel, Transport, Food, Attraction agents
├── solver/
│   └── dcsp_solver.py        # DCSPSolver -- the negotiation loop (core contribution)
├── baseline/
│   └── llm_modulo.py         # Reimplemented LLM-Modulo baseline for fair comparison
├── retrieval/
│   ├── case_library.py       # Builds a corpus of solved (query, plan) cases
│   └── retriever.py          # CaseRetriever -- nearest-neighbor case retrieval (RAG)
├── llm/
│   └── interface.py          # MockLLM (offline) / RealLLM (OpenAI API)
├── data/
│   ├── sandbox.py            # Mock flight/hotel/restaurant/attraction database
│   └── query_loader.py       # Synthetic query generator + real-dataset loader stub
└── eval/
    └── metrics.py            # TravelPlanner-style scoring (micro/macro pass rates)
```

## CLI reference

```bash
python run_experiment.py \
  --n 180 \                    # number of queries (use 180 to match TravelPlanner's validation set size)
  --difficulty mixed \          # easy | medium | hard | mixed
  --llm mock \                  # mock (offline) | real (needs OPENAI_API_KEY)
  --model gpt-4-turbo \         # only used when --llm real
  --seed 42 \                   # for reproducibility
  --n_train 45 \                # training queries for the RAG case library
  --no_rag                      # skip building System E entirely
```


---

## Running on the REAL TravelPlanner data (225 queries + all 6 databases)

**1. Download** (needs internet; ~ a few hundred MB for the database):
```
pip install -r requirements.txt
python scripts/download_data.py
```
This creates `data/travelplanner/train.csv` (45), `validation.csv` (180) and `database/`
(FlightSearch, GoogleDistanceMatrix, AccommodationSearch, RestaurantSearch,
AttractionSearch, CitySearch data). 45 + 180 = **225 queries** (the 1,000 test queries have
no reference answers, so they are leaderboard-only).

**2. Run all 225 queries through every system**
```
python run_experiment.py --data_dir data/travelplanner --db data/travelplanner/database --split all
python run_experiment.py --data_dir data/travelplanner --db data/travelplanner/database --split validation   # 180 only
```
`--split validation` builds the RAG case library from the 45 train queries; `--split all` uses
leave-one-out retrieval (a query never retrieves its own case).

**3. Single query demo on real data**
```
python demo.py --db data/travelplanner/database --source Sarasota --dest Chicago --days 3 --people 2 --budget 1800
python demo.py --db data/travelplanner/database --source Sarasota --dest Ohio --cities 2 --days 5 --people 2 --budget 5000
```
Use cities/states that exist in the database; for 2-3 cities `--dest` is a US **state**.

**4. Official scores** (what the thesis should report)
```
python export_submission.py --data_dir data/travelplanner --db data/travelplanner/database --split validation --out plans.jsonl
```
then score `plans.jsonl` with the official evaluator in github.com/OSU-NLP-Group/TravelPlanner
(`evaluation/eval.py`). The metrics in `eval/metrics.py` are structural approximations.

### What the real-data mode changes
| Item | Handling |
|---|---|
| Cost model | flights per person; self-driving (5 seats, 0.05/km) and taxi (4 seats, 1.0/km) per vehicle; hotels per room x ceil(people / max occupancy) |
| `house rule: smoking` | stay must NOT list "No smoking" |
| `room type: not shared room` | negated match supported |
| Restaurants with several cuisines | a cuisine is covered if it is in the restaurant's cuisine list |
| Transport | taxi allowed under every preference; flights and self-driving never mixed in one trip |
| Multi-city (5/7 days) | `dest` is a state; nearest usable cities are chosen; day pattern travel/stay/.../return |

### Testing without the download
`python tests/make_fixture.py data/fixture` creates a tiny fake dataset with the same layout and
column names, then run `python run_experiment.py --db data/fixture/database --data_dir data/fixture`.

### Honest limitations
* The real-data loader was written from the published schema and **tested only on that fixture**;
  if a column name differs in your download, the loader stops with an error that lists the real
  columns (edit the alias lists in `data/real_sandbox.py`).
* With the mock LLM the city skeleton is rule-based and the LLM-Modulo baseline's regenerate step
  is a deterministic stand-in. Use `--llm real` for a faithful baseline comparison.


---

## True ABT (Asynchronous Backtracking) -- `solver/abt_solver.py`

`dcsp_solver.py` is a round-based negotiation heuristic *inspired by* ABT. `abt_solver.py` is the
real algorithm (Yokoo et al.), selectable as **System G** in `run_experiment.py` and with `--solver abt` in `demo.py`.

| ABT ingredient | Implementation |
|---|---|
| Agents / variables | transport, hotel, food, attraction -- each owns ONE compound variable (a bundle of all its choices) |
| Fixed priority order | transport > hotel > food > attraction |
| Domain | candidate bundles in 12 cost tiers; agent-internal constraints (room type/rule, min nights, cuisine, no repeats, transport preference, no flight+self-drive mix) are enforced when the domain is built, using each agent's own `check()` |
| Value order | median bundle first, then outward |
| Inter-agent constraint | budget: cost(transport)+cost(hotel)+cost(food) <= budget, evaluated by the LOWEST-priority agent in its scope (food) |
| Messages | `ok?`, `nogood`, `add_link` over per-pair FIFO channels, delivered in random order (asynchrony, reproducible with a seed) |
| Nogoods | stored; resolvent built on dead ends and sent to the lowest-priority agent in it |
| Termination | quiescence = solved; **empty nogood = proved infeasible** |

```
python demo.py --source Mumbai --dest Goa --days 3 --people 3 --budget 25000 --currency INR --solver abt
python run_experiment.py --n 100            # System G_ABT appears in every table + an ABT statistics table
python tests/test_abt.py                    # soundness check against independent verification
```

Verified against brute force over all bundle combinations: 610 runs (random delivery orders,
synthetic + multi-city real-layout fixture) with 0 mismatches.

Limits to state honestly: ABT is complete *over the discretised bundle domain* (12 tiers), the agents run
in one process (simulated asynchrony, not a real network), and constraints other than budget are handled
at domain-construction time rather than by message passing.


### Testing on the real dataset
```
python test_real_dataset.py --data_dir data/travelplanner --db data/travelplanner/database
python test_real_dataset.py --data_dir data/travelplanner --db data/travelplanner/database --limit 20           # smoke test
python test_real_dataset.py --data_dir data/travelplanner --db data/travelplanner/database --bruteforce 15 --seeds 3
```
Preflight (database sizes, queries naming places missing from the database) -> runs ABT and negotiation on every
query (a crash on one query never stops the run) -> re-verifies every plan ABT calls solved with independent checks
-> reports pass rates by difficulty/split/constraint, ABT failure reasons, messages/nogoods, and saves a per-query CSV.
Exit code 1 means a verification failure or crash (a bug worth reporting).
