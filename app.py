"""
Streamlit UI for the Coordinated Multi-Agent DCSP travel planner.

    pip install streamlit
    streamlit run app.py

Enter constraints, pick the cities, press "Plan my trip", then watch the architecture run:
  1 Prompt & LLM planner   2 Agents propose   3 Negotiation (every round, every concession)   4 Final itinerary
"""
from __future__ import annotations
import os

import pandas as pd
import streamlit as st

from data_model import Query
from llm.interface import get_llm
from pipeline import run_pipeline, AGENT_ORDER

st.set_page_config(page_title="Multi-Agent DCSP Travel Planner", layout="wide", page_icon="🧭")

ROOM_TYPES = ["(any)", "Entire Room", "Private Room", "Shared Room", "Not Shared Room"]
ROOM_RULES = ["(none)", "No parties", "No smoking", "No children under 10", "No pets", "No visitors"]
TRANSPORT = ["(any)", "No flight", "No self-driving"]
KIND_NAME = {"transportation": "Transport", "accommodation": "Hotel", "breakfast": "Breakfast", "lunch": "Lunch",
             "dinner": "Dinner", "attraction": "Attractions"}
AGENT_ICON = {"budget": "💰 Budget", "transport": "✈️ Transport", "hotel": "🏨 Hotel", "food": "🍽️ Food",
              "attraction": "🎡 Attraction"}


# --------------------------------------------------------------------------- #
# cached resources
# --------------------------------------------------------------------------- #

@st.cache_resource(show_spinner="Loading the six TravelPlanner databases ...")
def load_database(path: str):
    from data.real_sandbox import use_real_database, valid_places
    from data import real_sandbox as RS
    from data_model import cuisines_of
    use_real_database(path)
    cities, states = valid_places()
    cuisines = set()
    for rows in RS._db.restaurants().values():
        for r in rows:
            cuisines |= cuisines_of(r)
    usable = {c for c in cities if RS._city_is_usable(c)}          # has hotels, restaurants and attractions
    return (sorted(cities), {k: sorted(c for c in v if c in usable) for k, v in states.items()},
            sorted(cuisines), sorted(usable))


@st.cache_resource(show_spinner="Starting the LLM backend ...")
def load_llm(backend: str, model: str):
    return get_llm(backend, model)


@st.cache_resource(show_spinner="Building the RAG case library from the 45 training queries ...")
def load_retriever(db: str, data_dir: str, backend: str, model: str):
    from data.query_loader import load_travelplanner_dir
    from retrieval.case_library import build_case_library
    from retrieval.retriever import CaseRetriever
    llm = load_llm(backend, model)
    cases = build_case_library(load_travelplanner_dir(data_dir, split="train"), llm)
    return CaseRetriever(cases)


@st.cache_data(show_spinner=False)
def load_queries(data_dir: str, split: str):
    from data.query_loader import load_travelplanner_dir
    return load_travelplanner_dir(data_dir, split)


# --------------------------------------------------------------------------- #
# sidebar
# --------------------------------------------------------------------------- #

st.sidebar.title("⚙️ Settings")
default_db = next((p for p in ("data/travelplanner/database", "data/fixture/database") if os.path.isdir(p)), "")
db_path = st.sidebar.text_input("Database folder (6 sources)", default_db,
                                help="Leave empty to use the small synthetic sandbox.")
data_dir = st.sidebar.text_input("Queries folder (train.csv / validation.csv)",
                                 "data/travelplanner" if os.path.isdir("data/travelplanner")
                                 else ("data/fixture" if os.path.isdir("data/fixture") else ""))
backend = st.sidebar.selectbox("LLM planner", ["ollama", "mock"], help="ollama = local model, no API cost")
model = st.sidebar.text_input("Ollama model", "llama3.2:1b") if backend == "ollama" else "n/a"
solver_label = st.sidebar.radio("Solver", ["DCSP negotiation (agents concede)", "ABT (asynchronous backtracking)"])
solver = "abt" if solver_label.startswith("ABT") else "negotiate"
max_rounds = st.sidebar.slider("Max negotiation rounds", 1, 20, 10)
restarts = st.sidebar.slider("Restarts", 0, 6, 4)
use_rag = st.sidebar.checkbox("Add RAG-retrieved similar case to the prompt / seed agents", value=False,
                              help="Builds a case library from the 45 training queries (needs the queries folder).")
cur = st.sidebar.radio("Show amounts in", ["USD", "INR"], horizontal=True)
rate = st.sidebar.number_input("1 USD = ? INR", 50.0, 150.0, 85.0) if cur == "INR" else 1.0
SYM = "Rs " if cur == "INR" else "USD "   # no "$": Streamlit markdown would read $...$ as math


def money(x) -> str:
    return f"{SYM}{x * rate:,.0f}"


# --------------------------------------------------------------------------- #
# inputs
# --------------------------------------------------------------------------- #

st.title("🧭 Coordinated Multi-Agent DCSP Travel Planner")
st.caption("LLM planner  →  Budget / Hotel / Transport / Food (+ Attraction) agents negotiate  →  valid itinerary")

cities = states = cuisines = dest_cities = None
if db_path:
    try:
        cities, states, cuisines, dest_cities = load_database(db_path)
    except Exception as e:                                                  # noqa: BLE001
        st.error(f"Could not load the database at '{db_path}': {e}")
        st.stop()
else:
    st.info("No database folder given: using the synthetic sandbox (any city name works).")

mode = st.radio("What do you want to plan?", ["My own trip", "A TravelPlanner dataset query"], horizontal=True)
query = None

if mode == "My own trip":
    c1, c2, c3 = st.columns(3)
    with c1:
        origin = st.selectbox("From (origin city)", cities) if cities else st.text_input("From", "Springfield")
        trip = st.radio("Trip type", ["One city", "Several cities in one state"])
    chosen, k = None, 1
    with c2:
        if trip == "One city":
            dest = st.selectbox("To (destination city)", dest_cities, index=min(1, len(dest_cities) - 1)) if cities \
                else st.text_input("To", "Fairview")
            days = st.slider("Days", 2, 7, 3)
        else:
            k = st.radio("How many cities?", [2, 3], horizontal=True)
            if states:
                dest = st.selectbox("State", sorted(states))
                picks = st.multiselect(f"Choose {k} cities (or leave empty: the system picks)", states[dest],
                                       max_selections=k)
                chosen = picks if len(picks) == k else None
                if picks and len(picks) != k:
                    st.warning(f"Pick exactly {k} cities, or clear the list to let the system choose.")
            else:
                dest = st.text_input("State", "Texas")
            days = 2 * k + 1
            st.caption(f"{k} cities → {days} days (TravelPlanner pattern: travel, stay, ..., return)")
    with c3:
        people = st.number_input("Travellers", 1, 8, 2)
        budget_in = st.number_input(f"Total budget ({cur})", 100.0, 10_000_000.0,
                                    float(round(450 * days * people * rate, -2)), step=100.0,
                                    key=f"budget_{days}_{people}_{cur}",
                                    help="Default = about 450 USD per person per day; lower it to force negotiation.")
    st.markdown("**Preferences (optional)**")
    d1, d2, d3, d4 = st.columns(4)
    cu = d1.multiselect("Cuisines wanted", cuisines or ["Italian", "Indian", "Chinese", "American", "Mexican"])
    rt = d2.selectbox("Room type", ROOM_TYPES)
    rr = d3.selectbox("House rule to avoid", ROOM_RULES)
    tr = d4.selectbox("Transport", TRANSPORT)
    lc = {}
    if cu:
        lc["cuisine"] = cu
    if rt != "(any)":
        lc["room type"] = rt
    if rr != "(none)":
        lc["room rule"] = rr
    if tr != "(any)":
        lc["transportation"] = tr
    budget = budget_in / rate
    query = Query(org=origin, dest=dest, days=days, visiting_city_number=k,
                  date=[f"2022-03-{d:02d}" for d in range(1, days + 1)], people_number=int(people), budget=budget,
                  query_text=f"{days}-day trip {origin} -> {dest} for {people} people, budget {budget:,.0f}",
                  local_constraint=lc, chosen_cities=chosen)
else:
    if not data_dir:
        st.warning("Set the queries folder in the sidebar (it must contain train.csv / validation.csv).")
        st.stop()
    e1, e2 = st.columns(2)
    split = e1.selectbox("Split", ["validation", "train"])
    try:
        qs = load_queries(data_dir, split)
    except Exception as e:                                                  # noqa: BLE001
        st.error(f"Could not load queries: {e}")
        st.stop()
    idx = e2.number_input(f"Query number (0 - {len(qs) - 1})", 0, len(qs) - 1, 0)
    query = qs[int(idx)]
    st.markdown(f"> {query.query_text}")
    m = st.columns(5)
    m[0].metric("Route", f"{query.org} → {query.dest}")
    m[1].metric("Days", query.days)
    m[2].metric("Travellers", query.people_number)
    m[3].metric("Budget", money(query.budget))
    m[4].metric("Level", getattr(query, "_difficulty", "n/a"))
    st.caption(f"Constraints: {query.local_constraint or 'none'}")

problems = []
if mode == "My own trip":
    if trip == "One city" and origin == dest:
        problems.append("Origin and destination are the same city.")
    if trip != "One city" and chosen and origin in chosen:
        problems.append(f"The origin ({origin}) cannot also be one of the cities to visit.")
for pmsg in problems:
    st.error(pmsg)
go = st.button("🚀 Plan my trip", type="primary", disabled=bool(problems))

if go:
    try:
        llm = load_llm(backend, model)
        retriever = None
        if use_rag:
            if not data_dir:
                st.warning("RAG needs the queries folder; continuing without it.")
            else:
                retriever = load_retriever(db_path, data_dir, backend, model)
        with st.spinner("Agents are negotiating ..."):
            st.session_state["res"] = run_pipeline(query, llm, solver, max_rounds, restarts, retriever)
            st.session_state["q"] = query
    except Exception as e:                                                  # noqa: BLE001
        st.error(f"{type(e).__name__}: {e}")
        st.stop()

res = st.session_state.get("res")
if not res:
    st.stop()
q = st.session_state["q"]

# --------------------------------------------------------------------------- #
# results
# --------------------------------------------------------------------------- #

st.divider()
ok = res["converged"] and res["score"]["final_pass"]
(st.success if ok else st.warning)(
    f"{'✅ Valid itinerary found' if ok else '⚠️ No fully valid itinerary'} — total {money(res['total'])} "
    f"of {money(res['budget'])} budget "
    + (f"| negotiation rounds {res['rounds']}, restarts {res['restarts_used']}"
       if res["solver"] == "negotiate" else f"| ABT messages {res['rounds']}"))

t0, t1, t2, t3, t4 = st.tabs(["🗺️ Architecture walk-through", "1️⃣ Prompt & LLM planner", "2️⃣ Agents propose",
                              "3️⃣ Negotiation", "4️⃣ Final itinerary"])

rsolver = res["solver"]
ev = res["events"]
flags = [e for e in ev if e["t"] == "flag"]
bp = [e for e in ev if e["t"] == "backprompt"]
used_rag = bool(res["llm_prompt"] and res["llm_prompt"]["parts"].get("rag_case"))

with t0:
    rows = [
        ("(1) User query / Problem specification", "done", f"{q.org} → {q.dest}, {q.days} days, {q.people_number} people"),
        ("(2) Prompt Generator", "done", "instruction + database context + output format + few-shot"
         + (" + RAG case" if used_rag else " (RAG case off)")),
        ("LLM Planner", "done" if res["llm_name"] != "MockLLM" else "rule-based",
         f"{res['llm_name']}" + (" (cached answer)" if res["llm_cached"] else "")),
        ("(3) LLM response → Reformatter → Plan Blackboard", "done", "skeleton: " + " | ".join(res["skeleton"])),
        ("(4) Constraint evaluation by the agents", "done" if rsolver == "negotiate" else "ABT",
         f"{len([e for e in ev if e['t'] == 'round'])} evaluation round(s), {len(flags)} violation(s) raised"
         if rsolver == "negotiate" else f"{res['rounds']} ABT messages"),
        ("(5) Negotiation / disagreement → Coordinator", "done" if flags or rsolver == "abt" else "not needed",
         "agents conceded: " + str(len([e for e in ev if e["t"] == "concede"])) if rsolver == "negotiate"
         else f"{res['abt']['nogoods_generated']} nogoods"),
        ("(6) Back prompt to the LLM", "used" if bp else "not needed",
         "only when every restart failed" if not bp else "LLM asked to re-plan"),
        ("(7) Valid solution", "yes" if ok else "no", f"{res['score']['hard'][0]}/{res['score']['hard'][1]} hard "
         f"and {res['score']['commonsense'][0]}/{res['score']['commonsense'][1]} commonsense checks pass"),
    ]
    st.dataframe(pd.DataFrame(rows, columns=["Step in the architecture diagram", "Status", "In this run"]),
                 hide_index=True)

with t1:
    st.subheader("Prompt Generator → LLM planner → Reformatter")
    p = res["llm_prompt"]
    if not res["prompt_sent"]:
        st.info("MockLLM is selected, so nothing was sent. This is the prompt that would be sent to Ollama:")
    for name, key in (("1 Instruction prompt", "instruction"), ("2 Context (from the databases)", "context"),
                      ("3 Output format", "output_format"), ("4 Few-shot example", "few_shot"),
                      ("5 RAG-retrieved similar case", "rag_case")):
        with st.expander(name, expanded=key in ("context", "rag_case")):
            st.text(p["parts"].get(key) or "(off — enable RAG in the sidebar)")
    st.markdown("**LLM response (raw)**")
    st.code(res["llm_raw"] or "(none: rule-based planner or cached)", language="json", wrap_lines=True)
    st.markdown("**Reformatter → plan blackboard (one entry per day)**")
    st.dataframe(pd.DataFrame({"Day": range(1, len(res["skeleton"]) + 1), "City / leg": res["skeleton"]}),
                 hide_index=True)

with t2:
    st.subheader("Each agent's first proposal (round 0)")
    if rsolver == "abt":
        st.info("ABT agents pick whole bundles; see the message log in tab 3.")
    first = {e["agent"]: e for e in ev if e["t"] == "proposal" and e["attempt"] == 0}
    for name in AGENT_ORDER:
        e = first.get(name)
        with st.expander(AGENT_ICON[name], expanded=name != "attraction"):
            if name == "budget":
                st.write("Owns no variables. It adds up the plan and, when the total exceeds the budget, "
                         "names the most expensive item and asks the agent that owns it to concede.")
            elif e and e["slots"]:
                df = pd.DataFrame([{"Day": s["day"], "Slot": KIND_NAME[s["kind"]], "Choice": s["label"],
                                    f"Cost ({cur})": round(s["cost"] * rate)} for s in e["slots"]])
                st.dataframe(df, hide_index=True)
            else:
                st.caption("(no proposal recorded)")

with t3:
    if rsolver == "abt":
        a = res["abt"]
        st.subheader("Asynchronous Backtracking")
        c = st.columns(5)
        c[0].metric("Messages", a["messages"]); c[1].metric("ok?", a["ok"]); c[2].metric("nogood", a["nogood"])
        c[3].metric("Nogoods generated", a["nogoods_generated"]); c[4].metric("Value changes", a["value_changes"])
        st.caption(f"Priority order: transport > hotel > food > attraction. Candidate bundles per agent: "
                   f"{a['domain_sizes']}. Budget is checked by the lowest-priority agent in its scope (food).")
        st.text_area("Message log (first 400)", "\n".join(a["trace"]), height=360)
        if not res["converged"]:
            st.warning(f"ABT result: {res['reason']}")
    else:
        st.subheader("Negotiation: constraint evaluation → disagreement → concessions")
        views = res["rounds_view"]
        attempts = sorted({v["attempt"] for v in views})
        att = st.selectbox("Attempt (restart)", attempts, index=len(attempts) - 1,
                           format_func=lambda a: f"attempt {a}" + ("  (last)" if a == attempts[-1] else "")) \
            if len(attempts) > 1 else (attempts[0] if attempts else 0)
        vs = [v for v in views if v["attempt"] == att]
        if vs:
            st.line_chart(pd.DataFrame({f"plan total ({cur})": [v["total"] * rate for v in vs],
                                        f"budget ({cur})": [v["budget"] * rate for v in vs]},
                                       index=[v["round"] for v in vs]))
        for v in vs:
            label = (f"Round {v['round']} — total {money(v['total'])} / budget {money(v['budget'])} — "
                     + ("✅ all agents agree" if v["agreed"] else f"❌ {len(v['flags'])} violation(s)"))
            with st.expander(label, expanded=not v["agreed"] and v["round"] <= 3):
                for f in v["flags"]:
                    st.markdown(f"**{AGENT_ICON.get(f['agent'], f['agent'])}** flags `{f['critic']}` "
                                f"({f['severity']}): {f['message']}  \n→ asks **{AGENT_ICON.get(f['asks'], f['asks'])}** "
                                f"to concede")
                if v["flags"]:
                    st.markdown(f"🧑‍⚖️ **Coordinator**: consolidating concessions from "
                                f"{', '.join(AGENT_ICON.get(a, a) for a in v['asked'])}")
                if v["concessions"]:
                    st.dataframe(pd.DataFrame([{
                        "Agent": AGENT_ICON.get(c["agent"], c["agent"]), "Day": c["day"], "Slot": KIND_NAME[c["kind"]],
                        "Was": c["old"]["label"], "Now": c["new"]["label"],
                        f"Saved ({cur})": round((c["old"]["cost"] - c["new"]["cost"]) * rate)} for c in v["concessions"]]),
                        hide_index=True)
                for s in (v["stuck"] if not v["concessions"] else []):
                    st.error(f"{AGENT_ICON.get(s['agent'], s['agent'])} has no other option left for this violation")
        for e in bp:
            st.warning("(6) Back prompt sent to the LLM planner:\n\n" + e["text"])
        if res["escalated"]:
            st.info("The LLM's revised skeleton led to a solution.")

with t4:
    st.subheader("Final itinerary")
    c = st.columns(4)
    left_over = res["budget"] - res["total"]
    c[0].metric("Total cost", money(res["total"]),
                delta=(f"{money(left_over)} under budget" if left_over >= 0 else f"-{money(-left_over)} over budget"))
    c[1].metric("Hard checks", f"{res['score']['hard'][0]}/{res['score']['hard'][1]}")
    c[2].metric("Commonsense", f"{res['score']['commonsense'][0]}/{res['score']['commonsense'][1]}")
    c[3].metric("Verdict", "PASS" if res["score"]["final_pass"] else "FAIL")
    left, right = st.columns([2, 1])
    with left:
        for d in res["days"]:
            with st.container(border=True):
                st.markdown(f"**Day {d['day']} — {d['city']}**")
                for it in d["items"]:
                    st.markdown(f"- **{KIND_NAME[it['kind']]}**: {it['label']}"
                                + (f" — {money(it['cost'])}" if it["cost"] else "")
                                + (f"  \n  <span style='color:gray'>{it['detail']}</span>" if it["detail"] else ""),
                                unsafe_allow_html=True)
    with right:
        st.markdown("**Where the money goes**")
        st.bar_chart(pd.DataFrame({cur: {k: v * rate for k, v in res["breakdown"].items()}}))
        if res["remaining"]:
            st.markdown("**Still violated**")
            for r in res["remaining"]:
                st.caption(r)
        st.download_button("⬇️ Download itinerary (JSON)", res["plan_json"], "itinerary.json", "application/json")