"""
Asynchronous Backtracking (ABT) -- Yokoo, Durfee, Ishida & Kuwabara (1992/1998).

This is a real ABT implementation for the travel-planning DCSP, kept separate from
`dcsp_solver.py` (the round-based negotiation heuristic) so the two can be compared.

MAPPING TO THE PROJECT
----------------------
* ABT agents (one compound variable each, as in multi-variable ABT):
      transport   value = a bundle of transport choices for ALL legs
      hotel       value = a bundle of hotel choices for ALL nights
      food        value = a bundle of breakfast/lunch/dinner choices for ALL days
      attraction  value = a bundle of attraction choices for ALL days
* Fixed PRIORITY ORDER (high -> low):  transport > hotel > food > attraction
  (same ordering as the agent priorities: the agent that is easiest to move is lowest).
* DOMAIN of an agent = candidate bundles from the sandbox, built in R cost tiers
  (cheapest ... priciest). Unary / agent-internal constraints (room type, room rule,
  minimum nights, transport preference, no flight+self-driving mix, requested cuisines,
  no repeated restaurant/attraction) are enforced when building the domain, by running
  the existing agents' own `check()` on every bundle.
* VALUE ORDER = nearest to the median bundle first (a reasonable first guess), then outward.
* The only inter-agent constraint is the BUDGET: cost(transport) + cost(hotel) + cost(food)
  <= budget. As in ABT, a constraint is evaluated by the LOWEST-priority agent in its scope
  (food), which therefore needs the values of transport and hotel (links).

THE ALGORITHM (per agent)
-------------------------
  ok?(sender, value)   -> update agent_view, check_agent_view()
  nogood(sender, N)    -> add links to unknown agents in N; if N is compatible with the
                          view, store it and check_agent_view() (re-send ok? to the sender if
                          my value did not change); otherwise reply ok?
  add_link(requester)  -> requester joins my out-links; send it my current value
  check_agent_view()   -> if my value violates a constraint or a compatible nogood, pick
                          the next consistent value and send ok? to my lower-priority
                          neighbours; if none exists -> BACKTRACK
  backtrack()          -> resolvent = union of the reasons that eliminated every value.
                          Empty resolvent => NO SOLUTION (proved infeasible).
                          Otherwise send nogood(resolvent) to the lowest-priority agent
                          in it and drop that agent from my view.

Messages travel over per-pair FIFO channels; the network delivers from a random non-empty
channel each step (asynchrony, reproducible with `seed`). The run ends when no message is
in flight (quiescence => every agent is consistent => a solution) or a nogood set is empty.
"""

from __future__ import annotations
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from data_model import PlanState, transport_cost, lodging_cost, meal_cost, cuisines_of
from data import sandbox
from agents.concrete import TransportAgent, HotelAgent, FoodAgent, AttractionAgent

PRIORITY_ORDER = ["transport", "hotel", "food", "attraction"]   # high -> low
RANK = {n: i for i, n in enumerate(PRIORITY_ORDER)}


# --------------------------------------------------------------------------- #
# Bundles (the values of an ABT agent)
# --------------------------------------------------------------------------- #

@dataclass
class Bundle:
    agent: str
    assign: dict          # (day, kind) -> chosen item (dict, or list of dicts for attractions)
    cost: float
    ident: tuple

    def label(self, idx: int) -> str:
        return f"#{idx}" + (f"(cost {self.cost:,.0f})" if self.cost else "")


def _ident(v):
    if isinstance(v, list):
        return tuple(_ident(x) for x in v)
    if isinstance(v, dict):
        return (v.get("Flight Number") or v.get("NAME") or v.get("Name") or f"{v.get('mode')}-{v.get('cost')}",)
    return (str(v),)


def _apply(plan: PlanState, assign: dict) -> None:
    for (day, kind), val in assign.items():
        setattr(plan.days_plan[day], kind, val)


def _make_bundle(base: PlanState, agent: str, assign: dict) -> Bundle:
    tmp = base.clone()
    _apply(tmp, assign)
    ident = tuple(sorted((k, _ident(v)) for k, v in assign.items()))
    return Bundle(agent, assign, tmp.total_cost(), ident), tmp


def _quantile_idx(t: int, tiers: int, n: int) -> int:
    q = t / (tiers - 1) if tiers > 1 else 0.5
    return int(round(q * (n - 1)))


def _finish_domain(base: PlanState, agent: str, raw: list, checker, strict: bool = True) -> list:
    """Dedupe, drop bundles violating the agent's OWN constraints (when strict), order median-first.
    strict=False keeps violating bundles: used only as a fallback so a plan can still be assembled."""
    seen, ok = set(), []
    for assign in raw:
        b, tmp = _make_bundle(base, agent, assign)
        if b.ident in seen:
            continue
        seen.add(b.ident)
        if strict and any(v.severity in ("hard", "commonsense") for v in checker(tmp)):
            continue
        ok.append(b)
    ok.sort(key=lambda b: b.cost)
    m = len(ok) // 2
    order = sorted(range(len(ok)), key=lambda i: (abs(i - m), i))      # median first, then outward
    return [ok[i] for i in order]


# ---- domain builders --------------------------------------------------------

def transport_domain(base: PlanState, tiers: int, strict: bool = True) -> list:
    ag = TransportAgent()
    legs = [d for d, dp in base.days_plan.items() if " to " in dp.current_city]
    raw = []
    for forbid in ("self-driving", "flight"):            # a trip never mixes the two
        pools = {d: ag._leg_pool(base, d, forbid) for d in legs}
        if any(not p for p in pools.values()):
            continue
        for t in range(tiers):
            raw.append({(d, "transportation"): pools[d][_quantile_idx(t, tiers, len(pools[d]))] for d in legs})
    return _finish_domain(base, "transport", raw, ag.check, strict)


def hotel_domain(base: PlanState, tiers: int, strict: bool = True) -> list:
    ag = HotelAgent()
    last = max(base.days_plan)
    nights = {}
    for d, dp in base.days_plan.items():
        if d == last:
            continue
        city = dp.current_city.split(" to ")[-1].strip() if " to " in dp.current_city else dp.current_city
        nights.setdefault(city, []).append(d)
    pools = {c: ag._candidates(base, c) for c in nights}
    if any(not p for p in pools.values()):
        return []
    raw = []
    for t in range(tiers):
        assign = {}
        for c, days in nights.items():
            pick = pools[c][_quantile_idx(t, tiers, len(pools[c]))]
            for d in days:
                assign[(d, "accommodation")] = pick
        raw.append(assign)
    return _finish_domain(base, "hotel", raw, ag.check, strict)


def food_domain(base: PlanState, tiers: int, strict: bool = True) -> list:
    ag = FoodAgent()
    days = sorted(base.days_plan)
    wanted = list(base.query.cuisine or [])
    coverage = {(days[i % len(days)], "dinner"): cu for i, cu in enumerate(wanted)}
    pools = {}
    for d in days:
        city = ag._city(base.days_plan[d])
        pools.setdefault(city, ag._all(city))
    slots = [(d, m) for d in days for m in ag.MEALS]
    slots.sort(key=lambda s: 0 if s in coverage else 1)      # cuisine-coverage slots first
    raw = []
    for t in range(tiers):
        used, assign = set(), {}
        for (d, m) in slots:
            pool = pools[ag._city(base.days_plan[d])]
            if not pool:
                continue
            cu = coverage.get((d, m))
            cands = [c for c in pool if c["Name"] not in used and (cu is None or cu in cuisines_of(c))]
            if not cands:
                cands = [c for c in pool if c["Name"] not in used] or pool
            pick = cands[_quantile_idx(t, tiers, len(cands))]
            assign[(d, m)] = pick
            used.add(pick["Name"])
        raw.append(assign)
    return _finish_domain(base, "food", raw, ag.check, strict)


def attraction_domain(base: PlanState, tiers: int, strict: bool = True) -> list:
    ag = AttractionAgent()
    tmp = base.clone()
    ag.propose(tmp, sandbox)
    assign = {(d, "attraction"): dp.attraction for d, dp in tmp.days_plan.items() if dp.attraction}
    return _finish_domain(base, "attraction", [assign], ag.check, strict)


# --------------------------------------------------------------------------- #
# Constraints
# --------------------------------------------------------------------------- #

@dataclass
class Constraint:
    name: str
    scope: list            # agent names (any order)
    holds: callable        # fn({agent: Bundle}) -> bool

    @property
    def owner(self) -> str:
        return max(self.scope, key=lambda a: RANK[a])     # lowest-priority agent in scope


# --------------------------------------------------------------------------- #
# ABT agent
# --------------------------------------------------------------------------- #

class ABTAgent:
    def __init__(self, name: str, domain: list, net: "Network"):
        self.name = name
        self.domain = domain                   # list[Bundle] in preference order; vid = index
        self.net = net
        self.value: Optional[int] = None
        self.view: dict = {}                   # higher-priority agent -> vid
        self.out_links: set = set()            # lower-priority agents to inform
        self.nogoods: list = []                # frozenset of (agent, vid); contains (self, d)
        self.constraints: list = []            # constraints this agent evaluates
        self.value_changes = 0

    # ---- reasons a value is ruled out -------------------------------------- #
    def reason_for(self, d: int):
        """None if value d is consistent with the view, otherwise the frozenset of
        higher-priority (agent, vid) pairs that rule it out (may be empty)."""
        for c in self.constraints:
            others = [a for a in c.scope if a != self.name]
            if all(a in self.view for a in others):
                vals = {a: self.net.agents[a].domain[self.view[a]] for a in others}
                vals[self.name] = self.domain[d]
                if not c.holds(vals):
                    return frozenset((a, self.view[a]) for a in others)
        for N in self.nogoods:
            if (self.name, d) in N:
                rest = [(a, v) for (a, v) in N if a != self.name]
                if all(self.view.get(a) == v for a, v in rest):
                    return frozenset(rest)
        return None

    # ---- message handlers ---------------------------------------------------- #
    def start(self):
        self.check_agent_view()

    def on_ok(self, sender: str, vid: int):
        self.view[sender] = vid
        self.check_agent_view()

    def on_add_link(self, requester: str):
        self.out_links.add(requester)
        self.net.send(self.name, requester, "ok", self.value)

    def on_nogood(self, sender: str, N: frozenset):
        for (a, v) in N:
            if a != self.name and a not in self.view:
                self.view[a] = v
                self.net.send(self.name, a, "add_link", None)
        compatible = ((self.name, self.value) in N and
                      all(self.view.get(a) == v for (a, v) in N if a != self.name))
        if compatible:
            if N not in self.nogoods:
                self.nogoods.append(N)
            old_value = self.value
            self.check_agent_view()
            if self.value == old_value:
                # Yokoo's rule: the sender dropped me from its view when it sent the nogood;
                # if my value did not change it must be told again.
                self.net.send(self.name, sender, "ok", self.value)
        else:
            self.net.send(self.name, sender, "ok", self.value)     # sender's view of me is stale

    # ---- core ---------------------------------------------------------------- #
    def check_agent_view(self):
        while not self.net.stopped:
            if self.value is not None and self.reason_for(self.value) is None:
                return
            new = next((d for d in range(len(self.domain)) if self.reason_for(d) is None), None)
            if new is not None:
                self.value = new
                self.value_changes += 1
                for a in self.out_links:
                    self.net.send(self.name, a, "ok", self.value)
                return
            self.backtrack()

    def backtrack(self):
        resolvent = frozenset().union(*[self.reason_for(d) for d in range(len(self.domain))]) \
            if self.domain else frozenset()
        if not resolvent:
            self.net.no_solution(self.name)
            return
        j = max((p[0] for p in resolvent), key=lambda a: RANK[a])        # lowest-priority in nogood
        self.net.nogoods_generated += 1
        self.net.send(self.name, j, "nogood", resolvent)
        self.view.pop(j, None)


# --------------------------------------------------------------------------- #
# Network + solver
# --------------------------------------------------------------------------- #

class Network:
    def __init__(self, seed: int):
        self.rng = random.Random(seed)
        self.agents: dict = {}
        self.channels: dict = {}              # (from, to) -> deque of messages (FIFO per pair)
        self.stopped = False
        self.proved_infeasible_by: Optional[str] = None
        self.counts = {"ok": 0, "nogood": 0, "add_link": 0}
        self.nogoods_generated = 0
        self.trace: list = []

    def send(self, frm: str, to: str, kind: str, payload):
        self.channels.setdefault((frm, to), deque()).append((kind, payload))
        self.counts[kind] += 1

    def no_solution(self, by: str):
        self.stopped = True
        self.proved_infeasible_by = by

    def _fmt(self, kind, payload, frm):
        if kind == "ok":
            v = self.agents[frm].domain[payload] if payload is not None else None
            return f"ok?({frm}=#{payload}" + (f", cost {v.cost:,.0f})" if v is not None and v.cost else ")")
        if kind == "nogood":
            return "nogood{" + ", ".join(f"{a}=#{v}" for a, v in sorted(payload, key=lambda p: RANK[p[0]])) + "}"
        return "add_link"

    def run(self, max_messages: int) -> str:
        """Returns 'solved' | 'no_solution' | 'max_messages'."""
        delivered = 0
        while not self.stopped:
            live = [k for k, q in self.channels.items() if q]
            if not live:
                return "solved"                      # quiescence
            if delivered >= max_messages:
                return "max_messages"
            key = self.rng.choice(sorted(live))     # asynchrony: any channel may deliver next
            kind, payload = self.channels[key].popleft()
            frm, to = key
            delivered += 1
            if len(self.trace) < 3000:
                self.trace.append(f"{frm} -> {to}: {self._fmt(kind, payload, frm)}")
            ag = self.agents[to]
            if kind == "ok":
                ag.on_ok(frm, payload)
            elif kind == "nogood":
                ag.on_nogood(frm, payload)
            else:
                ag.on_add_link(frm)
        return "no_solution"


@dataclass
class ABTResult:
    plan: PlanState
    converged: bool
    messages: int = 0
    ok_msgs: int = 0
    nogood_msgs: int = 0
    link_msgs: int = 0
    nogoods_generated: int = 0
    value_changes: int = 0
    failure_reason: Optional[str] = None
    trace: list = field(default_factory=list)
    domain_sizes: dict = field(default_factory=dict)


def solve_abt(base: PlanState, tiers: int = 12, seed: int = 0, max_messages: int = 100000) -> ABTResult:
    """Run ABT on the skeleton plan `base` (cities per day, nothing assigned yet)."""
    q = base.query
    builders = {"transport": transport_domain, "hotel": hotel_domain,
                "food": food_domain, "attraction": attraction_domain}
    domains = {k: f(base, tiers) for k, f in builders.items()}
    sizes = {k: len(v) for k, v in domains.items()}
    plan = base.clone()
    # An agent with NO option that satisfies its own constraints (e.g. no hotel of the wanted
    # room type in that city) makes the problem infeasible. Report that, but still assemble a
    # complete plan from relaxed bundles so the result can be scored like every other system.
    empty = [k for k, v in domains.items() if not v]
    for k in empty:
        domains[k] = builders[k](base, tiers, strict=False)
    if any(not v for v in domains.values()):
        return ABTResult(plan, False, failure_reason=f"empty_domain:{empty[0]}", domain_sizes=sizes)

    net = Network(seed)
    for name in PRIORITY_ORDER:
        net.agents[name] = ABTAgent(name, domains[name], net)

    budget = Constraint(
        "budget", ["transport", "hotel", "food"],
        lambda vals: sum(b.cost for b in vals.values()) <= q.budget + 1e-9)
    net.agents[budget.owner].constraints.append(budget)

    # initial links: every constraint-scope agent informs the lower-priority ones
    for a in budget.scope:
        for b in budget.scope:
            if RANK[a] < RANK[b]:
                net.agents[a].out_links.add(b)
                net.agents[b].view.setdefault(a, None)
    for ag in net.agents.values():                       # no value known yet -> not in view
        ag.view = {k: v for k, v in ag.view.items() if v is not None}

    for name in PRIORITY_ORDER:                          # every agent picks its first value
        net.agents[name].start()
    status = net.run(max_messages)

    res = ABTResult(plan, status == "solved" and not empty, messages=sum(net.counts.values()),
                    ok_msgs=net.counts["ok"], nogood_msgs=net.counts["nogood"],
                    link_msgs=net.counts["add_link"], nogoods_generated=net.nogoods_generated,
                    value_changes=sum(a.value_changes for a in net.agents.values()),
                    trace=net.trace, domain_sizes=sizes)
    if empty:
        res.failure_reason = f"empty_domain:{empty[0]}"          # own constraint unsatisfiable
    elif status == "no_solution":
        res.failure_reason = "proved_infeasible"        # empty nogood: no combination fits
    elif status == "max_messages":
        res.failure_reason = "max_messages"
    # assemble the plan from the agents' current values (best effort even on failure)
    for name in PRIORITY_ORDER:
        ag = net.agents[name]
        if ag.value is not None:
            _apply(plan, ag.domain[ag.value].assign)
    return res
