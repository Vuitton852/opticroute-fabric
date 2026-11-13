"""
tests/benchmark.py

Physics & performance benchmark: blind ECMP vs. the physics-aware OpticalRoutingEngine.

Both routers receive identical flows in identical order on the same FatTreeTopology and
are judged by one shared model, so the comparison is apples to apples.

Blind ECMP
    Pins every flow to a random shortest switch-only path. It never looks at booked
    wavelengths or switch temperature. It books wavelengths first-fit; when a flow needs
    more channels than are free it overlaps channels that are already in use (a WDM
    collision), and a copper link pushed past its bandwidth is a collision too. The
    flow is still sent, and still heats every switch on its path.

OpticalRoutingEngine
    Routes only over paths with free wavelengths, copper headroom and power budget, and
    steers around hot switches. A flow with no feasible path is blocked and sends nothing.

Failed flows
    ECMP: flows that collided when booked. Engine: flows that were blocked. Counted at
    booking time, so the earlier flows that later share a collided link are not counted
    as failed; the harm to them shows up in latency instead.

Shared scoring (computed after all flows are placed, from the final fabric state)
    link latency = base + utilization x serialization penalty
                   + an exponential penalty once utilization exceeds 100%
    thermal      = (switch temperature - ambient) x thermal penalty, summed over the
                   switches on the path
    Average latency is weighted by traffic volume and covers delivered flows: every flow
    for ECMP (collided ones included), routed flows only for the engine.

Both routers use the engine's own heating, thermal-penalty and serialization settings.
Three workloads are swept over 10-50 GB tensors: Ring All-Reduce, Cross-Pod Shift and a
Mixture-of-Experts (MoE) All-to-All token dispatch with skewed expert popularity.

Unavoidable failures
    GPU access links have a fixed capacity, so some flows cannot be carried by any router
    (a flow bigger than the link, or MoE traffic converging on a hot expert). Each row's
    floor is a lower bound on failed flows for any router, computed from those limits.
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import io
import math
import random
import statistics
import sys
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import networkx as nx

# Allow running this file directly (python tests/benchmark.py) while still
# importing the sibling `simulation`, `workloads` and `core` packages.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.routing_engine import AMBIENT_TEMP_C, OpticalRoutingEngine  # noqa: E402
from simulation.topology import FatTreeTopology  # noqa: E402
from workloads.traffic_generator import AIWorkloadGenerator, Flow, FlowPriority  # noqa: E402

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
TENSOR_SIZES_GB: tuple[float, ...] = (10.0, 20.0, 30.0, 40.0, 50.0)
THROTTLE_TEMP_C = 80.0  # a switch at or above this is thermally throttling
OVERLOAD_STEEPNESS = 3.0  # exponential growth rate of the overload latency penalty
MAX_OVERLOAD_EXCESS = 10.0  # cap on (utilization - 1) inside expm1(), avoids overflow
EPSILON = 1e-9
BITS_PER_BYTE = 8
SWITCH_TYPES = frozenset({"core", "aggregation", "edge"})
MOE_TOP_K = 2  # experts each GPU dispatches to in the MoE workload
MOE_SEED = 42  # seeds MoE expert popularity, expert selection and gate weights

Edge = tuple[str, str]


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class PhysicsModel:
    """Physical constants shared by both routers, read from the engine itself."""

    serialization_penalty_ns: float
    thermal_penalty_ns_per_c: float
    heating_c_per_gbps: float
    sync_window_s: float

    @classmethod
    def from_engine(cls, engine: OpticalRoutingEngine) -> PhysicsModel:
        return cls(
            serialization_penalty_ns=engine.serialization_penalty_ns,
            thermal_penalty_ns_per_c=engine.thermal_penalty_ns_per_c,
            heating_c_per_gbps=engine.heating_c_per_gbps,
            sync_window_s=engine.sync_window_s,
        )


@dataclass(frozen=True, slots=True)
class PlacedFlow:
    """A delivered flow pinned to a concrete path."""

    path: tuple[str, ...]
    volume_gbits: float
    demand_gbps: float


@dataclass(slots=True)
class RunResult:
    """Final fabric state after one router has handled every flow."""

    placed: list[PlacedFlow]
    load: dict[Edge, float]  # booked Gbps per directed edge
    temps: dict[str, float]  # final temperature per switch
    failed_flows: int  # ECMP: collided. Engine: blocked.


@dataclass(frozen=True, slots=True)
class Metrics:
    """Results for one routing run (or the mean of several)."""

    total_flows: int
    failed_flows: float
    hot_switches: float  # switches at or above THROTTLE_TEMP_C
    peak_temp_c: float
    avg_latency_ns: float  # NaN when no flow was delivered


@dataclass(frozen=True, slots=True)
class SweepRow:
    tensor_size_gb: float
    ecmp: Metrics  # mean across ECMP trials
    engine: Metrics
    unavoidable_failed: int  # lower bound on failed flows for any router (GPU port limits)

    @property
    def port_limited(self) -> bool:
        return self.unavoidable_failed > 0


@dataclass(frozen=True, slots=True)
class WorkloadSpec:
    name: str
    description: str
    builder: Callable[[AIWorkloadGenerator, float], list[Flow]]


# --------------------------------------------------------------------------- #
# Shared physical model
# --------------------------------------------------------------------------- #
def link_latency_ns(base_latency_ns: float, utilization: float, serialization_penalty_ns: float) -> float:
    """Latency of one link: linear in load, then exponential once overloaded."""
    latency = base_latency_ns + utilization * serialization_penalty_ns
    if utilization > 1.0 + EPSILON:
        excess = min(utilization - 1.0, MAX_OVERLOAD_EXCESS)
        latency += serialization_penalty_ns * math.expm1(OVERLOAD_STEEPNESS * excess)
    return latency


def evaluate(graph: nx.DiGraph, run: RunResult, physics: PhysicsModel, total_flows: int) -> Metrics:
    """Score a finished run. Reads only static link attributes and the run's final state."""
    weighted_latency = 0.0
    total_volume = 0.0
    for flow in run.placed:
        latency = 0.0
        for edge in zip(flow.path, flow.path[1:]):
            attrs = graph.edges[edge]
            utilization = run.load.get(edge, 0.0) / attrs["bandwidth_gbps"]
            latency += link_latency_ns(
                attrs["base_latency_ns"], utilization, physics.serialization_penalty_ns
            )
        for switch in flow.path[1:-1]:
            degrees_above_ambient = max(0.0, run.temps[switch] - AMBIENT_TEMP_C)
            latency += degrees_above_ambient * physics.thermal_penalty_ns_per_c
        weighted_latency += latency * flow.volume_gbits
        total_volume += flow.volume_gbits

    return Metrics(
        total_flows=total_flows,
        failed_flows=float(run.failed_flows),
        hot_switches=float(sum(1 for t in run.temps.values() if t >= THROTTLE_TEMP_C - EPSILON)),
        peak_temp_c=max(run.temps.values(), default=AMBIENT_TEMP_C),
        avg_latency_ns=weighted_latency / total_volume if total_volume else math.nan,
    )


def mean_metrics(runs: Sequence[Metrics]) -> Metrics:
    latencies = [m.avg_latency_ns for m in runs if not math.isnan(m.avg_latency_ns)]
    return Metrics(
        total_flows=runs[0].total_flows,
        failed_flows=statistics.fmean(m.failed_flows for m in runs),
        hot_switches=statistics.fmean(m.hot_switches for m in runs),
        peak_temp_c=statistics.fmean(m.peak_temp_c for m in runs),
        avg_latency_ns=statistics.fmean(latencies) if latencies else math.nan,
    )


# --------------------------------------------------------------------------- #
# Routers under test
# --------------------------------------------------------------------------- #
class ECMPRouter:
    """
    Blind ECMP with physical consequences. It works on its own private state, so it
    never touches the topology graph that the engine books onto.
    """

    def __init__(self, graph: nx.DiGraph, physics: PhysicsModel) -> None:
        self.graph = graph
        self.physics = physics
        self._switches = [n for n, a in graph.nodes(data=True) if a.get("type") in SWITCH_TYPES]
        self._path_cache: dict[Edge, list[tuple[str, ...]]] = {}

    def _equal_cost_paths(self, source: str, destination: str) -> list[tuple[str, ...]]:
        key = (source, destination)
        cached = self._path_cache.get(key)
        if cached is None:
            cached = [
                tuple(path)
                for path in nx.all_shortest_paths(self.graph, source, destination)
                if all(self.graph.nodes[node].get("type") != "gpu" for node in path[1:-1])
            ]
            if not cached:
                raise ValueError(f"No switch-only path from {source!r} to {destination!r}")
            self._path_cache[key] = cached
        return cached

    def route(self, flows: Sequence[Flow], rng: random.Random) -> RunResult:
        graph, physics = self.graph, self.physics
        booked: dict[Edge, set[int]] = defaultdict(set)
        load: dict[Edge, float] = defaultdict(float)
        temps = {switch: AMBIENT_TEMP_C for switch in self._switches}
        placed: list[PlacedFlow] = []
        collided_flows = 0

        for flow in flows:
            demand = flow.traffic_volume_gbits / physics.sync_window_s
            path = rng.choice(self._equal_cost_paths(flow.source_gpu, flow.destination_gpu))
            collided = False

            for edge in zip(path, path[1:]):
                attrs = graph.edges[edge]
                if attrs["media_type"] == "optical":
                    needed = OpticalRoutingEngine._channels_needed(attrs, demand)
                    in_use = booked[edge]
                    free = [c for c in range(attrs["total_wdm_channels"]) if c not in in_use]
                    taken = free[:needed]
                    if needed > len(free):
                        # Blind ECMP sends anyway and lands on wavelengths already in use.
                        collided = True
                        taken += sorted(in_use)[: needed - len(taken)]
                    in_use.update(taken)
                elif load[edge] + demand > attrs["bandwidth_gbps"] + EPSILON:
                    collided = True  # copper link pushed past its bandwidth
                load[edge] += demand

            heat_c = physics.heating_c_per_gbps * demand
            for switch in path[1:-1]:
                temps[switch] = min(graph.nodes[switch]["max_temp_c"], temps[switch] + heat_c)

            collided_flows += collided
            placed.append(PlacedFlow(path, flow.traffic_volume_gbits, demand))

        return RunResult(placed=placed, load=dict(load), temps=temps, failed_flows=collided_flows)


def run_engine(engine: OpticalRoutingEngine, flows: Sequence[Flow]) -> RunResult:
    """Route with the OpticalRoutingEngine and check its own books before reporting."""
    graph = engine.graph
    engine.reset_fabric_load()
    routes = engine.allocate_routes(flows)

    placed: list[PlacedFlow] = []
    load: dict[Edge, float] = defaultdict(float)
    channels_by_edge: dict[Edge, list[int]] = defaultdict(list)
    for index, route in routes.items():
        if not route.routed:
            continue
        placed.append(PlacedFlow(route.path, flows[index].traffic_volume_gbits, route.demand_gbps))
        for hop in route.hops:
            edge = (hop.source, hop.destination)
            load[edge] += route.demand_gbps
            channels_by_edge[edge].extend(hop.channels)

    # Integrity checks: the engine must never double-book a wavelength, and what it
    # reported must match what it actually booked on the graph.
    for edge, channels in channels_by_edge.items():
        if len(channels) != len(set(channels)):
            raise AssertionError(f"Engine double-booked a wavelength on {edge}")
    for u, v, attrs in graph.edges(data=True):
        if not math.isclose(attrs["current_load"], load.get((u, v), 0.0), abs_tol=1e-6):
            raise AssertionError(f"Engine load mismatch on {(u, v)}")
        if attrs["booked_channels"] != set(channels_by_edge.get((u, v), ())):
            raise AssertionError(f"Engine wavelength mismatch on {(u, v)}")

    temps = engine.get_switch_temperatures()  # a fresh copy, safe to keep after reset
    failed = sum(1 for route in routes.values() if not route.routed)
    engine.reset_fabric_load()  # leave the fabric clean for the next run
    return RunResult(placed=placed, load=dict(load), temps=temps, failed_flows=failed)


# --------------------------------------------------------------------------- #
# Workloads
# --------------------------------------------------------------------------- #
def build_ring_all_reduce(generator: AIWorkloadGenerator, tensor_size_gb: float) -> list[Flow]:
    return generator.generate_all_reduce_traffic(tensor_size_gb)


def build_cross_pod_shift(generator: AIWorkloadGenerator, tensor_size_gb: float) -> list[Flow]:
    """Each GPU sends `tensor_size_gb` to its same-index peer in the next pod."""
    by_pod: dict[int, list[str]] = defaultdict(list)
    for gpu in generator.get_gpu_nodes():
        by_pod[generator.pod_of(gpu)].append(gpu)

    pods = sorted(by_pod)
    if len(pods) < 2:
        raise ValueError("Cross-pod shift needs at least 2 pods")

    volume_gbits = tensor_size_gb * BITS_PER_BYTE
    return [
        Flow(source, destination, volume_gbits, FlowPriority.HIGH)
        for index, pod in enumerate(pods)
        for source, destination in zip(by_pod[pod], by_pod[pods[(index + 1) % len(pods)]])
    ]


def build_moe_all_to_all(
    generator: AIWorkloadGenerator,
    tensor_size_gb: float,
    top_k_experts: int = MOE_TOP_K,
    seed: int = MOE_SEED,
) -> list[Flow]:
    """Skewed MoE token dispatch: each GPU sends `tensor_size_gb` to its top-k experts in other pods."""
    return generator.generate_moe_all_to_all_traffic(
        tensor_size_gb, top_k_experts=top_k_experts, seed=seed
    )


def build_workloads(moe_seed: int = MOE_SEED) -> tuple[WorkloadSpec, ...]:
    """The benchmark's workloads. The MoE hot-spot layout depends on `moe_seed`."""
    return (
        WorkloadSpec(
            name="Ring All-Reduce",
            description="16-GPU ring sync; each GPU sends 2(N-1)/N x tensor to its successor",
            builder=build_ring_all_reduce,
        ),
        WorkloadSpec(
            name="Cross-Pod Shift",
            description="every GPU sends one full tensor to its peer in the next pod",
            builder=build_cross_pod_shift,
        ),
        WorkloadSpec(
            name="MoE All-to-All",
            description=(
                f"Top-{MOE_TOP_K} experts, skewed dispatch (seed {moe_seed}): every GPU sends its "
                f"tensor to {MOE_TOP_K} popular expert GPUs in other pods"
            ),
            builder=functools.partial(build_moe_all_to_all, seed=moe_seed),
        ),
    )


WORKLOADS: tuple[WorkloadSpec, ...] = build_workloads()


# --------------------------------------------------------------------------- #
# Benchmark driver
# --------------------------------------------------------------------------- #
def min_unavoidable_failures(graph: nx.DiGraph, flows: Sequence[Flow], sync_window_s: float) -> int:
    """
    Lower bound on the flows that no router can carry.

    A GPU's access link has a fixed capacity, so the flows a GPU sends (or receives) must
    fit inside it. The fewest flows to drop from a set of demands is found by dropping the
    largest first. Flows partition by source and by destination, so the per-GPU counts add
    up; the larger of the two totals is the bound. Both ECMP's collided flows and the
    engine's blocked flows are counted against the same link limits, so neither can fall
    below it.
    """

    def fewest_drops(demands: list[float], capacity_gbps: float) -> int:
        remaining = sum(demands)
        drops = 0
        for demand in sorted(demands, reverse=True):
            if remaining <= capacity_gbps + EPSILON:
                break
            remaining -= demand
            drops += 1
        return drops

    sent: dict[str, list[float]] = defaultdict(list)
    received: dict[str, list[float]] = defaultdict(list)
    for flow in flows:
        demand = flow.traffic_volume_gbits / sync_window_s
        sent[flow.source_gpu].append(demand)
        received[flow.destination_gpu].append(demand)

    egress = sum(
        fewest_drops(demands, max(graph.edges[gpu, n]["bandwidth_gbps"] for n in graph.successors(gpu)))
        for gpu, demands in sent.items()
    )
    ingress = sum(
        fewest_drops(demands, max(graph.edges[p, gpu]["bandwidth_gbps"] for p in graph.predecessors(gpu)))
        for gpu, demands in received.items()
    )
    return max(egress, ingress)


def exceeds_gpu_port(graph: nx.DiGraph, flows: Sequence[Flow], sync_window_s: float) -> bool:
    """True if GPU access-link limits make some flow undeliverable for any router."""
    return min_unavoidable_failures(graph, flows, sync_window_s) > 0


def benchmark_workload(
    spec: WorkloadSpec,
    generator: AIWorkloadGenerator,
    ecmp: ECMPRouter,
    engine: OpticalRoutingEngine,
    physics: PhysicsModel,
    tensor_sizes_gb: Sequence[float],
    trials: int,
    seed: int,
) -> list[SweepRow]:
    graph = engine.graph
    rows: list[SweepRow] = []
    for size_gb in tensor_sizes_gb:
        flows = spec.builder(generator, size_gb)

        ecmp_runs = [
            evaluate(graph, ecmp.route(flows, random.Random(seed + trial)), physics, len(flows))
            for trial in range(trials)
        ]
        engine_metrics = evaluate(graph, run_engine(engine, flows), physics, len(flows))

        rows.append(
            SweepRow(
                tensor_size_gb=size_gb,
                ecmp=mean_metrics(ecmp_runs),
                engine=engine_metrics,
                unavoidable_failed=min_unavoidable_failures(graph, flows, physics.sync_window_s),
            )
        )
    return rows


# --------------------------------------------------------------------------- #
# Dashboard
# --------------------------------------------------------------------------- #
_WIDTH = 118
_SEP = "  |  "


def _fmt_ns(value: float) -> str:
    return "n/a" if math.isnan(value) else f"{value:,.1f}"


def _fmt_gain(ecmp_ns: float, engine_ns: float) -> str:
    if math.isnan(ecmp_ns) or math.isnan(engine_ns) or ecmp_ns <= 0:
        return "n/a"
    return f"{(ecmp_ns - engine_ns) / ecmp_ns * 100.0:+.1f}%"


def _row_notes(row: SweepRow) -> str:
    return ("A" if row.engine.failed_flows > 0 else "") + ("B" if row.port_limited else "")


def _first_size(rows: Sequence[SweepRow], predicate: Callable[[SweepRow], bool]) -> str:
    for row in rows:
        if predicate(row):
            return f"{row.tensor_size_gb:.0f} GB"
    return "not reached"


def print_dashboard(
    results: dict[str, list[SweepRow]],
    k: int,
    gpu_count: int,
    trials: int,
    physics: PhysicsModel,
    workloads: Sequence[WorkloadSpec] = WORKLOADS,
) -> None:
    bar = "=" * _WIDTH
    print(f"\n{bar}")
    print("  ECMP vs. PHYSICS-AWARE OPTICAL ROUTING ENGINE - PHYSICS & PERFORMANCE DASHBOARD".center(_WIDTH))
    print(bar)
    print(
        f"  Fat-Tree k={k} | {gpu_count} GPUs | identical flows and order for both routers | "
        f"ECMP = mean of {trials} seeded trials"
    )
    print(
        f"  Physics: heating {physics.heating_c_per_gbps:g} C/Gbps (cap 85 C) | throttle at >= "
        f"{THROTTLE_TEMP_C:.0f} C | thermal penalty {physics.thermal_penalty_ns_per_c:g} ns/C per switch | "
        f"sync window {physics.sync_window_s:g} s"
    )
    print(
        f"  Latency = base + congestion ({physics.serialization_penalty_ns:g} ns x utilization, "
        f"exponential above 100%) + thermal"
    )

    for spec in workloads:
        rows = results[spec.name]
        flow_count = rows[0].ecmp.total_flows
        print(f"\n{'-' * _WIDTH}")
        print(f"  WORKLOAD: {spec.name} - {spec.description}")
        print("-" * _WIDTH)

        print(
            f"  {'Tensor':>6}{_SEP}{'Blocked/Collided':^17}{_SEP}"
            f"{f'Switches >= {THROTTLE_TEMP_C:.0f} C':^17}{_SEP}{'Peak Temp (C)':^17}{_SEP}"
            f"{'Avg Effective Latency (ns)':^30}{_SEP}{'Notes':<5}"
        )
        pair = f"{'ECMP':>8} {'Engine':>8}"
        print(
            f"  {'(GB)':>6}{_SEP}{pair}{_SEP}{pair}{_SEP}{pair}{_SEP}"
            f"{'ECMP':>10} {'Engine':>10} {'Gain':>8}{_SEP}{'':<5}"
        )
        print(
            f"  {'-' * 6}{_SEP}{'-' * 17}{_SEP}{'-' * 17}{_SEP}{'-' * 17}{_SEP}{'-' * 30}{_SEP}{'-' * 5}"
        )

        for row in rows:
            gain = _fmt_gain(row.ecmp.avg_latency_ns, row.engine.avg_latency_ns)
            if gain != "n/a" and row.engine.failed_flows > 0:
                gain += "*"  # engine latency covers fewer flows than ECMP's
            print(
                f"  {row.tensor_size_gb:>6.0f}{_SEP}"
                f"{row.ecmp.failed_flows:>8.1f} {row.engine.failed_flows:>8.1f}{_SEP}"
                f"{row.ecmp.hot_switches:>8.2f} {row.engine.hot_switches:>8.2f}{_SEP}"
                f"{row.ecmp.peak_temp_c:>8.1f} {row.engine.peak_temp_c:>8.1f}{_SEP}"
                f"{_fmt_ns(row.ecmp.avg_latency_ns):>10} {_fmt_ns(row.engine.avg_latency_ns):>10} "
                f"{gain:>8}{_SEP}"
                f"{_row_notes(row):<5}"
            )

        print(f"\n  Failed-flow counts are out of {flow_count} flows per run.")
        if any(r.port_limited for r in rows):
            floors = " | ".join(f"{r.tensor_size_gb:.0f} GB: {r.unavoidable_failed}" for r in rows)
            print(f"  Unavoidable failures (GPU port limits, any router): {floors}")
        print(
            f"  First failed flow:      ECMP at {_first_size(rows, lambda r: r.ecmp.failed_flows > 0)}"
            f" (in any trial) | Engine at {_first_size(rows, lambda r: r.engine.failed_flows > 0)}"
        )
        print(
            f"  First thermal throttle: ECMP at {_first_size(rows, lambda r: r.ecmp.hot_switches > 0)}"
            f" (in any trial) | Engine at {_first_size(rows, lambda r: r.engine.hot_switches > 0)}"
        )
        if any("A" in _row_notes(r) for r in rows):
            print(
                "  A = the engine blocked some flows. Its temperature and latency cover routed flows only,"
                " and blocked traffic generates no heat."
            )
            print("      A gain marked * compares the engine's routed flows with all of ECMP's flows.")
        if any(r.port_limited for r in rows):
            print(
                "  B = GPU port limits (a flow, or traffic converging on one GPU, exceeds its access link):"
                " some flows can't be carried by any router."
            )
            print("      Compare those rows by failed flows against the unavoidable floor, not by latency.")

    print(f"\n{bar}")
    print("  Notes")
    print("   - Blocked/Collided: ECMP counts flows that collided when booked (needed more wavelengths")
    print("     than were free, or overbooked a copper link). The engine counts flows it blocked.")
    print("   - Heating: both routers use the same model; ECMP heats for every flow it sends,")
    print("     including collided ones. Temperatures are the final state after all flows, and")
    print("     every ECMP figure in the tables is a mean across trials.")
    print("   - Latency: volume-weighted over delivered flows, scored after placement by one shared")
    print("     model. ECMP delivers every flow; the engine delivers the flows it routed.")
    print("   - Unavoidable failures: a lower bound set by GPU access-link capacity. No router's failed-flow")
    print("     count can fall below it, so only the excess over the floor reflects routing quality.")
    print("   - MoE hot-spots depend on the --moe-seed layout; try several seeds before drawing conclusions.")
    print("   - Thermal constants are model parameters, not measured hardware values.")
    print(bar)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark blind ECMP vs. the physics-aware OpticalRoutingEngine."
    )
    parser.add_argument("--k", type=int, default=4, help="Fat-Tree switch port count (even, >= 4)")
    parser.add_argument("--trials", type=int, default=50, help="seeded ECMP trials to average")
    parser.add_argument("--seed", type=int, default=2026, help="base seed for ECMP trials")
    parser.add_argument("--window", type=float, default=1.0, help="sync window in seconds")
    parser.add_argument(
        "--heating",
        type=float,
        default=None,
        help="switch heating in C per Gbps (default: the engine's own default)",
    )
    parser.add_argument(
        "--moe-seed",
        type=int,
        default=MOE_SEED,
        help="seed for the MoE workload's expert popularity and selection",
    )
    args = parser.parse_args(argv)

    if args.k < 4 or args.k % 2:
        parser.error("--k must be an even integer >= 4")
    if args.trials < 1:
        parser.error("--trials must be at least 1")
    if args.window <= 0:
        parser.error("--window must be positive")
    if args.heating is not None and args.heating < 0:
        parser.error("--heating must not be negative")

    with contextlib.redirect_stdout(io.StringIO()):  # silence the topology's build message
        topology = FatTreeTopology(k=args.k)

    engine_kwargs: dict[str, float] = {"sync_window_s": args.window}
    if args.heating is not None:
        engine_kwargs["heating_c_per_gbps"] = args.heating
    engine = OpticalRoutingEngine(topology, **engine_kwargs)

    physics = PhysicsModel.from_engine(engine)
    generator = AIWorkloadGenerator(topology)
    ecmp = ECMPRouter(topology.graph, physics)

    workloads = build_workloads(args.moe_seed)
    results = {
        spec.name: benchmark_workload(
            spec, generator, ecmp, engine, physics, TENSOR_SIZES_GB, args.trials, args.seed
        )
        for spec in workloads
    }

    print_dashboard(
        results,
        k=args.k,
        gpu_count=len(generator.get_gpu_nodes()),
        trials=args.trials,
        physics=physics,
        workloads=workloads,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())