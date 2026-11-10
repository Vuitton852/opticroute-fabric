"""
tests/benchmark.py

Head-to-head benchmark: standard ECMP vs. the congestion-aware OpticalRoutingEngine.

Both routers are given the exact same flows on the same FatTreeTopology and are
scored by the same physical model, so the comparison is apples to apples:

  * Link utilization  = booked demand (Gbps) / bandwidth_gbps.
  * Overloaded link   = a directed link with utilization above 100%.
  * Link latency      = base_latency_ns + utilization * SERIALIZATION_PENALTY_NS,
                        plus an exponential penalty once the link is overloaded.
  * Avg packet latency = per-flow path latency, weighted by traffic volume.

ECMP is modeled as load-blind: each flow is pinned to one randomly chosen shortest
switch-only path, which mimics hash-based path selection and its collisions. Because
a single random draw is noisy, ECMP is run over several seeded trials and the
reported ECMP figures are means across those trials.

Two workloads are swept over increasing tensor sizes:

  1. Ring All-Reduce (from AIWorkloadGenerator). Ring neighbours mostly share an edge
     switch, so there is little path diversity to exploit and the routers will often tie.
  2. Cross-pod shift. Every GPU sends one tensor to its peer in the next pod. This
     is the classic fat-tree stress pattern, where load-blind path choice collides.
"""

from __future__ import annotations

import argparse
import contextlib
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

from core.routing_engine import OpticalRoutingEngine  # noqa: E402
from simulation.topology import FatTreeTopology  # noqa: E402
from workloads.traffic_generator import AIWorkloadGenerator, Flow, FlowPriority  # noqa: E402

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
TENSOR_SIZES_GB: tuple[float, ...] = (10.0, 20.0, 30.0, 40.0, 50.0)
SERIALIZATION_PENALTY_NS = 100.0  # latency added to a link at 100% utilization
OVERLOAD_STEEPNESS = 3.0  # exponential growth rate of the overload penalty
MAX_OVERLOAD_EXCESS = 10.0  # cap on (utilization - 1) inside exp(), avoids overflow
OVERLOAD_EPSILON = 1e-9  # tolerance so exactly-100% links don't count as overloaded
BITS_PER_BYTE = 8


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class PlacedFlow:
    """A flow pinned to a concrete path by one of the routers."""

    path: tuple[str, ...]
    volume_gbits: float
    demand_gbps: float


@dataclass(frozen=True, slots=True)
class Metrics:
    """Fabric-wide results for one routing run (or the mean of several)."""

    total_volume_gbits: float
    overloaded_links: float
    avg_latency_ns: float
    max_utilization: float  # fraction, 1.0 == 100%


@dataclass(frozen=True, slots=True)
class SweepRow:
    tensor_size_gb: float
    ecmp: Metrics  # mean across ECMP trials
    engine: Metrics


@dataclass(frozen=True, slots=True)
class WorkloadSpec:
    name: str
    description: str
    builder: Callable[[AIWorkloadGenerator, float], list[Flow]]


# --------------------------------------------------------------------------- #
# Shared physical model
# --------------------------------------------------------------------------- #
def link_latency_ns(base_latency_ns: float, utilization: float) -> float:
    """Latency of one link: linear in load, then exponential once overloaded."""
    latency = base_latency_ns + utilization * SERIALIZATION_PENALTY_NS
    if utilization > 1.0:
        excess = min(utilization - 1.0, MAX_OVERLOAD_EXCESS)
        latency += SERIALIZATION_PENALTY_NS * math.expm1(OVERLOAD_STEEPNESS * excess)
    return latency


def evaluate(graph: nx.DiGraph, placed: Sequence[PlacedFlow]) -> Metrics:
    """Score a complete set of placed flows. Reads only static link attributes."""
    loads: dict[tuple[str, str], float] = defaultdict(float)
    for flow in placed:
        for edge in zip(flow.path, flow.path[1:]):
            loads[edge] += flow.demand_gbps

    utilization = {
        edge: load / graph.edges[edge]["bandwidth_gbps"] for edge, load in loads.items()
    }
    overloaded = sum(1 for util in utilization.values() if util > 1.0 + OVERLOAD_EPSILON)
    max_util = max(utilization.values(), default=0.0)

    total_volume = sum(flow.volume_gbits for flow in placed)
    weighted_latency = 0.0
    for flow in placed:
        path_latency = sum(
            link_latency_ns(graph.edges[edge]["base_latency_ns"], utilization[edge])
            for edge in zip(flow.path, flow.path[1:])
        )
        weighted_latency += path_latency * flow.volume_gbits

    return Metrics(
        total_volume_gbits=total_volume,
        overloaded_links=float(overloaded),
        avg_latency_ns=weighted_latency / total_volume if total_volume else 0.0,
        max_utilization=max_util,
    )


def mean_metrics(runs: Sequence[Metrics]) -> Metrics:
    return Metrics(
        total_volume_gbits=statistics.fmean(m.total_volume_gbits for m in runs),
        overloaded_links=statistics.fmean(m.overloaded_links for m in runs),
        avg_latency_ns=statistics.fmean(m.avg_latency_ns for m in runs),
        max_utilization=statistics.fmean(m.max_utilization for m in runs),
    )


# --------------------------------------------------------------------------- #
# Routers under test
# --------------------------------------------------------------------------- #
class ECMPRouter:
    """Load-blind ECMP: pin each flow to a random shortest switch-only path."""

    def __init__(self, topology: FatTreeTopology, sync_window_s: float) -> None:
        self.graph: nx.DiGraph = topology.graph
        self.sync_window_s = sync_window_s
        self._path_cache: dict[tuple[str, str], list[tuple[str, ...]]] = {}

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

    def route(self, flows: Sequence[Flow], rng: random.Random) -> list[PlacedFlow]:
        return [
            PlacedFlow(
                path=rng.choice(self._equal_cost_paths(flow.source_gpu, flow.destination_gpu)),
                volume_gbits=flow.traffic_volume_gbits,
                demand_gbps=flow.traffic_volume_gbits / self.sync_window_s,
            )
            for flow in flows
        ]


def run_engine(
    engine: OpticalRoutingEngine, graph: nx.DiGraph, flows: Sequence[Flow]
) -> Metrics:
    """Route with the OpticalRoutingEngine, then score with the shared model."""
    engine.reset_fabric_load()
    routes = engine.allocate_routes(flows)
    placed = [
        PlacedFlow(
            path=route.path,
            volume_gbits=flows[index].traffic_volume_gbits,
            demand_gbps=route.demand_gbps,
        )
        for index, route in routes.items()
    ]
    metrics = evaluate(graph, placed)

    # Cross-check: the engine's own booked loads must match what we just scored.
    booked_max = engine.fabric_utilization_summary()["max_utilization"]
    if not math.isclose(booked_max, metrics.max_utilization, rel_tol=1e-9, abs_tol=1e-9):
        raise AssertionError(
            f"Engine booked max utilization {booked_max:.6f} != scored {metrics.max_utilization:.6f}"
        )

    engine.reset_fabric_load()  # leave the fabric clean for the next run
    return metrics


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


WORKLOADS: tuple[WorkloadSpec, ...] = (
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
)


# --------------------------------------------------------------------------- #
# Benchmark driver
# --------------------------------------------------------------------------- #
def benchmark_workload(
    spec: WorkloadSpec,
    generator: AIWorkloadGenerator,
    ecmp: ECMPRouter,
    engine: OpticalRoutingEngine,
    graph: nx.DiGraph,
    tensor_sizes_gb: Sequence[float],
    trials: int,
    seed: int,
) -> list[SweepRow]:
    rows: list[SweepRow] = []
    for size_gb in tensor_sizes_gb:
        flows = spec.builder(generator, size_gb)

        ecmp_runs = [
            evaluate(graph, ecmp.route(flows, random.Random(seed + trial)))
            for trial in range(trials)
        ]
        rows.append(
            SweepRow(
                tensor_size_gb=size_gb,
                ecmp=mean_metrics(ecmp_runs),
                engine=run_engine(engine, graph, flows),
            )
        )
    return rows


# --------------------------------------------------------------------------- #
# Dashboard
# --------------------------------------------------------------------------- #
_WIDTH = 108


def _first_overload(rows: Sequence[SweepRow], pick: Callable[[SweepRow], Metrics]) -> str:
    for row in rows:
        if pick(row).overloaded_links > 0:
            return f"{row.tensor_size_gb:.0f} GB"
    return "not reached in sweep"


def print_dashboard(
    results: dict[str, list[SweepRow]],
    k: int,
    gpu_count: int,
    trials: int,
    sync_window_s: float,
) -> None:
    bar = "=" * _WIDTH
    print(f"\n{bar}")
    print("  ECMP vs. OPTICAL ROUTING ENGINE - BENCHMARK DASHBOARD".center(_WIDTH))
    print(bar)
    print(
        f"  Fat-Tree k={k} | {gpu_count} GPUs | ECMP = mean of {trials} seeded trials | "
        f"sync window {sync_window_s:g} s"
    )
    print(
        f"  Latency model: base + util x {SERIALIZATION_PENALTY_NS:g} ns, "
        f"+ {SERIALIZATION_PENALTY_NS:g} ns x (exp({OVERLOAD_STEEPNESS:g} x overload) - 1) when util > 100%"
    )

    for spec in WORKLOADS:
        rows = results[spec.name]
        print(f"\n{'-' * _WIDTH}")
        print(f"  WORKLOAD: {spec.name} - {spec.description}")
        print("-" * _WIDTH)
        print(
            f"  {'Tensor':>6}  {'Total Volume':>13}  |  {'Overloaded Links':^17}  |  "
            f"{'Avg Packet Latency (ns)':^33}  |  {'Max Link Util':^17}"
        )
        print(
            f"  {'(GB)':>6}  {'(Gbit)':>13}  |  {'ECMP':>8} {'Engine':>8}  |  "
            f"{'ECMP':>10} {'Engine':>10} {'Gain':>9}  |  {'ECMP':>8} {'Engine':>8}"
        )
        print(f"  {'-' * 6}  {'-' * 13}  |  {'-' * 17}  |  {'-' * 33}  |  {'-' * 17}")

        for row in rows:
            gain = (
                (row.ecmp.avg_latency_ns - row.engine.avg_latency_ns)
                / row.ecmp.avg_latency_ns
                * 100.0
                if row.ecmp.avg_latency_ns
                else 0.0
            )
            print(
                f"  {row.tensor_size_gb:>6.0f}  {row.ecmp.total_volume_gbits:>13,.0f}  |  "
                f"{row.ecmp.overloaded_links:>8.1f} {row.engine.overloaded_links:>8.1f}  |  "
                f"{row.ecmp.avg_latency_ns:>10,.1f} {row.engine.avg_latency_ns:>10,.1f} {gain:>+8.1f}%  |  "
                f"{row.ecmp.max_utilization * 100:>7.1f}% {row.engine.max_utilization * 100:>7.1f}%"
            )

        print(f"\n  ECMP first overloads a link at:   {_first_overload(rows, lambda r: r.ecmp)}")
        print(f"  Engine first overloads a link at: {_first_overload(rows, lambda r: r.engine)}")

    print(f"\n{bar}")
    print("  Notes")
    print("   - Overloaded = directed link with booked demand above its bandwidth. Flows are")
    print("     unsplittable, so a link that every path must cross (e.g. a GPU's own access link),")
    print("     or one that a single flow already exceeds, overloads under both routers.")
    print("   - Latency is scored by one shared model for both routers, not by the engine's")
    print("     internal selection-time estimate.")
    print(bar)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark ECMP vs. OpticalRoutingEngine.")
    parser.add_argument("--k", type=int, default=4, help="Fat-Tree switch port count (even, >= 4)")
    parser.add_argument("--trials", type=int, default=50, help="seeded ECMP trials to average")
    parser.add_argument("--seed", type=int, default=2026, help="base seed for ECMP trials")
    parser.add_argument("--window", type=float, default=1.0, help="sync window in seconds")
    args = parser.parse_args(argv)

    if args.k < 4 or args.k % 2:
        parser.error("--k must be an even integer >= 4")
    if args.trials < 1:
        parser.error("--trials must be at least 1")
    if args.window <= 0:
        parser.error("--window must be positive")

    with contextlib.redirect_stdout(io.StringIO()):  # silence the topology's build message
        topology = FatTreeTopology(k=args.k)

    generator = AIWorkloadGenerator(topology)
    ecmp = ECMPRouter(topology, sync_window_s=args.window)
    engine = OpticalRoutingEngine(
        topology,
        serialization_penalty_ns=SERIALIZATION_PENALTY_NS,
        sync_window_s=args.window,
    )

    results = {
        spec.name: benchmark_workload(
            spec,
            generator,
            ecmp,
            engine,
            topology.graph,
            TENSOR_SIZES_GB,
            trials=args.trials,
            seed=args.seed,
        )
        for spec in WORKLOADS
    }

    print_dashboard(
        results,
        k=args.k,
        gpu_count=len(generator.get_gpu_nodes()),
        trials=args.trials,
        sync_window_s=args.window,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())