"""
core/routing_engine.py

Congestion-aware routing for the Silicon Photonics fabric.

`OpticalRoutingEngine` replaces static ECMP-style hashing with a dynamic
path-selection loop. For every flow it:

  1. enumerates all candidate paths between the source and destination GPUs,
  2. scores each path with a load-dependent latency model,
  3. books the flow's bandwidth demand on the winning path's links, so later
     flows see the congestion created by earlier ones.

Units
-----
- `current_load` on every edge is the booked demand in **Gbps**, the same unit
  as `bandwidth_gbps`, so `current_load / bandwidth_gbps` is link utilization.
  Utilization above 1.0 means the link is oversubscribed.
- A flow's `traffic_volume_gbits` is converted into a rate by spreading it over
  `sync_window_s` seconds: `demand_gbps = traffic_volume_gbits / sync_window_s`.
- Latencies are in nanoseconds.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import networkx as nx

# Allow running this file directly (python core/routing_engine.py) while still
# importing the sibling `simulation` and `workloads` packages.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from simulation.topology import FatTreeTopology  # noqa: E402

_PRIORITY_BY_NAME = {"LOW": 0, "NORMAL": 1, "HIGH": 2, "CRITICAL": 3}


class RoutingError(Exception):
    """Raised when a flow cannot be validated or routed."""


@dataclass(frozen=True, slots=True)
class RouteAllocation:
    """The routing decision made for one flow."""

    flow_index: int
    source_gpu: str
    destination_gpu: str
    path: tuple[str, ...]
    effective_latency_ns: float  # latency the chosen path had when it was selected
    base_latency_ns: float  # zero-load latency of the chosen path
    bottleneck_utilization: float  # worst link utilization on the path at selection time
    demand_gbps: float  # bandwidth booked on every link of the path
    priority: int

    @property
    def hop_count(self) -> int:
        return len(self.path) - 1


@dataclass(frozen=True, slots=True, eq=False)
class _CandidatePath:
    """A candidate path plus live references to its edge attribute dicts."""

    nodes: tuple[str, ...]
    edges: tuple[dict[str, Any], ...]


@dataclass(frozen=True, slots=True)
class _ParsedFlow:
    index: int
    source: str
    destination: str
    demand_gbps: float
    priority: int


class OpticalRoutingEngine:
    """Dynamic, congestion-aware path selection over a Fat-Tree topology."""

    def __init__(
        self,
        topology: FatTreeTopology,
        serialization_penalty_ns: float = 100.0,
        sync_window_s: float = 1.0,
    ) -> None:
        """
        :param topology: The FatTreeTopology whose graph is routed over.
        :param serialization_penalty_ns: Latency added to a link at 100% utilization.
                                         The per-link penalty scales linearly with
                                         utilization: (load / bandwidth) * penalty.
        :param sync_window_s: Time window over which a flow's volume is spread to
                              derive its bandwidth demand in Gbps.
        """
        graph = getattr(topology, "graph", None)
        if not isinstance(graph, nx.DiGraph):
            raise TypeError("topology must expose a networkx DiGraph as `.graph`")
        if not math.isfinite(serialization_penalty_ns) or serialization_penalty_ns < 0:
            raise ValueError("serialization_penalty_ns must be a finite, non-negative number")
        if not math.isfinite(sync_window_s) or sync_window_s <= 0:
            raise ValueError("sync_window_s must be a finite, positive number")

        self.topology = topology
        self.graph: nx.DiGraph = graph
        self.serialization_penalty_ns = float(serialization_penalty_ns)
        self.sync_window_s = float(sync_window_s)
        self._candidate_cache: dict[tuple[str, str], tuple[_CandidatePath, ...]] = {}

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def allocate_routes(self, active_flows: Iterable[Any]) -> dict[int, RouteAllocation]:
        """
        Route every flow over its currently least-congested path and book the load.

        Flows are processed in descending priority order (input order breaks ties),
        so urgent traffic gets first pick of the uncongested paths. Load accumulates
        across calls; call `reset_fabric_load()` between training batches.

        Flows may be `Flow` objects or mappings with the keys `source_gpu`,
        `destination_gpu`, `traffic_volume_gbits` and (optionally) `priority`.

        All flows are validated before any load is booked, so a bad flow raises
        `RoutingError` without leaving the fabric half-updated.

        :return: Mapping of flow index (position in `active_flows`) to its RouteAllocation.
        """
        parsed = [self._parse_flow(i, flow) for i, flow in enumerate(active_flows)]
        for flow in parsed:
            self._candidate_paths(flow.source, flow.destination)  # fail fast

        results: dict[int, RouteAllocation] = {}
        for flow in sorted(parsed, key=lambda f: -f.priority):
            results[flow.index] = self._route_single_flow(flow)
        return {index: results[index] for index in sorted(results)}

    def reset_fabric_load(self) -> None:
        """Clear all booked load, e.g. between training batches."""
        for _, _, attrs in self.graph.edges(data=True):
            attrs["current_load"] = 0.0

    def clear_path_cache(self) -> None:
        """Drop cached candidate paths. Call this if the graph structure changes."""
        self._candidate_cache.clear()

    def fabric_utilization_summary(self) -> dict[str, Any]:
        """Summarize link utilization across the fabric."""
        max_util = 0.0
        hottest_link: tuple[str, str] | None = None
        loaded_utils: list[float] = []
        overloaded = 0

        for u, v, attrs in self.graph.edges(data=True):
            util = attrs["current_load"] / attrs["bandwidth_gbps"]
            if util <= 0.0:
                continue
            loaded_utils.append(util)
            if util > 1.0:
                overloaded += 1
            if util > max_util:
                max_util, hottest_link = util, (u, v)

        return {
            "loaded_links": len(loaded_utils),
            "max_utilization": max_util,
            "hottest_link": hottest_link,
            "mean_loaded_utilization": (
                sum(loaded_utils) / len(loaded_utils) if loaded_utils else 0.0
            ),
            "overloaded_links": overloaded,
        }

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _parse_flow(self, index: int, flow: Any) -> _ParsedFlow:
        def read(name: str, default: Any = None) -> Any:
            if isinstance(flow, Mapping):
                return flow.get(name, default)
            return getattr(flow, name, default)

        source = read("source_gpu")
        destination = read("destination_gpu")
        volume = read("traffic_volume_gbits")

        if source is None or destination is None or volume is None:
            raise RoutingError(
                f"Flow {index} must provide source_gpu, destination_gpu and traffic_volume_gbits"
            )
        for node in (source, destination):
            if node not in self.graph:
                raise RoutingError(f"Flow {index}: node {node!r} is not in the topology")
            if self.graph.nodes[node].get("type") != "gpu":
                raise RoutingError(f"Flow {index}: node {node!r} is not a GPU")
        if source == destination:
            raise RoutingError(f"Flow {index}: source and destination are both {source!r}")
        if not isinstance(volume, (int, float)) or not math.isfinite(volume) or volume <= 0:
            raise RoutingError(f"Flow {index}: traffic_volume_gbits must be a positive number")

        return _ParsedFlow(
            index=index,
            source=source,
            destination=destination,
            demand_gbps=float(volume) / self.sync_window_s,
            priority=self._priority_value(read("priority", 0)),
        )

    @staticmethod
    def _priority_value(priority: Any) -> int:
        try:
            return int(priority)
        except (TypeError, ValueError):
            return _PRIORITY_BY_NAME.get(str(priority).upper(), 0)

    def _candidate_paths(self, source: str, destination: str) -> tuple[_CandidatePath, ...]:
        """
        All minimum-hop paths between two GPUs, routed through switches only.

        In a Fat-Tree the minimum-hop paths are exactly the valley-free
        up-then-down paths, so this is the full set of useful routes. Results are
        cached because the structure is static; only edge loads change.
        """
        key = (source, destination)
        cached = self._candidate_cache.get(key)
        if cached is not None:
            return cached

        try:
            node_paths = list(nx.all_shortest_paths(self.graph, source, destination))
        except nx.NetworkXNoPath as exc:
            raise RoutingError(f"No path from {source!r} to {destination!r}") from exc

        candidates = tuple(
            _CandidatePath(
                nodes=tuple(path),
                edges=tuple(self.graph[u][v] for u, v in zip(path, path[1:])),
            )
            for path in node_paths
            if all(self.graph.nodes[n].get("type") != "gpu" for n in path[1:-1])
        )
        if not candidates:
            raise RoutingError(f"No switch-only path from {source!r} to {destination!r}")

        self._candidate_cache[key] = candidates
        return candidates

    def _score_path(self, candidate: _CandidatePath) -> tuple[float, float, float]:
        """
        Return (effective_latency_ns, base_latency_ns, bottleneck_utilization).

        Per link: effective = base_latency + (current_load / bandwidth) * penalty.
        The path latency is the sum over its links.
        """
        effective = 0.0
        base = 0.0
        worst = 0.0
        for edge in candidate.edges:
            utilization = edge["current_load"] / edge["bandwidth_gbps"]
            base += edge["base_latency_ns"]
            effective += edge["base_latency_ns"] + utilization * self.serialization_penalty_ns
            worst = max(worst, utilization)
        return effective, base, worst

    def _route_single_flow(self, flow: _ParsedFlow) -> RouteAllocation:
        best: _CandidatePath | None = None
        best_score: tuple[float, float, float] | None = None

        for candidate in self._candidate_paths(flow.source, flow.destination):
            score = self._score_path(candidate)
            # Lowest latency wins; ties go to the path with the coolest bottleneck link.
            if best_score is None or (score[0], score[2]) < (best_score[0], best_score[2]):
                best, best_score = candidate, score

        assert best is not None and best_score is not None  # candidates are never empty

        for edge in best.edges:  # book the resources
            edge["current_load"] += flow.demand_gbps

        return RouteAllocation(
            flow_index=flow.index,
            source_gpu=flow.source,
            destination_gpu=flow.destination,
            path=best.nodes,
            effective_latency_ns=best_score[0],
            base_latency_ns=best_score[1],
            bottleneck_utilization=best_score[2],
            demand_gbps=flow.demand_gbps,
            priority=flow.priority,
        )


if __name__ == "__main__":
    from workloads.traffic_generator import AIWorkloadGenerator

    topo = FatTreeTopology(k=4)
    generator = AIWorkloadGenerator(topo)
    engine = OpticalRoutingEngine(topo)

    tensor_size_gb = 10.0
    flows = generator.generate_all_reduce_traffic(tensor_size_gb)
    routes = engine.allocate_routes(flows)

    print(f"\nOptimized routes for {len(routes)} All-Reduce flows ({tensor_size_gb} GB tensor):\n")
    for route in routes.values():
        print(
            f"  [{route.flow_index:02d}] {route.source_gpu} -> {route.destination_gpu} "
            f"| {route.hop_count} hops | {route.effective_latency_ns:7.1f} ns "
            f"(base {route.base_latency_ns:5.1f} ns, bottleneck {route.bottleneck_utilization:.0%})"
        )
        print(f"       {' -> '.join(route.path)}")

    summary = engine.fabric_utilization_summary()
    print("\nFabric utilization after booking:")
    print(f"  Loaded links:        {summary['loaded_links']}")
    print(f"  Max utilization:     {summary['max_utilization']:.0%} on {summary['hottest_link']}")
    print(f"  Mean (loaded links): {summary['mean_loaded_utilization']:.0%}")
    print(f"  Overloaded links:    {summary['overloaded_links']}")

    engine.reset_fabric_load()
    assert all(d["current_load"] == 0.0 for _, _, d in topo.graph.edges(data=True))
    print("\nFabric load reset: all links back to 0.0 Gbps.")