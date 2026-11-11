"""
core/routing_engine.py

Congestion-, power- and thermal-aware routing for the Silicon Photonics fabric.

`OpticalRoutingEngine` routes each flow over the best *feasible* path. A candidate
path is a minimum-hop, switch-only path between the two GPUs. It must pass these
physical checks before it is even scored:

  1. Optical power budget   total insertion loss <= max_insertion_loss_db (15 dB).
  2. WDM channels           every optical hop needs `channels_needed` free wavelengths.
  3. Copper bandwidth       every copper hop needs enough spare bandwidth_gbps.
  4. Thermal limit          (optional) no switch on the path is already at max_temp_c.

Feasible paths are ranked by

    effective_latency = base_latency + congestion_penalty + thermal_penalty

and the winner is booked: wavelengths are reserved, link load is added, and every
switch on the path heats up in proportion to the traffic, so later flows steer
around hot switches. A flow with no feasible path is *blocked* (admission control),
not raised as an error, and is reported with `routed=False` and a reason.

Units
-----
- `current_load` is booked demand in Gbps, the same unit as `bandwidth_gbps`.
- A flow's rate is `traffic_volume_gbits / sync_window_s` (equal to the volume in
  Gbps at the default 1 s window). Channels needed = ceil(rate / channel_bandwidth_gbps).
- Latencies are in nanoseconds, temperatures in degrees Celsius.

Wavelengths are allocated independently on every optical hop. This matches a fabric
of electrical packet switches fed by optical I/O, where each switch terminates the
light and can re-transmit on any wavelength, so no end-to-end wavelength
continuity constraint applies.
"""

from __future__ import annotations

import math
import sys
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx

# Allow running this file directly (python core/routing_engine.py) while still
# importing the sibling `simulation` and `workloads` packages.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from simulation.topology import FatTreeTopology  # noqa: E402

AMBIENT_TEMP_C = 35.0
DEFAULT_MAX_INSERTION_LOSS_DB = 15.0

_SWITCH_TYPES = frozenset({"core", "aggregation", "edge"})
_MEDIA_TYPES = frozenset({"optical", "copper"})
_PRIORITY_BY_NAME = {"LOW": 0, "NORMAL": 1, "HIGH": 2, "CRITICAL": 3}
_EPSILON = 1e-9

REJECT_POWER_BUDGET = "power_budget"
REJECT_THERMAL_LIMIT = "thermal_limit"
REJECT_WDM_CHANNELS = "wdm_channels"
REJECT_COPPER_BANDWIDTH = "copper_bandwidth"

_REQUIRED_EDGE_ATTRS = (
    "bandwidth_gbps",
    "base_latency_ns",
    "current_load",
    "media_type",
    "insertion_loss_db",
    "total_wdm_channels",
    "channel_bandwidth_gbps",
    "booked_channels",
)
_REQUIRED_SWITCH_ATTRS = ("temperature_c", "max_temp_c")


class RoutingError(Exception):
    """Raised when a flow is malformed or its endpoints are unreachable."""


@dataclass(frozen=True, slots=True)
class HopAllocation:
    """What one hop of a routed flow was given."""

    source: str
    destination: str
    media_type: str
    channels: tuple[int, ...]  # WDM channel IDs reserved; empty on copper hops
    insertion_loss_db: float

    @property
    def is_optical(self) -> bool:
        return self.media_type == "optical"


@dataclass(frozen=True, slots=True)
class RouteAllocation:
    """
    The routing decision for one flow.

    For a blocked flow (`routed=False`) the path and hops are empty,
    `reject_reason` says why, and the latency, loss and temperature fields are NaN.
    """

    flow_index: int
    source_gpu: str
    destination_gpu: str
    routed: bool
    path: tuple[str, ...]
    hops: tuple[HopAllocation, ...]
    total_insertion_loss_db: float
    effective_latency_ns: float  # base + congestion + thermal, as seen at selection time
    base_latency_ns: float  # zero-load, ambient-temperature latency of the path
    congestion_penalty_ns: float
    thermal_penalty_ns: float
    bottleneck_utilization: float  # worst link utilization on the path at selection time
    peak_switch_temp_c: float  # hottest switch on the path after this flow's heating
    demand_gbps: float  # bandwidth booked on every link of the path
    priority: int
    reject_reason: str | None = None

    @property
    def hop_count(self) -> int:
        return len(self.path) - 1 if self.path else 0

    @property
    def optical_hops(self) -> tuple[HopAllocation, ...]:
        return tuple(hop for hop in self.hops if hop.is_optical)


@dataclass(frozen=True, slots=True, eq=False)
class _CandidatePath:
    """A candidate path with live references to each hop's edge attribute dict."""

    nodes: tuple[str, ...]
    hops: tuple[tuple[str, str, dict[str, Any]], ...]
    switches: tuple[str, ...]  # interior nodes, all of them switches


@dataclass(frozen=True, slots=True)
class _ParsedFlow:
    index: int
    source: str
    destination: str
    demand_gbps: float
    priority: int


class OpticalRoutingEngine:
    """Physics-aware, congestion- and thermal-aware routing over a Fat-Tree."""

    def __init__(
        self,
        topology: FatTreeTopology,
        serialization_penalty_ns: float = 100.0,
        sync_window_s: float = 1.0,
        max_insertion_loss_db: float = DEFAULT_MAX_INSERTION_LOSS_DB,
        thermal_penalty_ns_per_c: float = 2.0,
        heating_c_per_gbps: float = 0.02,
        enforce_thermal_limit: bool = False,
    ) -> None:
        """
        :param topology: A FatTreeTopology carrying the optical/thermal attributes.
        :param serialization_penalty_ns: Latency added to a link at 100% utilization;
                                         the per-link penalty is utilization x this.
        :param sync_window_s: Window over which a flow's volume is spread to get Gbps.
        :param max_insertion_loss_db: Optical power budget. Paths above it are rejected.
        :param thermal_penalty_ns_per_c: Latency penalty per degree C above ambient,
                                         summed over every switch on the path.
        :param heating_c_per_gbps: Temperature rise per Gbps booked, applied to every
                                   switch on the path and capped at its max_temp_c.
        :param enforce_thermal_limit: If True, a switch already at max_temp_c is
                                      treated as shut down and paths through it are
                                      rejected. If False the penalty alone steers
                                      traffic and the temperature simply stays capped.
        """
        graph = getattr(topology, "graph", None)
        if not isinstance(graph, nx.DiGraph):
            raise TypeError("topology must expose a networkx DiGraph as `.graph`")
        for name, value in (
            ("serialization_penalty_ns", serialization_penalty_ns),
            ("max_insertion_loss_db", max_insertion_loss_db),
            ("thermal_penalty_ns_per_c", thermal_penalty_ns_per_c),
            ("heating_c_per_gbps", heating_c_per_gbps),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite, non-negative number")
        if not math.isfinite(sync_window_s) or sync_window_s <= 0:
            raise ValueError("sync_window_s must be a finite, positive number")

        self.topology = topology
        self.graph: nx.DiGraph = graph
        self.serialization_penalty_ns = float(serialization_penalty_ns)
        self.sync_window_s = float(sync_window_s)
        self.max_insertion_loss_db = float(max_insertion_loss_db)
        self.thermal_penalty_ns_per_c = float(thermal_penalty_ns_per_c)
        self.heating_c_per_gbps = float(heating_c_per_gbps)
        self.enforce_thermal_limit = bool(enforce_thermal_limit)
        self._candidate_cache: dict[tuple[str, str], tuple[_CandidatePath, ...]] = {}

        self._validate_topology()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def allocate_routes(self, active_flows: Iterable[Any]) -> dict[int, RouteAllocation]:
        """
        Route every flow over its best feasible path and book the resources.

        Flows are processed in descending priority order (input order breaks ties),
        so urgent traffic gets first pick. Load, wavelengths and heat accumulate
        across calls; call `reset_fabric_load()` between training batches.

        Flows may be `Flow` objects or mappings with the keys `source_gpu`,
        `destination_gpu`, `traffic_volume_gbits` and (optionally) `priority`.

        All flows are validated before anything is booked, so a malformed flow
        raises `RoutingError` without leaving the fabric half-updated. A well-formed
        flow that no path can carry is not an error: it comes back with
        `routed=False` and a `reject_reason`, and books nothing.

        :return: Mapping of flow index (position in `active_flows`) to RouteAllocation.
        """
        parsed = [self._parse_flow(i, flow) for i, flow in enumerate(active_flows)]
        for flow in parsed:
            self._candidate_paths(flow.source, flow.destination)  # fail fast

        results: dict[int, RouteAllocation] = {}
        for flow in sorted(parsed, key=lambda f: -f.priority):
            results[flow.index] = self._route_single_flow(flow)
        return {index: results[index] for index in sorted(results)}

    def reset_fabric_load(self) -> None:
        """Clear load and wavelength bookings on every edge and cool every switch."""
        for _, _, attrs in self.graph.edges(data=True):
            attrs["current_load"] = 0.0
            attrs["booked_channels"] = set()
        for _, attrs in self.graph.nodes(data=True):
            if attrs.get("type") in _SWITCH_TYPES:
                attrs["temperature_c"] = AMBIENT_TEMP_C

    def clear_path_cache(self) -> None:
        """Drop cached candidate paths. Call this if the graph structure changes."""
        self._candidate_cache.clear()

    def get_switch_temperatures(self) -> dict[str, float]:
        """Current temperature of every switch, keyed by node name."""
        return {
            node: attrs["temperature_c"]
            for node, attrs in self.graph.nodes(data=True)
            if attrs.get("type") in _SWITCH_TYPES
        }

    def fabric_utilization_summary(self) -> dict[str, Any]:
        """Summarize link utilization, wavelength usage and switch temperatures."""
        max_util = 0.0
        hottest_link: tuple[str, str] | None = None
        loaded_utils: list[float] = []
        overloaded = 0
        optical_booked = 0
        optical_total = 0

        for u, v, attrs in self.graph.edges(data=True):
            if attrs["media_type"] == "optical":
                optical_booked += len(attrs["booked_channels"])
                optical_total += attrs["total_wdm_channels"]
            util = attrs["current_load"] / attrs["bandwidth_gbps"]
            if util <= 0.0:
                continue
            loaded_utils.append(util)
            if util > 1.0 + _EPSILON:
                overloaded += 1
            if util > max_util:
                max_util, hottest_link = util, (u, v)

        temps = self.get_switch_temperatures()
        hottest_switch = max(temps, key=temps.__getitem__, default=None)

        return {
            "loaded_links": len(loaded_utils),
            "max_utilization": max_util,
            "hottest_link": hottest_link,
            "mean_loaded_utilization": (
                sum(loaded_utils) / len(loaded_utils) if loaded_utils else 0.0
            ),
            "overloaded_links": overloaded,
            "optical_channels_booked": optical_booked,
            "optical_channels_total": optical_total,
            "peak_switch_temp_c": temps[hottest_switch] if hottest_switch else AMBIENT_TEMP_C,
            "hottest_switch": hottest_switch,
        }

    # ------------------------------------------------------------------ #
    # Setup and validation
    # ------------------------------------------------------------------ #
    def _validate_topology(self) -> None:
        if not callable(getattr(self.topology, "get_available_wavelengths", None)):
            raise TypeError(
                "topology must provide get_available_wavelengths(u, v); "
                "use the optical version of simulation/topology.py"
            )
        for u, v, attrs in self.graph.edges(data=True):
            missing = [key for key in _REQUIRED_EDGE_ATTRS if key not in attrs]
            if missing:
                raise ValueError(
                    f"Edge {u}->{v} is missing optical attributes {missing}; "
                    "use the optical version of simulation/topology.py"
                )
            if attrs["media_type"] not in _MEDIA_TYPES:
                raise ValueError(
                    f"Edge {u}->{v} has unsupported media_type {attrs['media_type']!r}"
                )
        for node, attrs in self.graph.nodes(data=True):
            if attrs.get("type") in _SWITCH_TYPES:
                missing = [key for key in _REQUIRED_SWITCH_ATTRS if key not in attrs]
                if missing:
                    raise ValueError(f"Switch {node} is missing thermal attributes {missing}")

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

    # ------------------------------------------------------------------ #
    # Path enumeration
    # ------------------------------------------------------------------ #
    def _candidate_paths(self, source: str, destination: str) -> tuple[_CandidatePath, ...]:
        """
        All minimum-hop paths between two GPUs, routed through switches only.

        In a Fat-Tree the minimum-hop paths are exactly the valley-free
        up-then-down paths. Results are cached because the structure is static;
        edge and node attributes are read live on every use.
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
                hops=tuple((u, v, self.graph[u][v]) for u, v in zip(path, path[1:])),
                switches=tuple(path[1:-1]),
            )
            for path in node_paths
            if all(self.graph.nodes[n].get("type") != "gpu" for n in path[1:-1])
        )
        if not candidates:
            raise RoutingError(f"No switch-only path from {source!r} to {destination!r}")

        self._candidate_cache[key] = candidates
        return candidates

    # ------------------------------------------------------------------ #
    # Feasibility, scoring and booking
    # ------------------------------------------------------------------ #
    @staticmethod
    def _channels_needed(edge: Mapping[str, Any], demand_gbps: float) -> int:
        """Wavelengths needed to carry `demand_gbps` on this edge (at least one)."""
        return max(1, math.ceil(demand_gbps / edge["channel_bandwidth_gbps"] - _EPSILON))

    def _rejection_reason(self, candidate: _CandidatePath, demand_gbps: float) -> str | None:
        """Return why a path cannot carry the flow, or None if it can."""
        total_loss = sum(edge["insertion_loss_db"] for _, _, edge in candidate.hops)
        if total_loss > self.max_insertion_loss_db + _EPSILON:
            return REJECT_POWER_BUDGET

        if self.enforce_thermal_limit:
            for switch in candidate.switches:
                attrs = self.graph.nodes[switch]
                if attrs["temperature_c"] >= attrs["max_temp_c"] - _EPSILON:
                    return REJECT_THERMAL_LIMIT

        for u, v, edge in candidate.hops:
            if edge["media_type"] == "optical":
                needed = self._channels_needed(edge, demand_gbps)
                if len(self.topology.get_available_wavelengths(u, v)) < needed:
                    return REJECT_WDM_CHANNELS
            elif edge["current_load"] + demand_gbps > edge["bandwidth_gbps"] + _EPSILON:
                return REJECT_COPPER_BANDWIDTH
        return None

    def _score_path(self, candidate: _CandidatePath) -> tuple[float, float, float, float, float]:
        """
        Return (effective, base, congestion, thermal, worst_utilization), latencies in ns.

        congestion = sum over links of (current_load / bandwidth) x serialization penalty
        thermal    = sum over switches of (temperature_c - ambient) x penalty per degree C
        """
        base = 0.0
        congestion = 0.0
        worst = 0.0
        for _, _, edge in candidate.hops:
            utilization = edge["current_load"] / edge["bandwidth_gbps"]
            base += edge["base_latency_ns"]
            congestion += utilization * self.serialization_penalty_ns
            worst = max(worst, utilization)

        degrees_above_ambient = sum(
            max(0.0, self.graph.nodes[switch]["temperature_c"] - AMBIENT_TEMP_C)
            for switch in candidate.switches
        )
        thermal = degrees_above_ambient * self.thermal_penalty_ns_per_c
        return base + congestion + thermal, base, congestion, thermal, worst

    def _book(
        self, flow: _ParsedFlow, candidate: _CandidatePath
    ) -> tuple[tuple[HopAllocation, ...], float]:
        """Reserve wavelengths, add load and heat the switches. Returns (hops, peak temp)."""
        hops: list[HopAllocation] = []
        for u, v, edge in candidate.hops:
            channels: tuple[int, ...] = ()
            if edge["media_type"] == "optical":
                needed = self._channels_needed(edge, flow.demand_gbps)
                channels = tuple(self.topology.get_available_wavelengths(u, v)[:needed])
                if len(channels) != needed:  # guarded by _rejection_reason
                    raise RuntimeError(f"Wavelength shortfall on {u}->{v} while booking")
                edge["booked_channels"].update(channels)
            edge["current_load"] += flow.demand_gbps
            hops.append(
                HopAllocation(
                    source=u,
                    destination=v,
                    media_type=edge["media_type"],
                    channels=channels,
                    insertion_loss_db=edge["insertion_loss_db"],
                )
            )

        heat_c = self.heating_c_per_gbps * flow.demand_gbps
        peak = AMBIENT_TEMP_C
        for switch in candidate.switches:
            attrs = self.graph.nodes[switch]
            attrs["temperature_c"] = min(attrs["max_temp_c"], attrs["temperature_c"] + heat_c)
            peak = max(peak, attrs["temperature_c"])
        return tuple(hops), peak

    def _blocked(self, flow: _ParsedFlow, reason: str) -> RouteAllocation:
        return RouteAllocation(
            flow_index=flow.index,
            source_gpu=flow.source,
            destination_gpu=flow.destination,
            routed=False,
            path=(),
            hops=(),
            total_insertion_loss_db=math.nan,
            effective_latency_ns=math.nan,
            base_latency_ns=math.nan,
            congestion_penalty_ns=math.nan,
            thermal_penalty_ns=math.nan,
            bottleneck_utilization=math.nan,
            peak_switch_temp_c=math.nan,
            demand_gbps=flow.demand_gbps,
            priority=flow.priority,
            reject_reason=reason,
        )

    def _route_single_flow(self, flow: _ParsedFlow) -> RouteAllocation:
        best: _CandidatePath | None = None
        best_score: tuple[float, float, float, float, float] | None = None
        rejections: Counter[str] = Counter()

        for candidate in self._candidate_paths(flow.source, flow.destination):
            reason = self._rejection_reason(candidate, flow.demand_gbps)
            if reason is not None:
                rejections[reason] += 1
                continue
            score = self._score_path(candidate)
            # Lowest effective latency wins; ties go to the coolest bottleneck link.
            if best_score is None or (score[0], score[4]) < (best_score[0], best_score[4]):
                best, best_score = candidate, score

        if best is None or best_score is None:
            detail = ", ".join(f"{reason} x{count}" for reason, count in rejections.most_common())
            return self._blocked(
                flow, f"all {sum(rejections.values())} candidate path(s) rejected: {detail}"
            )

        hops, peak_temp = self._book(flow, best)
        return RouteAllocation(
            flow_index=flow.index,
            source_gpu=flow.source,
            destination_gpu=flow.destination,
            routed=True,
            path=best.nodes,
            hops=hops,
            total_insertion_loss_db=sum(hop.insertion_loss_db for hop in hops),
            effective_latency_ns=best_score[0],
            base_latency_ns=best_score[1],
            congestion_penalty_ns=best_score[2],
            thermal_penalty_ns=best_score[3],
            bottleneck_utilization=best_score[4],
            peak_switch_temp_c=peak_temp,
            demand_gbps=flow.demand_gbps,
            priority=flow.priority,
        )


def _format_channels(channels: tuple[int, ...]) -> str:
    return "[" + ",".join(str(c) for c in channels) + "]"


def _print_route(route: RouteAllocation) -> None:
    header = f"  [{route.flow_index:02d}] {route.source_gpu} -> {route.destination_gpu}"
    if not route.routed:
        print(f"{header} | BLOCKED: {route.reject_reason}")
        return
    print(
        f"{header} | {route.hop_count} hops | loss {route.total_insertion_loss_db:.2f} dB | "
        f"{route.effective_latency_ns:7.1f} ns "
        f"(base {route.base_latency_ns:.1f} + congestion {route.congestion_penalty_ns:.1f} "
        f"+ thermal {route.thermal_penalty_ns:.1f}) | peak {route.peak_switch_temp_c:.1f} C"
    )
    print(f"       {' -> '.join(route.path)}")
    for hop in route.optical_hops:
        print(
            f"       WDM {hop.source} -> {hop.destination}: "
            f"channels {_format_channels(hop.channels)} ({hop.insertion_loss_db:.1f} dB)"
        )


if __name__ == "__main__":
    from workloads.traffic_generator import AIWorkloadGenerator

    topo = FatTreeTopology(k=4)
    generator = AIWorkloadGenerator(topo)
    engine = OpticalRoutingEngine(topo)

    tensor_size_gb = 10.0  # 150 Gbit per ring flow -> 3 x 50 Gbps wavelengths per optical hop
    flows = generator.generate_all_reduce_traffic(tensor_size_gb)
    routes = engine.allocate_routes(flows)

    routed = [r for r in routes.values() if r.routed]
    print(f"\nRouted {len(routed)} of {len(routes)} All-Reduce flows ({tensor_size_gb} GB tensor):\n")
    for route in routes.values():
        _print_route(route)

    temps = engine.get_switch_temperatures()
    heated = sorted(
        ((name, t) for name, t in temps.items() if t > AMBIENT_TEMP_C), key=lambda item: -item[1]
    )
    print(f"\nSwitch temperatures ({len(heated)} of {len(temps)} switches above {AMBIENT_TEMP_C:.0f} C):")
    for name, temp in heated:
        max_temp = topo.graph.nodes[name]["max_temp_c"]
        print(f"  {name:<18} {temp:5.1f} C  ({temp - AMBIENT_TEMP_C:+5.1f}, limit {max_temp:.0f} C)")

    summary = engine.fabric_utilization_summary()
    print("\nFabric summary:")
    print(f"  Max link utilization: {summary['max_utilization']:.0%} on {summary['hottest_link']}")
    print(
        f"  WDM channels booked:  {summary['optical_channels_booked']} "
        f"of {summary['optical_channels_total']}"
    )
    print(f"  Hottest switch:       {summary['hottest_switch']} at {summary['peak_switch_temp_c']:.1f} C")

    # ---- Self-checks ---------------------------------------------------- #
    per_edge: dict[tuple[str, str], list[int]] = {}
    for route in routed:
        assert route.total_insertion_loss_db <= engine.max_insertion_loss_db
        for hop in route.optical_hops:
            per_edge.setdefault((hop.source, hop.destination), []).extend(hop.channels)
    for (u, v), assigned in per_edge.items():
        assert len(assigned) == len(set(assigned)), f"channel double-booked on {u}->{v}"
        assert set(assigned) == topo.graph.edges[u, v]["booked_channels"], (u, v)
    assert all(t <= topo.graph.nodes[n]["max_temp_c"] for n, t in temps.items())
    print("\nSelf-checks passed: no double-booked wavelengths, loss within budget, temps within limits.")

    # ---- Power-budget enforcement demo ----------------------------------- #
    engine.reset_fabric_load()
    strict = OpticalRoutingEngine(topo, max_insertion_loss_db=2.0)
    strict_routes = strict.allocate_routes(flows)
    blocked = [r for r in strict_routes.values() if not r.routed]
    print(
        f"\nWith a 2.0 dB power budget: {len(strict_routes) - len(blocked)} routed, "
        f"{len(blocked)} blocked (cross-pod paths cost 2.6 dB)."
    )
    if blocked:
        print(f"  e.g. flow {blocked[0].flow_index}: {blocked[0].reject_reason}")
    assert blocked and all("power_budget" in r.reject_reason for r in blocked)

    strict.reset_fabric_load()
    assert all(
        d["current_load"] == 0.0 and not d["booked_channels"] for _, _, d in topo.graph.edges(data=True)
    )
    assert all(t == AMBIENT_TEMP_C for t in strict.get_switch_temperatures().values())
    print("Fabric reset: load, wavelengths and temperatures are back to baseline.")