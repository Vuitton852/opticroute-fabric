"""
app.py

OpticRoute Fabric | Silicon Photonics Digital Twin.

An interactive co-simulation of blind ECMP versus the physics-aware OpticalRoutingEngine
on a simulated Fat-Tree fabric with WDM wavelength, insertion-loss and thermal limits.

Run with:
    pip install streamlit plotly pandas
    streamlit run app.py

The ECMP model, the shared latency/thermal scoring and the workload builders come from
tests/benchmark.py, so the numbers here match the benchmark dashboard for the same
settings. That file is loaded by path rather than with `import tests.benchmark`, so a
third-party package that ships a top-level `tests` module cannot shadow it.
"""

from __future__ import annotations

import contextlib
import importlib.util
import inspect
import io
import math
import random
import statistics
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.routing_engine import AMBIENT_TEMP_C, OpticalRoutingEngine  # noqa: E402
from simulation.topology import FatTreeTopology  # noqa: E402
from workloads.traffic_generator import AIWorkloadGenerator  # noqa: E402


def _load_benchmark() -> Any:
    """Load tests/benchmark.py by file path under a private module name."""
    name = "photonics_benchmark"
    cached = sys.modules.get(name)
    if cached is not None:
        return cached

    path = _PROJECT_ROOT / "tests" / "benchmark.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load the benchmark module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # must be registered before exec: dataclasses look it up
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise

    required = (
        "ECMPRouter", "PhysicsModel", "WORKLOADS", "THROTTLE_TEMP_C", "evaluate",
        "link_latency_ns", "mean_metrics", "run_engine", "exceeds_gpu_port",
    )  # fmt: skip
    missing = [attr for attr in required if not hasattr(module, attr)]
    if missing:
        sys.modules.pop(name, None)
        raise ImportError(f"{path} is missing expected names: {missing}")
    return module


bench = _load_benchmark()

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
PAGE_TITLE = "OpticRoute Fabric | Silicon Photonics Digital Twin"
ECMP_NAME = "Blind ECMP"
ENGINE_NAME = "OpticRoute Engine"

CLUSTER_SCALES: tuple[int, ...] = (4, 6, 8)
WORKLOAD_ORDER: tuple[str, ...] = ("Cross-Pod Shift", "Ring All-Reduce")
WORKLOADS = {spec.name: spec for spec in bench.WORKLOADS}
_missing_workloads = [name for name in WORKLOAD_ORDER if name not in WORKLOADS]
if _missing_workloads:
    raise ImportError(f"Benchmark is missing workloads: {_missing_workloads}")

TIERS: tuple[str, ...] = ("core", "aggregation", "edge")
NODE_LABELS = {"core": "Core", "aggregation": "Aggregation", "edge": "Edge", "gpu": "GPU"}

THROTTLE_TEMP_C: float = bench.THROTTLE_TEMP_C
MAX_TEMP_C: float = getattr(FatTreeTopology, "SWITCH_MAX_TEMP_C", 85.0)
DEFAULT_HEATING_C_PER_GBPS: float = inspect.signature(OpticalRoutingEngine.__init__).parameters[
    "heating_c_per_gbps"
].default

ECMP_COLOR = "#D55E00"  # Okabe-Ito vermillion
ENGINE_COLOR = "#0072B2"  # Okabe-Ito blue
THRESHOLD_COLOR = "#D62728"
STATE_KEY = "cosim_result"
_EPSILON = 1e-9

HOP_COLUMNS = [
    "Flow", "Source", "Destination", "Hop", "Link", "Link tier",
    "Wavelength channels", "Channels", "Reserved (Gbps)", "Hop loss (dB)",
]  # fmt: skip
FLOW_COLUMNS = [
    "Flow", "Source", "Destination", "Hops", "Path", "Insertion loss (dB)",
    "Peak switch temp (°C)", "WDM channels per optical link",
]  # fmt: skip
BLOCKED_COLUMNS = ["Flow", "Source", "Destination", "Reason"]

_HEADER_CSS = """
<style>
.or-header {
    background: linear-gradient(90deg, #0B1F3A 0%, #0A4C7A 60%, #0072B2 100%);
    border-radius: 10px;
    padding: 1.1rem 1.5rem;
    margin-bottom: 1rem;
}
.or-title {
    color: #FFFFFF;
    font-size: 1.9rem;
    font-weight: 700;
    letter-spacing: 0.02em;
    line-height: 1.2;
}
.or-subtitle {
    color: #CFE3F3;
    font-size: 1rem;
    margin-top: 0.2rem;
}
</style>
<div class="or-header">
    <div class="or-title">OpticRoute Fabric</div>
    <div class="or-subtitle">Silicon Photonics Digital Twin &middot; Co-Simulation Dashboard</div>
</div>
"""


# --------------------------------------------------------------------------- #
# Simulation (no Streamlit calls in this section)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class SimParams:
    k: int
    workload: str
    tensor_gb: int
    power_budget_db: float
    heating_c_per_gbps: float
    ecmp_trials: int
    seed: int


@dataclass(frozen=True, slots=True)
class TierTemps:
    peak: float
    mean: float


@dataclass(frozen=True, slots=True)
class LatencyParts:
    """Volume-weighted average latency split into its three components, in ns."""

    base: float
    congestion: float  # includes the exponential penalty on overloaded links
    thermal: float

    @property
    def total(self) -> float:
        return self.base + self.congestion + self.thermal


@dataclass(slots=True, eq=False)
class CoSimResult:
    params: SimParams
    total_flows: int
    gpu_count: int
    ecmp: Any  # benchmark.Metrics, mean across ECMP trials
    engine: Any  # benchmark.Metrics
    ecmp_latency: LatencyParts
    engine_latency: LatencyParts
    ecmp_over_budget_flows: float  # mean flows per trial that ECMP sends over a path above budget
    port_limited: bool  # a single flow exceeds its GPU's access link
    ecmp_tiers: dict[str, TierTemps]
    engine_tiers: dict[str, TierTemps]
    switch_temps: pd.DataFrame
    flow_table: pd.DataFrame
    hop_table: pd.DataFrame
    blocked_table: pd.DataFrame
    max_routed_path_loss_db: float  # NaN when the engine routed nothing
    wdm_channels_booked: int
    wdm_channels_total: int
    elapsed_s: float


def format_channels(channels: Sequence[int]) -> str:
    """Compress channel IDs into ranges, e.g. [0, 1, 2, 5] -> '0-2, 5'."""
    if not channels:
        return "-"
    ordered = sorted(channels)
    ranges: list[tuple[int, int]] = []
    start = previous = ordered[0]
    for channel in ordered[1:]:
        if channel == previous + 1:
            previous = channel
            continue
        ranges.append((start, previous))
        start = previous = channel
    ranges.append((start, previous))
    return ", ".join(str(a) if a == b else f"{a}-{b}" for a, b in ranges)


def latency_gain_pct(ecmp_ns: float, engine_ns: float) -> float | None:
    """Percent latency reduction of the engine versus ECMP, or None if undefined."""
    if math.isnan(ecmp_ns) or math.isnan(engine_ns) or ecmp_ns <= 0:
        return None
    return (ecmp_ns - engine_ns) / ecmp_ns * 100.0


def latency_breakdown(graph: Any, run: Any, physics: Any) -> LatencyParts:
    """
    Split a run's average latency into base, congestion and thermal parts, using exactly
    the benchmark's scoring: link latency from bench.link_latency_ns, plus a thermal
    penalty per degree above ambient on every switch along the path.
    """
    base = congestion = thermal = volume = 0.0
    for flow in run.placed:
        flow_base = flow_congestion = flow_thermal = 0.0
        for edge in zip(flow.path, flow.path[1:]):
            attrs = graph.edges[edge]
            utilization = run.load.get(edge, 0.0) / attrs["bandwidth_gbps"]
            link_ns = bench.link_latency_ns(
                attrs["base_latency_ns"], utilization, physics.serialization_penalty_ns
            )
            flow_base += attrs["base_latency_ns"]
            flow_congestion += link_ns - attrs["base_latency_ns"]
        for switch in flow.path[1:-1]:
            degrees_above_ambient = max(0.0, run.temps[switch] - AMBIENT_TEMP_C)
            flow_thermal += degrees_above_ambient * physics.thermal_penalty_ns_per_c
        base += flow_base * flow.volume_gbits
        congestion += flow_congestion * flow.volume_gbits
        thermal += flow_thermal * flow.volume_gbits
        volume += flow.volume_gbits

    if volume == 0:
        return LatencyParts(math.nan, math.nan, math.nan)
    return LatencyParts(base / volume, congestion / volume, thermal / volume)


def _mean_latency_parts(parts: Sequence[LatencyParts]) -> LatencyParts:
    defined = [p for p in parts if not math.isnan(p.base)]
    if not defined:
        return LatencyParts(math.nan, math.nan, math.nan)
    return LatencyParts(
        base=statistics.fmean(p.base for p in defined),
        congestion=statistics.fmean(p.congestion for p in defined),
        thermal=statistics.fmean(p.thermal for p in defined),
    )


def _check_breakdown(label: str, parts: LatencyParts, metrics: Any) -> None:
    """The breakdown must add up to the benchmark's own average latency."""
    if math.isnan(parts.total) and math.isnan(metrics.avg_latency_ns):
        return
    if not math.isclose(parts.total, metrics.avg_latency_ns, rel_tol=1e-6, abs_tol=1e-6):
        raise AssertionError(
            f"{label} latency breakdown {parts.total:.6f} ns != benchmark {metrics.avg_latency_ns:.6f} ns"
        )


def _tier_stats(temps: dict[str, float], tier_of: dict[str, str]) -> dict[str, TierTemps]:
    stats: dict[str, TierTemps] = {}
    for tier in TIERS:
        values = [temp for switch, temp in temps.items() if tier_of[switch] == tier]
        stats[tier] = TierTemps(peak=max(values), mean=statistics.fmean(values))
    return stats


def _average_tier_stats(runs: Sequence[dict[str, TierTemps]]) -> dict[str, TierTemps]:
    return {
        tier: TierTemps(
            peak=statistics.fmean(run[tier].peak for run in runs),
            mean=statistics.fmean(run[tier].mean for run in runs),
        )
        for tier in TIERS
    }


def run_cosimulation(params: SimParams) -> CoSimResult:
    """Run blind ECMP and the OpticalRoutingEngine on identical flows and score both."""
    started = time.perf_counter()
    spec = WORKLOADS[params.workload]

    # A fresh topology and engine per run: the engine books load onto the graph, so
    # nothing is shared between runs or between browser sessions.
    with contextlib.redirect_stdout(io.StringIO()):  # silence the topology build message
        topology = FatTreeTopology(k=params.k)
    graph = topology.graph
    engine = OpticalRoutingEngine(
        topology,
        max_insertion_loss_db=params.power_budget_db,
        heating_c_per_gbps=params.heating_c_per_gbps,
    )
    physics = bench.PhysicsModel.from_engine(engine)
    generator = AIWorkloadGenerator(topology)
    flows = spec.builder(generator, float(params.tensor_gb))
    total_flows = len(flows)
    tier_of = {node: attrs["type"] for node, attrs in graph.nodes(data=True) if attrs["type"] in TIERS}

    # ---- Blind ECMP: mean over seeded trials ------------------------------- #
    ecmp_router = bench.ECMPRouter(graph, physics)
    ecmp_runs = [
        ecmp_router.route(flows, random.Random(params.seed + trial))
        for trial in range(params.ecmp_trials)
    ]
    ecmp_metrics = bench.mean_metrics(
        [bench.evaluate(graph, run, physics, total_flows) for run in ecmp_runs]
    )
    ecmp_latency = _mean_latency_parts([latency_breakdown(graph, run, physics) for run in ecmp_runs])
    _check_breakdown("ECMP", ecmp_latency, ecmp_metrics)

    loss_cache: dict[tuple[str, ...], float] = {}

    def path_loss_db(path: tuple[str, ...]) -> float:
        if path not in loss_cache:
            loss_cache[path] = sum(
                graph.edges[edge]["insertion_loss_db"] for edge in zip(path, path[1:])
            )
        return loss_cache[path]

    over_budget = statistics.fmean(
        sum(1 for flow in run.placed if path_loss_db(flow.path) > params.power_budget_db + _EPSILON)
        for run in ecmp_runs
    )
    ecmp_tiers = _average_tier_stats([_tier_stats(run.temps, tier_of) for run in ecmp_runs])
    ecmp_switch_temps = {
        switch: statistics.fmean(run.temps[switch] for run in ecmp_runs) for switch in tier_of
    }

    # ---- OpticRoute Engine -------------------------------------------------- #
    engine_run = bench.run_engine(engine, flows)  # includes the engine's integrity checks
    engine_metrics = bench.evaluate(graph, engine_run, physics, total_flows)
    engine_latency = latency_breakdown(graph, engine_run, physics)
    _check_breakdown("Engine", engine_latency, engine_metrics)
    engine_tiers = _tier_stats(engine_run.temps, tier_of)

    # The engine is deterministic, so a second pass reproduces the same routes. This
    # one is kept for the path inspector (run_engine resets the fabric when done).
    engine.reset_fabric_load()
    routes = engine.allocate_routes(flows)
    blocked = [route for route in routes.values() if not route.routed]
    if len(blocked) != engine_run.failed_flows:
        raise AssertionError("Engine runs disagree on how many flows were blocked")
    summary = engine.fabric_utilization_summary()

    flow_rows: list[dict[str, Any]] = []
    hop_rows: list[dict[str, Any]] = []
    for index, route in routes.items():
        if not route.routed:
            continue
        optical = route.optical_hops
        flow_rows.append(
            {
                "Flow": index,
                "Source": route.source_gpu,
                "Destination": route.destination_gpu,
                "Hops": route.hop_count,
                "Path": " -> ".join(route.path),
                "Insertion loss (dB)": round(route.total_insertion_loss_db, 2),
                "Peak switch temp (°C)": round(route.peak_switch_temp_c, 1),
                "WDM channels per optical link": "; ".join(
                    f"{hop.source} -> {hop.destination}: {format_channels(hop.channels)}"
                    for hop in optical
                )
                or "none (copper only)",
            }
        )
        for hop_number, hop in enumerate(route.hops, start=1):
            if not hop.is_optical:
                continue
            attrs = graph.edges[hop.source, hop.destination]
            hop_rows.append(
                {
                    "Flow": index,
                    "Source": route.source_gpu,
                    "Destination": route.destination_gpu,
                    "Hop": hop_number,
                    "Link": f"{hop.source} -> {hop.destination}",
                    "Link tier": (
                        f"{NODE_LABELS[graph.nodes[hop.source]['type']]} -> "
                        f"{NODE_LABELS[graph.nodes[hop.destination]['type']]}"
                    ),
                    "Wavelength channels": format_channels(hop.channels),
                    "Channels": len(hop.channels),
                    "Reserved (Gbps)": len(hop.channels) * attrs["channel_bandwidth_gbps"],
                    "Hop loss (dB)": round(hop.insertion_loss_db, 2),
                }
            )
    blocked_rows = [
        {
            "Flow": route.flow_index,
            "Source": route.source_gpu,
            "Destination": route.destination_gpu,
            "Reason": route.reject_reason,
        }
        for route in blocked
    ]
    switch_rows = [
        {
            "Switch": switch,
            "Tier": NODE_LABELS[tier_of[switch]],
            "ECMP": ecmp_switch_temps[switch],
            "Engine": engine_run.temps[switch],
        }
        for switch in sorted(tier_of, key=lambda s: (TIERS.index(tier_of[s]), s))
    ]

    routed_losses = [route.total_insertion_loss_db for route in routes.values() if route.routed]
    engine.reset_fabric_load()

    return CoSimResult(
        params=params,
        total_flows=total_flows,
        gpu_count=len(generator.get_gpu_nodes()),
        ecmp=ecmp_metrics,
        engine=engine_metrics,
        ecmp_latency=ecmp_latency,
        engine_latency=engine_latency,
        ecmp_over_budget_flows=over_budget,
        port_limited=bench.exceeds_gpu_port(graph, flows, physics.sync_window_s),
        ecmp_tiers=ecmp_tiers,
        engine_tiers=engine_tiers,
        switch_temps=pd.DataFrame(switch_rows, columns=["Switch", "Tier", "ECMP", "Engine"]),
        flow_table=pd.DataFrame(flow_rows, columns=FLOW_COLUMNS),
        hop_table=pd.DataFrame(hop_rows, columns=HOP_COLUMNS),
        blocked_table=pd.DataFrame(blocked_rows, columns=BLOCKED_COLUMNS),
        max_routed_path_loss_db=max(routed_losses, default=math.nan),
        wdm_channels_booked=int(summary["optical_channels_booked"]),
        wdm_channels_total=int(summary["optical_channels_total"]),
        elapsed_s=time.perf_counter() - started,
    )


# --------------------------------------------------------------------------- #
# Formatting helpers
# --------------------------------------------------------------------------- #
def _fmt_num(value: float, decimals: int = 2) -> str:
    text = f"{value:.{decimals}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return "0" if text in ("", "-0") else text


def _signed(value: float, decimals: int = 2) -> str:
    text = f"{value:+.{decimals}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def _lower_is_better_delta(delta: float, unit: str = "", decimals: int = 2) -> tuple[str, str]:
    """Delta text and Streamlit colour mode for a metric where lower is better."""
    if abs(delta) < 0.5 * 10 ** (-decimals):
        return "no change vs ECMP", "off"
    return f"{_signed(delta, decimals)}{unit} vs ECMP", "inverse"


def _stretch(func: Any) -> dict[str, Any]:
    """
    Full-width kwargs that work on both older and newer Streamlit releases.

    Newer releases take width="stretch" (the parameter's default is a string); older ones
    use use_container_width=True. Some old widgets, such as st.dataframe, already have an
    integer `width` parameter, so its mere presence is not enough to tell them apart.
    """
    try:
        width = inspect.signature(func).parameters.get("width")
    except (TypeError, ValueError):
        width = None
    if width is not None and isinstance(width.default, str):
        return {"width": "stretch"}
    return {"use_container_width": True}


def _nan_to_none(value: float) -> float | None:
    return None if math.isnan(value) else value


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #
def build_tier_chart(result: CoSimResult, statistic: str) -> go.Figure:
    """Chart 1: grouped bars of switch temperature per tier, ECMP vs. engine."""
    attr = "peak" if statistic == "Peak" else "mean"
    tiers = [NODE_LABELS[tier] for tier in TIERS]
    fig = go.Figure()
    for name, stats, color in (
        (f"{ECMP_NAME} (mean of trials)", result.ecmp_tiers, ECMP_COLOR),
        (ENGINE_NAME, result.engine_tiers, ENGINE_COLOR),
    ):
        values = [getattr(stats[tier], attr) for tier in TIERS]
        fig.add_bar(
            name=name,
            x=tiers,
            y=values,
            marker_color=color,
            text=[f"{value:.1f}" for value in values],
            textposition="outside",
            hovertemplate=f"%{{x}}<br>{statistic} %{{y:.1f}} °C<extra>{name}</extra>",
        )
    fig.add_hline(
        y=THROTTLE_TEMP_C,
        line_dash="dash",
        line_color=THRESHOLD_COLOR,
        annotation_text=f"Throttle threshold {THROTTLE_TEMP_C:.1f} °C",
        annotation_position="top left",
    )
    fig.update_layout(
        barmode="group",
        xaxis_title="Switch tier",
        yaxis_title=f"{statistic} switch temperature (°C)",
        yaxis_range=[AMBIENT_TEMP_C - 5.0, MAX_TEMP_C + 10.0],
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=10, r=10, t=40, b=10),
        height=420,
    )
    return fig


def build_latency_chart(result: CoSimResult) -> go.Figure:
    """Chart 2: base, congestion and thermal latency components, ECMP vs. engine."""
    labels = ["Base latency", "Congestion penalty", "Thermal drift penalty"]
    fig = go.Figure()
    for name, parts, color in (
        (f"{ECMP_NAME} (mean of trials)", result.ecmp_latency, ECMP_COLOR),
        (ENGINE_NAME, result.engine_latency, ENGINE_COLOR),
    ):
        values = [_nan_to_none(v) for v in (parts.base, parts.congestion, parts.thermal)]
        fig.add_bar(
            name=name,
            x=labels,
            y=values,
            marker_color=color,
            text=["n/a" if v is None else f"{v:,.1f}" for v in values],
            textposition="outside",
            hovertemplate=f"%{{x}}<br>%{{y:,.1f}} ns<extra>{name}</extra>",
        )
    fig.update_layout(
        barmode="group",
        xaxis_title="Latency component",
        yaxis_title="Average latency per flow (ns)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=10, r=10, t=40, b=10),
        height=420,
    )
    return fig


def build_switch_chart(switch_temps: pd.DataFrame) -> go.Figure:
    """Grouped bars of every individual switch's temperature."""
    fig = go.Figure()
    for name, column, color in (
        (f"{ECMP_NAME} (mean of trials)", "ECMP", ECMP_COLOR),
        (ENGINE_NAME, "Engine", ENGINE_COLOR),
    ):
        fig.add_bar(
            name=name,
            x=switch_temps["Switch"],
            y=switch_temps[column],
            marker_color=color,
            hovertemplate=f"%{{x}}<br>%{{y:.1f}} °C<extra>{name}</extra>",
        )
    fig.add_hline(y=THROTTLE_TEMP_C, line_dash="dash", line_color=THRESHOLD_COLOR)
    fig.update_xaxes(
        categoryorder="array",
        categoryarray=list(switch_temps["Switch"]),
        tickangle=-60,
        tickfont=dict(size=9),
    )
    fig.update_layout(
        barmode="group",
        yaxis_title="Temperature (°C)",
        yaxis_range=[AMBIENT_TEMP_C - 5.0, MAX_TEMP_C + 10.0],
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=10, r=10, t=40, b=10),
        height=460,
    )
    return fig


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
def render_header() -> None:
    st.markdown(_HEADER_CSS, unsafe_allow_html=True)


def render_sidebar() -> tuple[SimParams, bool]:
    with st.sidebar:
        st.header("Simulation controls")
        k = st.selectbox(
            "Cluster scale",
            CLUSTER_SCALES,
            format_func=lambda value: f"k = {value}  ({value**3 // 4} GPUs)",
        )
        workload = st.selectbox("Workload pattern", WORKLOAD_ORDER)
        tensor_gb = st.slider("Tensor size", min_value=10, max_value=50, value=50, step=5, format="%d GB")
        power_budget_db = st.slider(
            "Optical power budget",
            min_value=2.0,
            max_value=15.0,
            value=15.0,
            step=0.1,
            format="%.1f dB",
            help="The engine rejects any path whose total insertion loss exceeds this. "
            "Blind ECMP does not check it.",
        )
        with st.expander("Advanced model settings"):
            heating = st.number_input(
                "Switch heating (°C per Gbps)",
                min_value=0.0,
                max_value=1.0,
                value=float(DEFAULT_HEATING_C_PER_GBPS),
                step=0.005,
                format="%.3f",
                help="Temperature rise per Gbps carried, applied to every switch on a flow's path. "
                "A model parameter, not a measured hardware value.",
            )
            trials = st.slider("ECMP random trials", min_value=5, max_value=100, value=50, step=5)
            seed = st.number_input("ECMP random seed", min_value=0, value=2026, step=1)
        run_clicked = st.button("Run Co-Simulation", type="primary", **_stretch(st.button))
        st.caption("Both routers receive identical flows in identical order.")

    params = SimParams(
        k=int(k),
        workload=str(workload),
        tensor_gb=int(tensor_gb),
        power_budget_db=round(float(power_budget_db), 1),
        heating_c_per_gbps=float(heating),
        ecmp_trials=int(trials),
        seed=int(seed),
    )
    return params, bool(run_clicked)


def _kpi_card(
    column: Any,
    label: str,
    engine_value: str,
    delta: str | None,
    delta_color: str,
    ecmp_line: str,
    help_text: str,
) -> None:
    with column, st.container(border=True):
        st.metric(label, engine_value, delta=delta, delta_color=delta_color, help=help_text)
        st.caption(ecmp_line)


def render_kpis(result: CoSimResult) -> None:
    ecmp, engine = result.ecmp, result.engine
    trials = result.params.ecmp_trials
    columns = st.columns(4)

    delta, color = _lower_is_better_delta(engine.failed_flows - ecmp.failed_flows)
    _kpi_card(
        columns[0],
        "Blocked / Collided Flows",
        _fmt_num(engine.failed_flows),
        delta,
        color,
        f"{ECMP_NAME}: {_fmt_num(ecmp.failed_flows)} of {result.total_flows} flows "
        f"(collided, mean of {trials} trials)",
        f"{ENGINE_NAME} value is flows it blocked (no feasible path). ECMP collides when a flow "
        "needs more wavelengths than are free or overbooks a copper link.",
    )

    delta, color = _lower_is_better_delta(engine.peak_temp_c - ecmp.peak_temp_c, " °C", 1)
    _kpi_card(
        columns[1],
        "Peak Switch Temp (°C)",
        f"{engine.peak_temp_c:.1f}",
        delta,
        color,
        f"{ECMP_NAME}: {ecmp.peak_temp_c:.1f} °C (mean of {trials} trials)",
        f"Hottest switch after all flows. Ambient {AMBIENT_TEMP_C:.0f} °C, limit {MAX_TEMP_C:.0f} °C.",
    )

    delta, color = _lower_is_better_delta(engine.hot_switches - ecmp.hot_switches)
    _kpi_card(
        columns[2],
        f"Throttled Switches (≥{THROTTLE_TEMP_C:.0f}°C)",
        _fmt_num(engine.hot_switches),
        delta,
        color,
        f"{ECMP_NAME}: {_fmt_num(ecmp.hot_switches)} (mean of {trials} trials)",
        f"Switches at or above {THROTTLE_TEMP_C:.0f} °C. A fractional ECMP value means only some "
        "trials throttled.",
    )

    gain = latency_gain_pct(ecmp.avg_latency_ns, engine.avg_latency_ns)
    ecmp_text = "n/a" if math.isnan(ecmp.avg_latency_ns) else f"{ecmp.avg_latency_ns:,.1f} ns"
    engine_text = "n/a" if math.isnan(engine.avg_latency_ns) else f"{engine.avg_latency_ns:,.1f}"
    _kpi_card(
        columns[3],
        "Avg Effective Latency (ns)",
        engine_text,
        None if gain is None else f"{gain:+.1f}% gain",
        "normal",
        f"{ECMP_NAME}: {ecmp_text} (mean of {trials} trials)",
        "Base + congestion + thermal penalties, weighted by traffic volume, over delivered flows.",
    )


def render_notices(result: CoSimResult) -> None:
    engine = result.engine
    if engine.failed_flows > 0:
        st.warning(
            f"The engine blocked {_fmt_num(engine.failed_flows)} of {result.total_flows} flows. Its "
            "latency and temperature figures cover the routed flows only, and blocked traffic "
            "generates no heat. Compare these results by failed-flow counts first."
        )
    if result.port_limited:
        st.info(
            "At this tensor size a single flow exceeds its GPU's access link, so no router can "
            "carry it cleanly. Compare by failed flows, not latency."
        )
    if result.ecmp_over_budget_flows > 0:
        st.warning(
            f"Blind ECMP ignores the optical power budget: on average "
            f"{_fmt_num(result.ecmp_over_budget_flows)} of its flows take paths above "
            f"{result.params.power_budget_db:.1f} dB, where the signal would be unreadable. These are "
            "not included in its collided-flow count."
        )


def render_charts(result: CoSimResult) -> None:
    st.subheader("Physics diagnostics")
    left, right = st.columns(2)

    with left:
        st.markdown("**Thermal load balancing**")
        statistic = st.radio("Statistic", ["Peak", "Mean"], horizontal=True, key="tier_statistic")
        st.plotly_chart(
            build_tier_chart(result, statistic), key="tier_chart", **_stretch(st.plotly_chart)
        )
        st.caption(
            "Peak is the hottest switch in each tier; Mean averages the tier. Dashed red line: "
            f"{THROTTLE_TEMP_C:.1f} °C throttle threshold."
        )

    with right:
        st.markdown("**Latency & congestion breakdown**")
        st.plotly_chart(build_latency_chart(result), key="latency_chart", **_stretch(st.plotly_chart))
        ecmp_total, engine_total = result.ecmp_latency.total, result.engine_latency.total
        engine_total_text = "n/a" if math.isnan(engine_total) else f"{engine_total:,.1f} ns"
        st.caption(
            f"Total: {ECMP_NAME} {ecmp_total:,.1f} ns vs {ENGINE_NAME} {engine_total_text}. "
            "Congestion includes the exponential penalty on overloaded links."
        )

    with st.expander("Per-switch temperatures"):
        st.plotly_chart(
            build_switch_chart(result.switch_temps), key="switch_chart", **_stretch(st.plotly_chart)
        )


def render_inspector(result: CoSimResult) -> None:
    st.subheader("Optical wavelength & path inspector")
    with st.expander("Routed flows: path hops, insertion loss, peak temperature and WDM channels"):
        if result.flow_table.empty:
            st.write("The engine routed no flows with these settings.")
        else:
            st.caption(
                f"{len(result.flow_table):,} routed flows from the {ENGINE_NAME}. Peak switch temp is the "
                "hottest switch on the path right after the flow was booked. Each optical link reserves "
                "the lowest free wavelength IDs; copper hops use no wavelengths."
            )
            st.dataframe(result.flow_table, hide_index=True, **_stretch(st.dataframe))
            if not result.hop_table.empty:
                st.markdown("**Per-hop wavelength detail**")
                st.dataframe(result.hop_table, hide_index=True, **_stretch(st.dataframe))

    if not result.blocked_table.empty:
        with st.expander(f"Blocked flows ({len(result.blocked_table)})"):
            st.dataframe(result.blocked_table, hide_index=True, **_stretch(st.dataframe))


def render_results(result: CoSimResult) -> None:
    params = result.params
    loss_text = (
        "n/a" if math.isnan(result.max_routed_path_loss_db) else f"{result.max_routed_path_loss_db:.2f} dB"
    )
    channel_pct = (
        result.wdm_channels_booked / result.wdm_channels_total * 100.0
        if result.wdm_channels_total
        else 0.0
    )
    st.caption(
        f"k = {params.k} ({result.gpu_count} GPUs) · {params.workload} · {params.tensor_gb} GB "
        f"· {result.total_flows} flows · power budget {params.power_budget_db:.1f} dB "
        f"· ran in {result.elapsed_s:.1f} s"
    )

    render_kpis(result)
    render_notices(result)
    st.caption(
        f"{ENGINE_NAME}: longest routed path {loss_text} of the {params.power_budget_db:.1f} dB budget "
        f"· {result.wdm_channels_booked:,} of {result.wdm_channels_total:,} wavelength channels "
        f"booked ({channel_pct:.1f}%)"
    )
    render_charts(result)
    render_inspector(result)

    with st.expander("How to read this"):
        st.markdown(
            f"- **{ECMP_NAME}** picks a random shortest path per flow and ignores wavelengths, "
            "temperature and power budget. A flow that needs more wavelengths than are free "
            "lands on channels already in use (a collision) and is still sent.\n"
            f"- **{ENGINE_NAME}** only uses paths with free wavelengths, copper headroom and enough "
            "power budget, and steers around hot switches.\n"
            "- Both use the same heating model and the same latency scoring, evaluated after all "
            "flows are placed. Collisions are counted when a flow is booked.\n"
            "- Thermal and latency constants are model parameters, not measured hardware values."
        )


def main() -> None:
    st.set_page_config(page_title=PAGE_TITLE, layout="wide")
    render_header()

    params, run_clicked = render_sidebar()

    if run_clicked:
        with st.spinner("Running co-simulation..."):
            try:
                st.session_state[STATE_KEY] = run_cosimulation(params)
            except Exception as exc:  # show the failure instead of a blank page
                st.session_state.pop(STATE_KEY, None)
                st.error(f"The co-simulation failed: {exc}")
                st.exception(exc)
                return

    result = st.session_state.get(STATE_KEY)
    if result is None:
        st.info("Choose settings in the sidebar, then click **Run Co-Simulation**.")
        return
    if result.params != params:
        st.info("The sidebar settings changed since this run. Click **Run Co-Simulation** to refresh.")

    render_results(result)


if __name__ == "__main__":
    main()