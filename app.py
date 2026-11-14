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
import csv
import importlib.util
import inspect
import io
import json
import math
import random
import statistics
import sys
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:  # PyYAML is optional: without it, flat key: value YAML files still load.
    import yaml
except ImportError:  # pragma: no cover - depends on the environment
    yaml = None

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
        "link_latency_ns", "mean_metrics", "run_engine", "min_unavoidable_failures",
        "build_moe_all_to_all", "MOE_TOP_K", "MOE_SEED",
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
MOE_WORKLOAD = "MoE All-to-All"
WORKLOAD_ORDER: tuple[str, ...] = ("Cross-Pod Shift", "Ring All-Reduce", MOE_WORKLOAD)
TOP_K_RANGE: tuple[int, int] = (1, 4)
DEFAULT_TOP_K: int = min(max(int(bench.MOE_TOP_K), TOP_K_RANGE[0]), TOP_K_RANGE[1])
DEFAULT_MOE_SEED: int = int(bench.MOE_SEED)
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

# Hardware profile ---------------------------------------------------------- #
_HARDWARE_DEFAULTS: dict[str, Any] = {
    name: inspect.signature(FatTreeTopology.__init__).parameters[name].default
    for name in (
        "core_wdm_channels",
        "agg_wdm_channels",
        "channel_bandwidth_gbps",
        "core_insertion_loss_db",
        "agg_insertion_loss_db",
        "gpu_bandwidth_gbps",
    )
}
MAX_PROFILE_K = 12  # dashboard limit so a run stays interactive (k=12 is 432 GPUs)
MAX_PROFILE_BYTES = 64 * 1024
PROFILE_EXTENSIONS: tuple[str, ...] = ("yaml", "yml", "json")
SAMPLE_PROFILE_FILENAME = "opticroute_hardware_spec.yaml"


@dataclass(frozen=True, slots=True)
class HardwareProfile:
    """Link hardware for the fabric. The defaults are the original reference fabric."""

    name: str | None = None
    k: int | None = None  # None: use the sidebar's cluster scale
    core_wdm_channels: int = _HARDWARE_DEFAULTS["core_wdm_channels"]
    agg_wdm_channels: int = _HARDWARE_DEFAULTS["agg_wdm_channels"]
    channel_bandwidth_gbps: float = _HARDWARE_DEFAULTS["channel_bandwidth_gbps"]
    core_insertion_loss_db: float = _HARDWARE_DEFAULTS["core_insertion_loss_db"]
    agg_insertion_loss_db: float = _HARDWARE_DEFAULTS["agg_insertion_loss_db"]
    gpu_bandwidth_gbps: float = _HARDWARE_DEFAULTS["gpu_bandwidth_gbps"]

    def topology_kwargs(self) -> dict[str, Any]:
        return {key: getattr(self, key) for key in _HARDWARE_DEFAULTS}

    def label(self, k: int) -> str:
        """Short text for captions and the CSV report."""
        text = self.summary(f"k={k}")
        return f"{self.name}: {text}" if self.name else text

    def summary(self, k_text: str) -> str:
        if self.core_wdm_channels == self.agg_wdm_channels:
            channels = f"{self.core_wdm_channels} WDM ch"
        else:
            channels = f"{self.core_wdm_channels} core / {self.agg_wdm_channels} agg WDM ch"
        return (
            f"{k_text} | {channels} @ {self.channel_bandwidth_gbps:g} Gbps "
            f"| {self.gpu_bandwidth_gbps:g} Gbps GPU link"
        )


DEFAULT_HARDWARE = HardwareProfile()

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
    top_k_experts: int = DEFAULT_TOP_K  # used by MoE All-to-All only
    moe_seed: int = DEFAULT_MOE_SEED  # used by MoE All-to-All only
    hardware: HardwareProfile = DEFAULT_HARDWARE  # k above is the effective cluster scale


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
    unavoidable_failed_flows: int  # lower bound on failures from GPU access-link limits, any router
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


# --------------------------------------------------------------------------- #
# Hardware profiles (YAML / JSON)
# --------------------------------------------------------------------------- #
def sample_profile_yaml() -> str:
    """A clean YAML template with the reference hardware, ready to edit."""
    d = _HARDWARE_DEFAULTS
    return (
        "# OpticRoute Fabric - hardware profile\n"
        "# Every setting is optional; a missing setting uses the reference value shown here.\n"
        "# If k is present it sets the cluster scale (an even number, 2 to "
        f"{MAX_PROFILE_K} in this dashboard).\n"
        "name: Reference profile\n"
        "k: 4\n"
        "\n"
        "# Wavelengths per optical link, and the capacity of one wavelength\n"
        f"core_wdm_channels: {d['core_wdm_channels']}   # aggregation <-> core links\n"
        f"agg_wdm_channels: {d['agg_wdm_channels']}    # edge <-> aggregation links\n"
        f"channel_bandwidth_gbps: {d['channel_bandwidth_gbps']:g}\n"
        "\n"
        "# Insertion loss of each optical link, in dB\n"
        f"core_insertion_loss_db: {d['core_insertion_loss_db']:g}\n"
        f"agg_insertion_loss_db: {d['agg_insertion_loss_db']:g}\n"
        "\n"
        "# GPU <-> edge link: a single lane, no WDM\n"
        f"gpu_bandwidth_gbps: {d['gpu_bandwidth_gbps']:g}\n"
    )


_SCALAR_INT = re.compile(r"[+-]?\d+")
_SCALAR_FLOAT = re.compile(r"[+-]?(\d+\.\d*|\.\d+|\d+)([eE][+-]?\d+)?")


def _flat_yaml_scalar(text: str) -> Any:
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        return text[1:-1]
    lowered = text.lower()
    if lowered in ("", "null", "~"):
        return None
    if lowered in ("true", "yes", "on"):
        return True
    if lowered in ("false", "no", "off"):
        return False
    if _SCALAR_INT.fullmatch(text):
        return int(text)
    if _SCALAR_FLOAT.fullmatch(text):
        return float(text)
    return text


def _parse_flat_yaml(text: str) -> dict[str, Any]:
    """Minimal parser for flat `key: value` YAML, used only when PyYAML is not installed."""
    result: dict[str, Any] = {}
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#") or line in ("---", "..."):
            continue
        if raw.startswith((" ", "\t")) or line.startswith("- ") or ":" not in line:
            raise ValueError(
                f"Line {number}: only flat 'key: value' lines are supported without PyYAML "
                "(install it with: pip install pyyaml)"
            )
        key, _, value = line.partition(":")
        value = value.strip()
        if value.startswith(('"', "'")):
            quote = value[0]
            end = value.find(quote, 1)
            if end == -1:
                raise ValueError(f"Line {number}: unterminated quoted value")
            value = value[: end + 1]
        else:
            value = re.split(r"\s+#", value, maxsplit=1)[0].strip()
        key = key.strip()
        if key in result:
            raise ValueError(f"Line {number}: duplicate setting {key!r}")
        result[key] = _flat_yaml_scalar(value)
    return result


def parse_hardware_profile(filename: str, raw: bytes) -> HardwareProfile:
    """
    Parse and validate an uploaded profile. Raises ValueError with a message meant for
    the user. Validation is FatTreeTopology.validate_config, so the dashboard and the
    topology always agree on what a valid profile is.
    """
    if len(raw) > MAX_PROFILE_BYTES:
        raise ValueError(f"The file is larger than {MAX_PROFILE_BYTES // 1024} KB")
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("The file is not UTF-8 text") from exc

    extension = Path(filename).suffix.lower().lstrip(".")
    if extension not in PROFILE_EXTENSIONS:
        raise ValueError("Use a .yaml, .yml or .json file")
    try:
        if extension == "json":
            data = json.loads(text)
        elif yaml is not None:
            data = yaml.safe_load(text)
        else:
            data = _parse_flat_yaml(text)
    except ValueError:  # includes json.JSONDecodeError
        raise
    except Exception as exc:  # yaml.YAMLError and friends
        mark = getattr(exc, "problem_mark", None)
        problem = getattr(exc, "problem", None) or (str(exc).splitlines() or ["unreadable"])[0]
        where = f" (line {mark.line + 1})" if mark is not None else ""
        raise ValueError(f"Could not parse the file{where}: {problem}") from exc

    if data is None:
        raise ValueError("The file is empty")
    if not isinstance(data, Mapping):
        raise ValueError("The file must hold a mapping of settings (key: value pairs)")

    clean = FatTreeTopology.validate_config(data)
    k = clean.get("k")
    if k is not None and k > MAX_PROFILE_K:
        raise ValueError(f"k must be at most {MAX_PROFILE_K} in this dashboard, got {k}")
    return HardwareProfile(
        name=clean.get("name"),
        k=k,
        **{key: clean[key] for key in _HARDWARE_DEFAULTS if key in clean},
    )


# --------------------------------------------------------------------------- #
# Engineering report (CSV)
# --------------------------------------------------------------------------- #
REPORT_COLUMNS = [
    "Workload", "Cluster k", "Tensor size (GB)", "Power budget (dB)", "Hardware profile",
    "Status", "Flow", "Source", "Destination", "Hops", "Path", "Insertion loss (dB)",
    "Peak switch temp (°C)", "WDM channels per optical link", "Block reason",
]  # fmt: skip


def _csv_safe(value: Any) -> Any:
    """Stop spreadsheets from running text that starts with a formula character."""
    if isinstance(value, str) and value[:1] in ("=", "+", "-", "@", "\t", "\r"):
        return "'" + value
    return value


def build_report_csv(result: CoSimResult) -> bytes:
    """
    One row per flow from the engine run: routed path, insertion loss, peak switch
    temperature and WDM channel allocations, or the reason the flow was blocked.
    UTF-8 with a BOM so Excel shows the degree sign correctly.
    """
    params = result.params
    context = [
        params.workload,
        params.k,
        params.tensor_gb,
        params.power_budget_db,
        params.hardware.label(params.k),
    ]
    rows: list[tuple[int, list[Any]]] = []
    for record in result.flow_table.to_dict("records"):
        rows.append(
            (
                int(record["Flow"]),
                [
                    "Routed",
                    record["Flow"],
                    record["Source"],
                    record["Destination"],
                    record["Hops"],
                    record["Path"],
                    record["Insertion loss (dB)"],
                    record["Peak switch temp (°C)"],
                    record["WDM channels per optical link"],
                    "",
                ],
            )
        )
    for record in result.blocked_table.to_dict("records"):
        rows.append(
            (
                int(record["Flow"]),
                ["Blocked", record["Flow"], record["Source"], record["Destination"], "", "", "", "", "", record["Reason"]],
            )
        )
    rows.sort(key=lambda item: item[0])

    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer)
    writer.writerow(REPORT_COLUMNS)
    for _, values in rows:
        writer.writerow([_csv_safe(value) for value in context + values])
    return buffer.getvalue().encode("utf-8-sig")


def report_filename(params: SimParams) -> str:
    workload = re.sub(r"[^a-z0-9]+", "_", params.workload.lower()).strip("_")
    return f"opticroute_diagnostic_report_k{params.k}_{workload}_{params.tensor_gb}gb.csv"


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
        topology = FatTreeTopology(k=params.k, **params.hardware.topology_kwargs())
    graph = topology.graph
    engine = OpticalRoutingEngine(
        topology,
        max_insertion_loss_db=params.power_budget_db,
        heating_c_per_gbps=params.heating_c_per_gbps,
    )
    physics = bench.PhysicsModel.from_engine(engine)
    generator = AIWorkloadGenerator(topology)
    if params.workload == MOE_WORKLOAD:
        flows = bench.build_moe_all_to_all(
            generator,
            float(params.tensor_gb),
            top_k_experts=params.top_k_experts,
            seed=params.moe_seed,
        )
    else:
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
        unavoidable_failed_flows=bench.min_unavoidable_failures(graph, flows, physics.sync_window_s),
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


def render_hardware_section() -> tuple[HardwareProfile, bool]:
    """
    Sidebar section for the custom hardware profile. Returns the profile to simulate and
    whether the Run button may be used (False while an uploaded file is invalid).
    """
    st.subheader("Custom Hardware Spec (YAML / JSON)")
    st.download_button(
        "Download Sample Hardware YAML",
        data=sample_profile_yaml(),
        file_name=SAMPLE_PROFILE_FILENAME,
        mime="text/yaml",
        key="sample_profile_download",
        help="A template with the reference hardware. Edit it and upload it below.",
        **_stretch(st.download_button),
    )
    uploaded = st.file_uploader(
        "Upload hardware profile",
        type=list(PROFILE_EXTENSIONS),
        key="hardware_profile_upload",
        help=f"A .yaml, .yml or .json file of up to {MAX_PROFILE_BYTES // 1024} KB. "
        "Every setting is optional; missing ones use the reference hardware.",
    )
    if uploaded is None:
        st.caption("No file uploaded: simulating the reference hardware.")
        return DEFAULT_HARDWARE, True

    try:
        profile = parse_hardware_profile(uploaded.name, uploaded.getvalue())
    except ValueError as exc:
        st.error(f"Invalid hardware profile ({uploaded.name}): {exc}")
        st.caption("Fix the file and upload it again. Run Co-Simulation is disabled until then.")
        return DEFAULT_HARDWARE, False

    k_text = f"k={profile.k}" if profile.k is not None else "k from sidebar"
    title = f" ({profile.name})" if profile.name else ""
    st.success(f"Custom Profile Active{title}: {profile.summary(k_text)}")
    st.caption(
        f"Insertion loss: {profile.core_insertion_loss_db:g} dB per core link, "
        f"{profile.agg_insertion_loss_db:g} dB per aggregation link."
    )
    return profile, True


def render_sidebar() -> tuple[SimParams, bool]:
    with st.sidebar:
        st.header("Simulation controls")
        scale_slot = st.container()  # filled in below, once the hardware profile is known
        workload = st.selectbox("Workload pattern", WORKLOAD_ORDER)
        top_k = DEFAULT_TOP_K
        if workload == MOE_WORKLOAD:
            top_k = st.slider(
                "Top-K experts",
                min_value=TOP_K_RANGE[0],
                max_value=TOP_K_RANGE[1],
                value=DEFAULT_TOP_K,
                step=1,
                help="Experts each GPU dispatches its tokens to. Destinations are skewed toward "
                "a few popular experts, which creates hot spots.",
            )
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
        hardware, profile_ok = render_hardware_section()
        with scale_slot:
            if hardware.k is not None:
                k = st.selectbox(
                    "Cluster scale",
                    (hardware.k,),
                    format_func=lambda value: f"k = {value}  ({value**3 // 4} GPUs)",
                    disabled=True,
                    help="Set by the uploaded hardware profile.",
                )
            else:
                k = st.selectbox(
                    "Cluster scale",
                    CLUSTER_SCALES,
                    format_func=lambda value: f"k = {value}  ({value**3 // 4} GPUs)",
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
            if workload == MOE_WORKLOAD:
                moe_seed = st.number_input(
                    "MoE dispatch seed",
                    min_value=0,
                    value=DEFAULT_MOE_SEED,
                    step=1,
                    help="Seeds which experts are popular and which GPUs get picked. "
                    "The same seed always gives the same traffic.",
                )
            else:
                moe_seed = DEFAULT_MOE_SEED
        run_clicked = st.button(
            "Run Co-Simulation", type="primary", disabled=not profile_ok, **_stretch(st.button)
        )
        st.caption("Both routers receive identical flows in identical order.")

    params = SimParams(
        k=int(k),
        workload=str(workload),
        tensor_gb=int(tensor_gb),
        power_budget_db=round(float(power_budget_db), 1),
        heating_c_per_gbps=float(heating),
        ecmp_trials=int(trials),
        seed=int(seed),
        top_k_experts=int(top_k),
        moe_seed=int(moe_seed),
        hardware=hardware,
    )
    return params, bool(run_clicked) and profile_ok


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


def render_floor(result: CoSimResult) -> None:
    """Show the unavoidable GPU port-limit failures and how far each router sits above it."""
    floor = result.unavoidable_failed_flows
    engine_extra = result.engine.failed_flows - floor
    ecmp_extra = result.ecmp.failed_flows - floor
    trials = result.params.ecmp_trials
    st.markdown(
        f"**Unavoidable GPU port-limit failures: {_fmt_num(floor)}** of {result.total_flows} flows "
        "(lower bound for any router)"
    )
    st.caption(
        f"Each GPU has one {result.params.hardware.gpu_bandwidth_gbps:g} Gbps access link, so flows "
        "beyond what a source or destination GPU can send or receive in the sync window fail no "
        "matter how they are routed. "
        f"Avoidable fabric failures above this floor: {ENGINE_NAME} {_fmt_num(max(engine_extra, 0.0))}, "
        f"{ECMP_NAME} {_fmt_num(max(ecmp_extra, 0.0))} (mean of {trials} trials)."
    )
    if abs(engine_extra) <= _EPSILON:
        if floor == 0:
            st.success(f"{ENGINE_NAME} blocked no flows: zero avoidable fabric failures.")
        else:
            st.success(
                f"{ENGINE_NAME} sits exactly on the floor: every flow it blocked was unavoidable, "
                "so it caused zero avoidable fabric failures."
            )
    elif engine_extra > 0:
        st.warning(
            f"{ENGINE_NAME} blocked {_fmt_num(engine_extra)} flows above the floor "
            "(no feasible path: wavelengths, copper headroom, power budget or thermal limit)."
        )
    if floor > 0:
        st.caption("With a non-zero floor, compare routers by failures above it, not by latency.")


def render_notices(result: CoSimResult) -> None:
    engine = result.engine
    if engine.failed_flows > 0:
        st.warning(
            f"The engine blocked {_fmt_num(engine.failed_flows)} of {result.total_flows} flows. Its "
            "latency and temperature figures cover the routed flows only, and blocked traffic "
            "generates no heat. Compare these results by failed-flow counts first."
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
    params = result.params
    st.download_button(
        "Download Engineering Diagnostic Report (CSV)",
        data=build_report_csv(result),
        file_name=report_filename(params),
        mime="text/csv",
        key="diagnostic_report_download",
        help="One row per flow from the run shown below: routed path, insertion loss, peak "
        "switch temperature and WDM channel allocations, or the reason a flow was blocked.",
    )
    st.caption(
        f"The report covers the run shown here: {len(result.flow_table):,} routed and "
        f"{len(result.blocked_table):,} blocked flows."
    )
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
    hardware_text = (
        f"custom hardware ({params.hardware.label(params.k)})"
        if params.hardware != DEFAULT_HARDWARE
        else "reference hardware"
    )
    workload_text = (
        f"{params.workload} (top-{params.top_k_experts} experts)"
        if params.workload == MOE_WORKLOAD
        else params.workload
    )
    st.caption(
        f"k = {params.k} ({result.gpu_count} GPUs) · {workload_text} · {params.tensor_gb} GB "
        f"· {result.total_flows} flows · power budget {params.power_budget_db:.1f} dB "
        f"· {hardware_text} · ran in {result.elapsed_s:.1f} s"
    )

    render_kpis(result)
    render_floor(result)
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
            "- **MoE All-to-All** sends each GPU's tokens to its top-K experts in other pods, "
            "skewed toward popular experts. Hot experts receive far more than their access link "
            "can take, so some failures are unavoidable for any router.\n"
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