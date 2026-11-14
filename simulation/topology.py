"""
simulation/topology.py

3-tier Fat-Tree fabric with silicon-photonics hardware constraints, in the style of
Lightmatter Passage and Ayar Labs TeraPHY optical I/O.

Node attributes
---------------
  type            "core" | "aggregation" | "edge" | "gpu"
  temperature_c   current temperature, switches only (starts at 35.0)
  max_temp_c      thermal limit, switches only (85.0)

Edge attributes (every physical link is two opposing directed edges)
--------------------------------------------------------------------
  bandwidth_gbps, base_latency_ns, current_load   legacy fields, unchanged
  media_type               "optical" | "copper"
  insertion_loss_db        optical power lost across the link
  total_wdm_channels       wavelengths available on the link (IDs 0 .. N-1)
  channel_bandwidth_gbps   capacity of one wavelength
  booked_channels          set of wavelength IDs currently occupied

Each directed edge owns its own `booked_channels` set, so the two directions of a
link allocate wavelengths independently.

Hardware profile
----------------
The link hardware can be overridden with keyword arguments (or a config dict through
`FatTreeTopology.from_config_dict`). The defaults reproduce the original fabric exactly:

  core_wdm_channels         wavelengths on each aggregation <-> core link          (16)
  agg_wdm_channels          wavelengths on each edge <-> aggregation link          (8)
  channel_bandwidth_gbps    capacity of one wavelength on the optical links        (50.0)
  core_insertion_loss_db    loss of each aggregation <-> core link                 (0.8)
  agg_insertion_loss_db     loss of each edge <-> aggregation link                 (0.5)
  gpu_bandwidth_gbps        GPU <-> edge copper link, one lane, no WDM             (400.0)

Link bandwidth is always channels x channel bandwidth, so the two cannot disagree.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import networkx as nx

SWITCH_TYPES: frozenset[str] = frozenset({"core", "aggregation", "edge"})

# Sanity limits for hardware profiles. They catch typos (a 5000 dB loss, a negative
# channel count); they are not physical limits of any particular product.
MAX_WDM_CHANNELS = 1024
MAX_CHANNEL_BANDWIDTH_GBPS = 10_000.0
MAX_GPU_BANDWIDTH_GBPS = 100_000.0
MAX_INSERTION_LOSS_DB = 100.0
MAX_PROFILE_NAME_LENGTH = 80


def _int_param(name: str, value: Any, lo: int, hi: int) -> int:
    """Validate an integer parameter. Whole-number floats such as 16.0 are accepted."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a whole number, got {value!r}")
    if isinstance(value, float) and not (math.isfinite(value) and value.is_integer()):
        raise ValueError(f"{name} must be a whole number, got {value!r}")
    value = int(value)
    if not lo <= value <= hi:
        raise ValueError(f"{name} must be between {lo} and {hi}, got {value}")
    return value


def _float_param(name: str, value: Any, lo: float, hi: float, *, allow_zero: bool = False) -> float:
    """Validate a real-valued parameter: finite, >0 (or >=0 when allow_zero), and <= hi."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number, got {value!r}")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    too_low = value < lo if allow_zero else value <= lo
    if too_low or value > hi:
        bound = f">= {lo:g}" if allow_zero else f"> {lo:g}"
        raise ValueError(f"{name} must be {bound} and <= {hi:g}, got {value:g}")
    return value


class FatTreeTopology:
    # Switch thermal envelope
    SWITCH_START_TEMP_C = 35.0
    SWITCH_MAX_TEMP_C = 85.0

    #: Keys accepted by `from_config_dict` (all optional; "name" is a free-text label).
    CONFIG_KEYS: tuple[str, ...] = (
        "name",
        "k",
        "core_wdm_channels",
        "agg_wdm_channels",
        "channel_bandwidth_gbps",
        "core_insertion_loss_db",
        "agg_insertion_loss_db",
        "gpu_bandwidth_gbps",
    )

    def __init__(
        self,
        k: int = 4,
        *,
        core_wdm_channels: int = 16,
        agg_wdm_channels: int = 8,
        channel_bandwidth_gbps: float = 50.0,
        core_insertion_loss_db: float = 0.8,
        agg_insertion_loss_db: float = 0.5,
        gpu_bandwidth_gbps: float = 400.0,
    ):
        """
        Initializes a 3-Tier Fat-Tree Network Topology.
        :param k: The port count of the switches (determines cluster scale).
                  Total GPUs = (k^3) // 4. For k=4, that's 16 simulated GPUs.
                  Scale out to k=48 or higher for massive cluster limits.
        :param core_wdm_channels: Wavelengths per aggregation <-> core optical link.
        :param agg_wdm_channels: Wavelengths per edge <-> aggregation optical link.
        :param channel_bandwidth_gbps: Capacity of one wavelength, in Gbps.
        :param core_insertion_loss_db: Insertion loss of an aggregation <-> core link.
        :param agg_insertion_loss_db: Insertion loss of an edge <-> aggregation link.
        :param gpu_bandwidth_gbps: Capacity of the GPU <-> edge copper link.
        The hardware parameters default to the original fabric, so `FatTreeTopology(k=4)`
        behaves exactly as before. Invalid values raise ValueError.
        """
        if isinstance(k, bool) or not isinstance(k, int) or k < 2 or k % 2:
            raise ValueError(f"k must be an even integer >= 2, got {k!r}")

        self.core_wdm_channels = _int_param("core_wdm_channels", core_wdm_channels, 1, MAX_WDM_CHANNELS)
        self.agg_wdm_channels = _int_param("agg_wdm_channels", agg_wdm_channels, 1, MAX_WDM_CHANNELS)
        self.channel_bandwidth_gbps = _float_param(
            "channel_bandwidth_gbps", channel_bandwidth_gbps, 0.0, MAX_CHANNEL_BANDWIDTH_GBPS
        )
        self.core_insertion_loss_db = _float_param(
            "core_insertion_loss_db", core_insertion_loss_db, 0.0, MAX_INSERTION_LOSS_DB, allow_zero=True
        )
        self.agg_insertion_loss_db = _float_param(
            "agg_insertion_loss_db", agg_insertion_loss_db, 0.0, MAX_INSERTION_LOSS_DB, allow_zero=True
        )
        self.gpu_bandwidth_gbps = _float_param(
            "gpu_bandwidth_gbps", gpu_bandwidth_gbps, 0.0, MAX_GPU_BANDWIDTH_GBPS
        )

        self.k = k
        self.profile_name: str | None = None  # set by from_config_dict
        self.graph = nx.DiGraph()  # Directed Graph representing two-way fabric paths

        # Calculate structure limits
        self.num_pods = k
        self.core_switches = (k // 2) ** 2
        self.switches_per_pod = k // 2  # Half aggregation, half edge
        self.gpus_per_edge_switch = k // 2

        self._build_network()

    @classmethod
    def validate_config(cls, config: Mapping[str, Any]) -> dict[str, Any]:
        """
        Validate a hardware-profile dict without building a fabric.

        Returns a normalized dict that holds only the keys present in `config`: integers as
        int, real values as float, "name" stripped of blanks (and dropped if empty). Unknown
        keys, wrong types and out-of-range values raise ValueError naming the offending key.
        """
        if not isinstance(config, Mapping):
            raise ValueError(f"The hardware profile must be a mapping of settings, got {type(config).__name__}")
        bad_keys = [key for key in config if not isinstance(key, str)]
        if bad_keys:
            raise ValueError(f"Hardware profile keys must be text, got {bad_keys!r}")
        unknown = sorted(set(config) - set(cls.CONFIG_KEYS))
        if unknown:
            raise ValueError(
                f"Unknown hardware profile setting(s): {', '.join(unknown)}. "
                f"Allowed settings: {', '.join(cls.CONFIG_KEYS)}"
            )

        clean: dict[str, Any] = {}
        if "name" in config and config["name"] is not None:
            name = config["name"]
            if not isinstance(name, str):
                raise ValueError(f"name must be text, got {name!r}")
            name = name.strip()
            if len(name) > MAX_PROFILE_NAME_LENGTH:
                raise ValueError(f"name must be at most {MAX_PROFILE_NAME_LENGTH} characters")
            if name:
                clean["name"] = name
        if "k" in config:
            k = _int_param("k", config["k"], 2, 1_000_000)
            if k % 2:
                raise ValueError(f"k must be an even integer >= 2, got {k}")
            clean["k"] = k
        for key, lo, hi in (("core_wdm_channels", 1, MAX_WDM_CHANNELS), ("agg_wdm_channels", 1, MAX_WDM_CHANNELS)):
            if key in config:
                clean[key] = _int_param(key, config[key], lo, hi)
        if "channel_bandwidth_gbps" in config:
            clean["channel_bandwidth_gbps"] = _float_param(
                "channel_bandwidth_gbps", config["channel_bandwidth_gbps"], 0.0, MAX_CHANNEL_BANDWIDTH_GBPS
            )
        for key in ("core_insertion_loss_db", "agg_insertion_loss_db"):
            if key in config:
                clean[key] = _float_param(key, config[key], 0.0, MAX_INSERTION_LOSS_DB, allow_zero=True)
        if "gpu_bandwidth_gbps" in config:
            clean["gpu_bandwidth_gbps"] = _float_param(
                "gpu_bandwidth_gbps", config["gpu_bandwidth_gbps"], 0.0, MAX_GPU_BANDWIDTH_GBPS
            )
        return clean

    @classmethod
    def from_config_dict(cls, config: Mapping[str, Any]) -> FatTreeTopology:
        """
        Build a topology from a hardware-profile dict (for example parsed from YAML/JSON).

        Every key is optional and falls back to the default hardware (k defaults to 4).
        Accepted keys are listed in `CONFIG_KEYS`. Unknown keys, wrong types and out-of-range
        values raise ValueError with a message that names the offending key.
        """
        clean = cls.validate_config(config)
        name = clean.pop("name", None)
        k = clean.pop("k", 4)
        topology = cls(k=k, **clean)
        topology.profile_name = name
        return topology

    @property
    def hardware_profile(self) -> dict[str, Any]:
        """The effective hardware settings of this fabric (a copy, safe to modify)."""
        return {
            "k": self.k,
            "core_wdm_channels": self.core_wdm_channels,
            "agg_wdm_channels": self.agg_wdm_channels,
            "channel_bandwidth_gbps": self.channel_bandwidth_gbps,
            "core_insertion_loss_db": self.core_insertion_loss_db,
            "agg_insertion_loss_db": self.agg_insertion_loss_db,
            "gpu_bandwidth_gbps": self.gpu_bandwidth_gbps,
        }

    # ------------------------------------------------------------------ #
    # Construction helpers
    # ------------------------------------------------------------------ #
    def _add_switches(self, names: list[str], switch_type: str) -> None:
        self.graph.add_nodes_from(
            names,
            type=switch_type,
            temperature_c=self.SWITCH_START_TEMP_C,
            max_temp_c=self.SWITCH_MAX_TEMP_C,
        )

    def _add_link(
        self,
        u: str,
        v: str,
        *,
        bandwidth_gbps: float,
        base_latency_ns: float,
        media_type: str,
        insertion_loss_db: float,
        total_wdm_channels: int,
        channel_bandwidth_gbps: float,
    ) -> None:
        """Add one physical link as two directed edges with independent state."""
        if not math.isclose(total_wdm_channels * channel_bandwidth_gbps, bandwidth_gbps):
            raise ValueError(
                f"Link {u}<->{v}: {total_wdm_channels} x {channel_bandwidth_gbps} Gbps "
                f"channels do not add up to {bandwidth_gbps} Gbps"
            )
        for src, dst in ((u, v), (v, u)):
            self.graph.add_edge(
                src,
                dst,
                # Legacy attributes (kept for backward compatibility)
                bandwidth_gbps=bandwidth_gbps,
                base_latency_ns=base_latency_ns,
                current_load=0.0,
                # Optical physics attributes
                media_type=media_type,
                insertion_loss_db=insertion_loss_db,
                total_wdm_channels=total_wdm_channels,
                channel_bandwidth_gbps=channel_bandwidth_gbps,
                booked_channels=set(),  # fresh set per directed edge, never shared
            )

    def _build_network(self):
        # 1. Create Core Switches
        core_nodes = [f"core_{i}" for i in range(self.core_switches)]
        self._add_switches(core_nodes, "core")

        # 2. Build Pods (Aggregation Tier, Edge Tier, and GPUs)
        for pod in range(self.num_pods):
            agg_nodes = [f"pod_{pod}_agg_{i}" for i in range(self.switches_per_pod)]
            edge_nodes = [f"pod_{pod}_edge_{i}" for i in range(self.switches_per_pod)]

            self._add_switches(agg_nodes, "aggregation")
            self._add_switches(edge_nodes, "edge")

            # Link Aggregation tier to Core tier
            for agg_idx, agg_node in enumerate(agg_nodes):
                core_offset = agg_idx * (self.k // 2)
                for c in range(self.k // 2):
                    core_node = core_nodes[core_offset + c]
                    # Long-reach optical fiber (default: 16 wavelengths x 50 Gbps = 800 Gbps)
                    self._add_link(
                        agg_node,
                        core_node,
                        bandwidth_gbps=self.core_wdm_channels * self.channel_bandwidth_gbps,
                        base_latency_ns=50.0,
                        media_type="optical",
                        insertion_loss_db=self.core_insertion_loss_db,
                        total_wdm_channels=self.core_wdm_channels,
                        channel_bandwidth_gbps=self.channel_bandwidth_gbps,
                    )

            # Link Edge tier to Aggregation tier within the same Pod
            for agg_node in agg_nodes:
                for edge_node in edge_nodes:
                    # Short-reach optical (default: 8 wavelengths x 50 Gbps = 400 Gbps)
                    self._add_link(
                        edge_node,
                        agg_node,
                        bandwidth_gbps=self.agg_wdm_channels * self.channel_bandwidth_gbps,
                        base_latency_ns=100.0,
                        media_type="optical",
                        insertion_loss_db=self.agg_insertion_loss_db,
                        total_wdm_channels=self.agg_wdm_channels,
                        channel_bandwidth_gbps=self.channel_bandwidth_gbps,
                    )

            # 3. Create and Link GPUs to Edge Switches
            for edge_idx, edge_node in enumerate(edge_nodes):
                for g in range(self.gpus_per_edge_switch):
                    gpu_id = f"pod_{pod}_gpu_{edge_idx * self.gpus_per_edge_switch + g}"
                    self.graph.add_node(gpu_id, type="gpu")
                    # Electrical NVLink/PCIe trace: single lane, no WDM (default 400 Gbps)
                    self._add_link(
                        gpu_id,
                        edge_node,
                        bandwidth_gbps=self.gpu_bandwidth_gbps,
                        base_latency_ns=20.0,
                        media_type="copper",
                        insertion_loss_db=0.0,
                        total_wdm_channels=1,
                        channel_bandwidth_gbps=self.gpu_bandwidth_gbps,
                    )

        print(f"Successfully constructed a Fat-Tree cluster fabric with {len(self.graph.nodes)} interconnected nodes.")

    # ------------------------------------------------------------------ #
    # Wavelength helpers
    # ------------------------------------------------------------------ #
    def get_available_wavelengths(self, u: str, v: str) -> list[int]:
        """Return the sorted free wavelength (channel) IDs on directed edge (u, v)."""
        if not self.graph.has_edge(u, v):
            raise KeyError(f"No edge from {u!r} to {v!r} in the topology")
        attrs = self.graph.edges[u, v]
        return sorted(set(range(attrs["total_wdm_channels"])) - attrs["booked_channels"])


if __name__ == "__main__":
    topo = FatTreeTopology(k=4)
    graph = topo.graph

    def show_link(label: str, u: str, v: str) -> None:
        print(f"\n{label}: {u} -> {v}")
        for key, value in graph.edges[u, v].items():
            print(f"  {key:<24} {value}")
        free = topo.get_available_wavelengths(u, v)
        print(f"  {'available_wavelengths':<24} {free}  ({len(free)} free)")

    # Sample switch node
    print("\nSample switch node: core_0")
    for key, value in graph.nodes["core_0"].items():
        print(f"  {key:<24} {value}")

    show_link("Agg <-> Core optical link", "pod_0_agg_0", "core_0")
    show_link("Edge <-> Agg optical link", "pod_0_edge_0", "pod_0_agg_0")
    show_link("GPU <-> Edge copper link", "pod_0_gpu_0", "pod_0_edge_0")

    # Book a few wavelengths on one direction and confirm the helper tracks it,
    # and that the reverse direction is unaffected.
    u, v = "pod_0_agg_0", "core_0"
    graph.edges[u, v]["booked_channels"].update({0, 3, 7})
    print(f"\nAfter booking channels {{0, 3, 7}} on {u} -> {v}:")
    print(f"  forward free: {topo.get_available_wavelengths(u, v)}")
    print(f"  reverse free: {topo.get_available_wavelengths(v, u)}  (independent)")

    # Custom hardware profile: 32 wavelengths per core link, 800 Gbps GPU links
    custom = FatTreeTopology.from_config_dict(
        {"name": "demo", "k": 6, "core_wdm_channels": 32, "gpu_bandwidth_gbps": 800}
    )
    core_link = custom.graph.edges["pod_0_agg_0", "core_0"]
    gpu_link = custom.graph.edges["pod_0_gpu_0", "pod_0_edge_0"]
    print(f"\nCustom profile {custom.profile_name!r}: {custom.hardware_profile}")
    print(f"  agg->core: {core_link['total_wdm_channels']} ch x {core_link['channel_bandwidth_gbps']} Gbps "
          f"= {core_link['bandwidth_gbps']} Gbps; GPU link {gpu_link['bandwidth_gbps']} Gbps")

    for bad in ({"k": 5}, {"core_wdm_channels": 0}, {"gpu_bandwidth_gbps": "fast"}, {"bogus": 1}):
        try:
            FatTreeTopology.from_config_dict(bad)
        except ValueError as exc:
            print(f"  rejected {bad}: {exc}")