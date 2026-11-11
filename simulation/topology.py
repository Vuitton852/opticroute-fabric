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
"""

from __future__ import annotations

import math

import networkx as nx

SWITCH_TYPES: frozenset[str] = frozenset({"core", "aggregation", "edge"})


class FatTreeTopology:
    # Switch thermal envelope
    SWITCH_START_TEMP_C = 35.0
    SWITCH_MAX_TEMP_C = 85.0

    def __init__(self, k: int = 4):
        """
        Initializes a 3-Tier Fat-Tree Network Topology.
        :param k: The port count of the switches (determines cluster scale).
                  Total GPUs = (k^3) // 4. For k=4, that's 16 simulated GPUs.
                  Scale out to k=48 or higher for massive cluster limits.
        """
        if k < 2 or k % 2:
            raise ValueError(f"k must be an even integer >= 2, got {k}")

        self.k = k
        self.graph = nx.DiGraph()  # Directed Graph representing two-way fabric paths

        # Calculate structure limits
        self.num_pods = k
        self.core_switches = (k // 2) ** 2
        self.switches_per_pod = k // 2  # Half aggregation, half edge
        self.gpus_per_edge_switch = k // 2

        self._build_network()

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
                    # Long-reach optical fiber: 16 wavelengths x 50 Gbps = 800 Gbps
                    self._add_link(
                        agg_node,
                        core_node,
                        bandwidth_gbps=800.0,
                        base_latency_ns=50.0,
                        media_type="optical",
                        insertion_loss_db=0.8,
                        total_wdm_channels=16,
                        channel_bandwidth_gbps=50.0,
                    )

            # Link Edge tier to Aggregation tier within the same Pod
            for agg_node in agg_nodes:
                for edge_node in edge_nodes:
                    # Short-reach optical: 8 wavelengths x 50 Gbps = 400 Gbps
                    self._add_link(
                        edge_node,
                        agg_node,
                        bandwidth_gbps=400.0,
                        base_latency_ns=100.0,
                        media_type="optical",
                        insertion_loss_db=0.5,
                        total_wdm_channels=8,
                        channel_bandwidth_gbps=50.0,
                    )

            # 3. Create and Link GPUs to Edge Switches
            for edge_idx, edge_node in enumerate(edge_nodes):
                for g in range(self.gpus_per_edge_switch):
                    gpu_id = f"pod_{pod}_gpu_{edge_idx * self.gpus_per_edge_switch + g}"
                    self.graph.add_node(gpu_id, type="gpu")
                    # Electrical NVLink/PCIe trace: single 400 Gbps lane, no WDM
                    self._add_link(
                        gpu_id,
                        edge_node,
                        bandwidth_gbps=400.0,
                        base_latency_ns=20.0,
                        media_type="copper",
                        insertion_loss_db=0.0,
                        total_wdm_channels=1,
                        channel_bandwidth_gbps=400.0,
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