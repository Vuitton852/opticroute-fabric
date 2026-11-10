"""
workloads/traffic_generator.py

Synthetic AI workload generation for the Silicon Photonics routing fabric.

Simulates the communication bursts of distributed LLM training across a
Fat-Tree topology. The first supported pattern is a ring All-Reduce sync phase.
"""

from __future__ import annotations

import re
import sys
from dataclasses import asdict, dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any

import networkx as nx

# Allow running this file directly (python workloads/traffic_generator.py)
# while still importing the sibling `simulation` package.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from simulation.topology import FatTreeTopology  # noqa: E402

_GPU_NAME_PATTERN = re.compile(r"^pod_(\d+)_gpu_(\d+)$")
_BITS_PER_BYTE = 8


class FlowPriority(IntEnum):
    """Traffic priority classes. Higher value means more urgent."""

    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass(frozen=True, slots=True)
class Flow:
    """A single point-to-point communication demand between two GPUs."""

    source_gpu: str
    destination_gpu: str
    traffic_volume_gbits: float
    priority: FlowPriority = FlowPriority.HIGH

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["priority"] = self.priority.name
        return data


class AIWorkloadGenerator:
    """Generates distributed-training communication flows over a Fat-Tree."""

    def __init__(self, topology: FatTreeTopology) -> None:
        if not hasattr(topology, "graph") or not isinstance(topology.graph, nx.DiGraph):
            raise TypeError("topology must expose a networkx DiGraph as `.graph`")
        self.topology = topology
        self.active_flows: list[Flow] = []

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _gpu_sort_key(name: str) -> tuple[int, int]:
        match = _GPU_NAME_PATTERN.match(name)
        if match is None:
            raise ValueError(f"Unexpected GPU node name: {name!r}")
        return int(match.group(1)), int(match.group(2))

    @staticmethod
    def pod_of(gpu: str) -> int:
        """Return the pod index of a GPU node name."""
        return AIWorkloadGenerator._gpu_sort_key(gpu)[0]

    def get_gpu_nodes(self) -> list[str]:
        """
        Return all GPU nodes in topology-aware order (by pod, then local index).

        This ordering places ring neighbors under the same edge switch whenever
        possible, so only a small number of ring hops cross pod boundaries.
        """
        gpus = [
            node
            for node, attrs in self.topology.graph.nodes(data=True)
            if attrs.get("type") == "gpu"
        ]
        return sorted(gpus, key=self._gpu_sort_key)

    # ------------------------------------------------------------------ #
    # Workload patterns
    # ------------------------------------------------------------------ #
    def generate_all_reduce_traffic(
        self,
        tensor_size_gb: float,
        priority: FlowPriority = FlowPriority.HIGH,
    ) -> list[Flow]:
        """
        Build the flows for one ring All-Reduce over every GPU in the cluster.

        Ring All-Reduce on N GPUs runs a reduce-scatter followed by an
        all-gather, each taking N-1 steps with chunks of size S/N. Each GPU
        therefore sends 2 * (N-1)/N * S in total to its ring successor.

        :param tensor_size_gb: Size of the tensor being synchronized, in gigabytes.
        :param priority: Priority class assigned to every generated flow.
        :return: One Flow per ring edge (GPU i -> GPU i+1, wrapping around).
        """
        if tensor_size_gb <= 0:
            raise ValueError("tensor_size_gb must be positive")

        gpus = self.get_gpu_nodes()
        n = len(gpus)
        if n < 2:
            raise ValueError(f"All-Reduce needs at least 2 GPUs, found {n}")

        per_link_gb = 2.0 * (n - 1) / n * tensor_size_gb
        per_link_gbits = per_link_gb * _BITS_PER_BYTE

        self.active_flows = [
            Flow(
                source_gpu=gpus[i],
                destination_gpu=gpus[(i + 1) % n],
                traffic_volume_gbits=per_link_gbits,
                priority=priority,
            )
            for i in range(n)
        ]
        return list(self.active_flows)


if __name__ == "__main__":
    topo = FatTreeTopology(k=4)
    generator = AIWorkloadGenerator(topo)

    tensor_size_gb = 10.0
    flows = generator.generate_all_reduce_traffic(tensor_size_gb)

    print(f"\nGenerated {len(flows)} All-Reduce flows for a {tensor_size_gb} GB tensor:\n")
    for flow in flows:
        print(
            f"  {flow.source_gpu:>14} -> {flow.destination_gpu:<14} "
            f"{flow.traffic_volume_gbits:8.2f} Gbit  [{flow.priority.name}]"
        )

    # Sanity checks against the topology.
    graph = topo.graph
    for flow in flows:
        assert flow.source_gpu in graph and flow.destination_gpu in graph, flow
        assert nx.has_path(graph, flow.source_gpu, flow.destination_gpu), flow

    cross_pod = sum(
        1 for f in flows if generator.pod_of(f.source_gpu) != generator.pod_of(f.destination_gpu)
    )
    total_gbits = sum(f.traffic_volume_gbits for f in flows)
    print(f"\nTotal volume:       {total_gbits:.2f} Gbit")
    print(f"Cross-pod flows:    {cross_pod} of {len(flows)}")
    print("All flows validated: endpoints exist and are reachable.")