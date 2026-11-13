"""
workloads/traffic_generator.py

Synthetic AI workload generation for the Silicon Photonics routing fabric.

Simulates the communication bursts of distributed LLM training across a
Fat-Tree topology. Supported patterns:

  * Ring All-Reduce sync phase (data-parallel gradient synchronization).
  * Mixture-of-Experts (MoE) All-to-All token dispatch (expert-parallel routing).
"""

from __future__ import annotations

import math
import random
import re
import sys
from collections import Counter
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
_MIN_POPULARITY = 1e-300  # keeps every expert samplable even at extreme skew


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

    def generate_moe_all_to_all_traffic(
        self,
        tensor_size_gb: float,
        top_k_experts: int = 2,
        seed: int = 42,
        *,
        skew: float = 1.0,
        priority: FlowPriority = FlowPriority.HIGH,
    ) -> list[Flow]:
        """
        Build the flows for one Mixture-of-Experts token-dispatch burst.

        Every GPU holds `tensor_size_gb` of token activations and dispatches them to
        `top_k_experts` distinct expert GPUs in *other* pods. Expert popularity is
        skewed (Zipf-like with exponent `skew`), so a few "hot" experts receive
        converging traffic from many senders while most experts stay quiet. Each
        source splits its tensor across its experts by randomly drawn gate weights.
        Only the dispatch phase is generated; the combine phase mirrors it.

        Fully deterministic for a given (topology, tensor size, top_k, seed, skew), and
        independent of Python's global random state.

        :param tensor_size_gb: Token payload each GPU dispatches in total, in gigabytes.
        :param top_k_experts: Experts each GPU sends to (at most the GPUs outside its pod).
        :param seed: Seed for expert popularity, expert selection and gate weights.
        :param skew: Popularity skew. 0 is uniform; larger values concentrate traffic on
                     fewer hot experts.
        :param priority: Priority class assigned to every generated flow.
        :return: top_k_experts Flows per GPU, in topology order of the source GPU.
        """
        if not math.isfinite(tensor_size_gb) or tensor_size_gb <= 0:
            raise ValueError("tensor_size_gb must be a finite, positive number")
        if isinstance(top_k_experts, bool) or not isinstance(top_k_experts, int) or top_k_experts < 1:
            raise ValueError("top_k_experts must be a positive integer")
        if not math.isfinite(skew) or skew < 0:
            raise ValueError("skew must be a finite, non-negative number")

        gpus = self.get_gpu_nodes()
        pod_sizes = Counter(self.pod_of(gpu) for gpu in gpus)
        if len(pod_sizes) < 2:
            raise ValueError("MoE All-to-All needs at least 2 pods")
        max_experts = len(gpus) - max(pod_sizes.values())
        if top_k_experts > max_experts:
            raise ValueError(
                f"top_k_experts={top_k_experts} exceeds the {max_experts} GPUs "
                "available outside the largest pod"
            )

        rng = random.Random(seed)  # private stream: never touches the global random state

        # Which GPUs are popular is itself random but reproducible: shuffle, then rank.
        ranked = list(gpus)
        rng.shuffle(ranked)
        popularity = {
            gpu: max(math.exp(-skew * math.log(rank + 1)), _MIN_POPULARITY)
            for rank, gpu in enumerate(ranked)
        }

        dispatch_gbits = tensor_size_gb * _BITS_PER_BYTE
        flows: list[Flow] = []
        for source in gpus:
            source_pod = self.pod_of(source)
            pool = [gpu for gpu in gpus if self.pod_of(gpu) != source_pod]

            experts: list[str] = []
            for _ in range(top_k_experts):  # weighted sampling without replacement
                pick = rng.choices(pool, weights=[popularity[gpu] for gpu in pool], k=1)[0]
                experts.append(pick)
                pool.remove(pick)

            gates = [rng.uniform(0.5, 1.5) for _ in experts]
            gate_total = sum(gates)
            for expert, gate in zip(experts, gates):
                flows.append(
                    Flow(
                        source_gpu=source,
                        destination_gpu=expert,
                        traffic_volume_gbits=dispatch_gbits * gate / gate_total,
                        priority=priority,
                    )
                )

        self.active_flows = flows
        return list(self.active_flows)


def _ingress_by_gpu(flows: list[Flow]) -> dict[str, float]:
    """Total Gbit arriving at each destination GPU."""
    ingress: dict[str, float] = {}
    for flow in flows:
        ingress[flow.destination_gpu] = ingress.get(flow.destination_gpu, 0.0) + flow.traffic_volume_gbits
    return ingress


def _imbalance(flows: list[Flow], gpu_count: int) -> float:
    """Hottest GPU's ingress divided by the mean ingress over all GPUs (1.0 is perfectly even)."""
    ingress = _ingress_by_gpu(flows)
    mean = sum(ingress.values()) / gpu_count
    return max(ingress.values()) / mean


if __name__ == "__main__":
    topo = FatTreeTopology(k=4)
    generator = AIWorkloadGenerator(topo)
    graph = topo.graph
    gpu_count = len(generator.get_gpu_nodes())

    # ------------------------------------------------------------------ #
    # Ring All-Reduce
    # ------------------------------------------------------------------ #
    tensor_size_gb = 10.0
    flows = generator.generate_all_reduce_traffic(tensor_size_gb)

    print(f"\nGenerated {len(flows)} All-Reduce flows for a {tensor_size_gb} GB tensor:\n")
    for flow in flows:
        print(
            f"  {flow.source_gpu:>14} -> {flow.destination_gpu:<14} "
            f"{flow.traffic_volume_gbits:8.2f} Gbit  [{flow.priority.name}]"
        )

    # Sanity checks against the topology.
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

    # ------------------------------------------------------------------ #
    # MoE All-to-All
    # ------------------------------------------------------------------ #
    moe_tensor_gb = 10.0
    top_k = 2
    seed = 42
    moe_flows = generator.generate_moe_all_to_all_traffic(moe_tensor_gb, top_k_experts=top_k, seed=seed)

    print(
        f"\n{'=' * 72}\nGenerated {len(moe_flows)} MoE All-to-All flows "
        f"({gpu_count} GPUs x top-{top_k}, {moe_tensor_gb} GB per GPU, seed {seed}):\n"
    )
    for flow in moe_flows[:8]:
        print(
            f"  {flow.source_gpu:>14} -> {flow.destination_gpu:<14} "
            f"{flow.traffic_volume_gbits:8.2f} Gbit  [{flow.priority.name}]"
        )
    print(f"  ... {len(moe_flows) - 8} more")

    ingress = _ingress_by_gpu(moe_flows)
    senders = Counter(flow.destination_gpu for flow in moe_flows)
    port_gbps = {
        gpu: max(graph.edges[pred, gpu]["bandwidth_gbps"] for pred in graph.predecessors(gpu))
        for gpu in ingress
    }
    print("\nHottest experts (converging traffic):")
    for gpu, gbits in sorted(ingress.items(), key=lambda item: -item[1])[:5]:
        print(
            f"  {gpu:<14} {gbits:8.2f} Gbit in from {senders[gpu]:>2} senders "
            f"({gbits / port_gbps[gpu]:.2f}x its {port_gbps[gpu]:.0f} Gbps port)"
        )
    idle = gpu_count - len(ingress)
    uniform_flows = generator.generate_moe_all_to_all_traffic(
        moe_tensor_gb, top_k_experts=top_k, seed=seed, skew=0.0
    )
    print(f"\nExperts receiving nothing: {idle} of {gpu_count}")
    print(
        f"Ingress imbalance (hottest / mean): {_imbalance(moe_flows, gpu_count):.2f}x "
        f"(uniform popularity: {_imbalance(uniform_flows, gpu_count):.2f}x)"
    )

    # Sanity checks.
    by_source: dict[str, list[Flow]] = {}
    for flow in moe_flows:
        by_source.setdefault(flow.source_gpu, []).append(flow)
    assert len(moe_flows) == gpu_count * top_k
    assert set(by_source) == set(generator.get_gpu_nodes())
    for source, group in by_source.items():
        destinations = [f.destination_gpu for f in group]
        assert len(group) == top_k and len(set(destinations)) == top_k, source
        assert all(generator.pod_of(d) != generator.pod_of(source) for d in destinations), source
        assert math.isclose(sum(f.traffic_volume_gbits for f in group), moe_tensor_gb * 8), source
    assert all(nx.has_path(graph, f.source_gpu, f.destination_gpu) for f in moe_flows)
    assert all(f.traffic_volume_gbits > 0 and f.priority == FlowPriority.HIGH for f in moe_flows)
    assert all(set(f.to_dict()) == {"source_gpu", "destination_gpu", "traffic_volume_gbits", "priority"} for f in moe_flows)

    # Reproducibility: same seed gives identical flows, another seed does not,
    # and the global random state is left untouched.
    random.seed(123)
    state_before = random.getstate()
    again = generator.generate_moe_all_to_all_traffic(moe_tensor_gb, top_k_experts=top_k, seed=seed)
    assert random.getstate() == state_before, "generator must not touch the global random state"
    assert again == moe_flows, "same seed must reproduce identical flows"
    other = generator.generate_moe_all_to_all_traffic(moe_tensor_gb, top_k_experts=top_k, seed=seed + 1)
    assert other != moe_flows, "a different seed should give different flows"

    # Skew concentrates traffic; invalid inputs are rejected.
    assert _imbalance(moe_flows, gpu_count) >= _imbalance(uniform_flows, gpu_count)
    for bad_kwargs in ({"tensor_size_gb": -1.0}, {"tensor_size_gb": 10.0, "top_k_experts": 0},
                       {"tensor_size_gb": 10.0, "top_k_experts": gpu_count}):
        try:
            generator.generate_moe_all_to_all_traffic(**bad_kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {bad_kwargs}")
    print("\nAll MoE checks passed: shape, cross-pod targets, volume, reachability, determinism.")