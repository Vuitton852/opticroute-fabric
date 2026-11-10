import networkx as nx

class FatTreeTopology:
    def __init__(self, k: int = 4):
        """
        Initializes a 3-Tier Fat-Tree Network Topology.
        :param k: The port count of the switches (determines cluster scale).
                  Total GPUs = (k^3) // 4. For k=4, that's 16 simulated GPUs.
                  Scale out to k=48 or higher for massive cluster limits.
        """
        self.k = k
        self.graph = nx.DiGraph() # Directed Graph representing two-way fabric paths
        
        # Calculate structure limits
        self.num_pods = k
        self.core_switches = (k // 2) ** 2
        self.switches_per_pod = k // 2 # Half aggregation, half edge
        self.gpus_per_edge_switch = k // 2
        
        self._build_network()

    def _build_network(self):
        # 1. Create Core Switches
        core_nodes = [f"core_{i}" for i in range(self.core_switches)]
        self.graph.add_nodes_from(core_nodes, type="core")

        # 2. Build Pods (Aggregation Tier, Edge Tier, and GPUs)
        for pod in range(self.num_pods):
            agg_nodes = [f"pod_{pod}_agg_{i}" for i in range(self.switches_per_pod)]
            edge_nodes = [f"pod_{pod}_edge_{i}" for i in range(self.switches_per_pod)]
            
            self.graph.add_nodes_from(agg_nodes, type="aggregation")
            self.graph.add_nodes_from(edge_nodes, type="edge")

            # Link Aggregation tier to Core tier
            for agg_idx, agg_node in enumerate(agg_nodes):
                core_offset = agg_idx * (self.k // 2)
                for c in range(self.k // 2):
                    core_node = core_nodes[core_offset + c]
                    # Optical fiber simulation (High bandwidth, lightning-fast light-speed link)
                    self.graph.add_edge(agg_node, core_node, bandwidth_gbps=800.0, base_latency_ns=50.0, current_load=0.0)
                    self.graph.add_edge(core_node, agg_node, bandwidth_gbps=800.0, base_latency_ns=50.0, current_load=0.0)

            # Link Edge tier to Aggregation tier within the same Pod
            for agg_node in agg_nodes:
                for edge_node in edge_nodes:
                    # High-speed copper or local short-reach optical fabric link
                    self.graph.add_edge(edge_node, agg_node, bandwidth_gbps=400.0, base_latency_ns=100.0, current_load=0.0)
                    self.graph.add_edge(agg_node, edge_node, bandwidth_gbps=400.0, base_latency_ns=100.0, current_load=0.0)

            # 3. Create and Link GPUs to Edge Switches
            for edge_idx, edge_node in enumerate(edge_nodes):
                for g in range(self.gpus_per_edge_switch):
                    gpu_id = f"pod_{pod}_gpu_{edge_idx * self.gpus_per_edge_switch + g}"
                    self.graph.add_node(gpu_id, type="gpu")
                    # Direct short copper NVLink/PCIe or server backplane interface
                    self.graph.add_edge(gpu_id, edge_node, bandwidth_gbps=400.0, base_latency_ns=20.0, current_load=0.0)
                    self.graph.add_edge(edge_node, gpu_id, bandwidth_gbps=400.0, base_latency_ns=20.0, current_load=0.0)

        print(f"Successfully constructed a Fat-Tree cluster fabric with {len(self.graph.nodes)} interconnected nodes.")

if __name__ == "__main__":
    # Test execution block to verify topology works standalone
    topo = FatTreeTopology(k=4)
