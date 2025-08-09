import networkx as nx
import matplotlib.pyplot as plt

class CausalGraph:
    """
    A simple directed graph to represent the causal relationships between concepts.
    """
    def __init__(self, name="Ground Truth"):
        self.graph = nx.DiGraph()
        self.name = name

    def plot(self, save_path="results/causal_graph.png"):
        """Saves a visualization of the causal graph."""
        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(self.graph, k=0.9, seed=42)
        nx.draw(self.graph, pos, with_labels=True, node_size=2500, node_color='#a31f34',
                font_size=10, font_color='white', font_weight='bold',
                arrows=True, arrowsize=20, edge_color='gray', width=2)
        plt.title(f"{self.name} Causal Graph", size=16)
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Causal graph visualization saved to {save_path}")

def get_kinematics_causal_graph() -> CausalGraph:
    """
    Returns the ground-truth causal graph for our kinematics domain.
    An edge A -> B means A is a direct cause of B.
    """
    cg = CausalGraph(name="Kinematics Ground Truth")
    
    # Foundational concepts (no parents)
    cg.graph.add_node("time")
    cg.graph.add_node("displacement")
    
    # First-order relationships
    cg.graph.add_edge("time", "velocity")
    cg.graph.add_edge("displacement", "velocity") # Velocity is the rate of change of displacement over time
    
    # Second-order relationships
    cg.graph.add_edge("velocity", "acceleration") # Acceleration is the rate of change of velocity
    cg.graph.add_edge("time", "acceleration")
    
    # Properties
    cg.graph.add_edge("velocity", "speed") # Speed is the magnitude of velocity
    cg.graph.add_edge("velocity", "direction") # Direction is a component of velocity
    
    # Add other concepts as nodes, even if they aren't part of the core causal chain
    cg.graph.add_nodes_from(["vector", "scalar", "gravity", "projectile motion"])
    
    return cg

if __name__ == '__main__':
    # Generate and save the plot of the ground-truth graph
    ground_truth = get_kinematics_causal_graph()
    ground_truth.plot()