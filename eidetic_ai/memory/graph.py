import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

class AssociativeMemoryGraph:
    """
    A graph-based memory system to store and strengthen associations
    between concepts that co-occur in successful explanations.
    """
    def __init__(self):
        self.graph = nx.Graph()

    def add_concepts(self, concepts: list):
        """Ensure all concepts exist as nodes in the graph."""
        self.graph.add_nodes_from(concepts)
        for node in concepts:
            if 'weight' not in self.graph.nodes[node]:
                self.graph.nodes[node]['weight'] = 1.0

    def potentiate(self, chosen_concepts: list, utility: float, eta=0.1, decay=0.01):
        """
        Strengthen connections between concepts that appeared together in a
        successful interaction (Hebbian learning).
        """
        if utility <= 0:
            return

        from itertools import combinations
        # Apply global decay
        for u, v, d in self.graph.edges(data=True):
            d['weight'] = max(0, d.get('weight', 0) * (1.0 - decay))

        # Strengthen connections
        for concept1, concept2 in combinations(chosen_concepts, 2):
            new_weight = eta * utility
            if self.graph.has_edge(concept1, concept2):
                self.graph[concept1][concept2]['weight'] += new_weight
            else:
                self.graph.add_edge(concept1, concept2, weight=new_weight)

    def visualize(self, save_path="results/memory_graph.png"):
        """Saves a visualization of the memory graph."""
        if self.graph.number_of_nodes() == 0:
            print("Graph is empty, cannot visualize.")
            return

        plt.figure(figsize=(14, 14))
        ax = plt.gca()
        ax.set_title("Associative Memory Graph", size=20)

        # Use a layout that is stable and works for all graph structures
        pos = nx.spring_layout(self.graph, k=0.9, iterations=50, seed=42)

        try:
            # Draw Edges first
            if self.graph.number_of_edges() > 0:
                edges = self.graph.edges()
                weights = [self.graph[u][v].get('weight', 0) for u, v in edges]
                max_weight = max(weights) if weights else 1.0
                edge_widths = [1 + 6 * w / max_weight for w in weights]
                nx.draw_networkx_edges(pos, edgelist=edges, width=edge_widths, alpha=0.7, edge_color='gray', ax=ax)

            # Draw Nodes and Labels
            nx.draw_networkx_nodes(pos, self.graph.nodes(), node_size=3000, node_color='#a31f34', alpha=0.95, ax=ax)
            nx.draw_networkx_labels(pos, font_size=10, font_color='white', font_weight='bold', ax=ax)
            
        except Exception as e:
            print(f"\n\033[93mWarning: An error occurred during graph visualization: {e}\033[0m")
            print("Skipping graph image generation.")
            plt.close()
            return

        ax.margins(0.1)
        ax.set_axis_off()
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Memory graph visualization saved to {save_path}")