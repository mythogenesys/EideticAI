import torch
import torch.nn as nn
import numpy as np
from eidetic_ai.c3.causal_graph import CausalGraph

class SimulatedStudent:
    """
    Represents a student's internal state, including a latent knowledge vector
    and an evolving internal causal graph.
    """
    def __init__(self, all_concepts: list, ground_truth_graph: CausalGraph, noise_level=0.7):
        """
        Initializes the student with a 'broken' understanding.
        
        Args:
            all_concepts (list): List of all possible concept names.
            ground_truth_graph (CausalGraph): The correct causal graph.
            noise_level (float): Probability of an edge in the student's graph being
                                 incorrect (either missing or spurious).
        """
        self.all_concepts = all_concepts
        self.num_concepts = len(all_concepts)
        self.concept_map = {concept: i for i, concept in enumerate(all_concepts)}
        self.ground_truth = ground_truth_graph
        
        # Latent knowledge vector: 0=unknown, 1=mastered
        self.knowledge_vector = torch.zeros(self.num_concepts)
        
        # Internal Causal Graph: starts as a corrupted version of the ground truth
        self.internal_graph = self._initialize_broken_graph(noise_level)

    def _initialize_broken_graph(self, noise_level: float) -> CausalGraph:
        """Creates a corrupted copy of the ground-truth graph."""
        student_cg = CausalGraph(name="Student's Internal Model")
        student_cg.graph.add_nodes_from(self.all_concepts)
        
        # Add/remove edges from the ground truth with some probability
        for u, v in self.ground_truth.graph.edges():
            if np.random.rand() > noise_level:
                student_cg.graph.add_edge(u, v) # Keep a true edge
        
        # Add some spurious, incorrect edges
        for i in range(self.num_concepts // 2): # Add a few random edges
             u, v = np.random.choice(self.all_concepts, 2, replace=False)
             if not self.ground_truth.graph.has_edge(u, v) and np.random.rand() > (1 - noise_level):
                 student_cg.graph.add_edge(u, v)
                 
        return student_cg

    def receive_lesson(self, chosen_concepts: set):
        """
        Updates the student's knowledge and causal graph based on a lesson.
        A good lesson (containing causally linked concepts) helps fix the graph.
        """
        # 1. Update knowledge vector
        for concept in chosen_concepts:
            idx = self.concept_map[concept]
            # Simple update rule: knowledge increases, capped at 1.0
            self.knowledge_vector[idx] = min(1.0, self.knowledge_vector[idx] + 0.5)

        # 2. Update internal causal graph (the 'aha!' moment)
        # If the lesson contains two concepts that are causally linked in reality,
        # the student has a chance to learn that link.
        from itertools import combinations
        for u, v in combinations(chosen_concepts, 2):
            # Check both directions for the link
            if self.ground_truth.graph.has_edge(u, v) and not self.internal_graph.graph.has_edge(u, v):
                if np.random.rand() > 0.3: # 70% chance of learning a correct link
                    self.internal_graph.graph.add_edge(u, v)
            
            if self.ground_truth.graph.has_edge(v, u) and not self.internal_graph.graph.has_edge(v, u):
                if np.random.rand() > 0.3:
                    self.internal_graph.graph.add_edge(v, u)

    def get_causal_identifiability_score(self) -> float:
        """
        Calculates how close the student's graph is to the ground truth.
        Uses a graph edit distance-like metric (Jaccard similarity of edge sets).
        """
        student_edges = set(self.internal_graph.graph.edges())
        true_edges = set(self.ground_truth.graph.edges())
        
        intersection = len(student_edges.intersection(true_edges))
        union = len(student_edges.union(true_edges))
        
        return intersection / union if union > 0 else 0.0