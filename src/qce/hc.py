import torch
import torch.nn as nn

class HamiltonianConstructor(nn.Module):
    """
    A simple MLP to construct the coefficients of a conceptual Hamiltonian.

    For the Phase 1 prototype, this model is used with random weights
    to demonstrate the architectural pipeline. Training this model with
    reinforcement learning is the objective of Phase 2.
    """
    def __init__(self, embedding_dim: int, num_concepts: int):
        super(HamiltonianConstructor, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_concepts = num_concepts

        # A simple two-layer MLP
        self.layer1 = nn.Linear(embedding_dim, 128)
        self.activation = nn.ReLU()
        self.layer2 = nn.Linear(128, 64)

        # Output heads for Hamiltonian coefficients
        # Head for linear terms (a_i)
        self.a_head = nn.Linear(64, 1)
        
        # Head for quadratic terms (b_ij) - requires processing pairs
        # We model this by learning a representation for each concept
        # and then computing interactions.
        self.b_head_transform = nn.Linear(64, 32) # Projects concept representation into interaction space

    def forward(self, concept_embeddings: torch.Tensor):
        """
        Args:
            concept_embeddings (torch.Tensor): A tensor of shape (num_concepts, embedding_dim).

        Returns:
            dict: A dictionary containing the Hamiltonian coefficients 'a' and 'b'.
                  'a': (num_concepts,)
                  'b': (num_concepts, num_concepts)
        """
        if concept_embeddings.shape[0] != self.num_concepts:
            raise ValueError(f"Expected {self.num_concepts} concept embeddings, but got {concept_embeddings.shape[0]}")

        # Pass concept embeddings through the shared MLP layers
        x = self.activation(self.layer1(concept_embeddings))
        concept_reps = self.activation(self.layer2(x)) # Shape: (num_concepts, 64)

        # --- Calculate Linear Coefficients (a_i) ---
        # Each concept gets its own intrinsic utility score.
        a = self.a_head(concept_reps).squeeze(-1) # Shape: (num_concepts,)

        # --- Calculate Quadratic Coefficients (b_ij) ---
        # The interaction strength between two concepts is modeled as the
        # dot product of their transformed representations.
        interaction_reps = self.b_head_transform(concept_reps) # Shape: (num_concepts, 32)
        
        # b_ij = rep_i^T * rep_j. We compute this for all pairs.
        # This creates a symmetric interaction matrix.
        b = torch.matmul(interaction_reps, interaction_reps.T)

        # Zero out the diagonal, as x_i * x_i = x_i is a linear term
        # and self-interactions are not needed in the quadratic part.
        b.fill_diagonal_(0)

        # We only need the upper triangle for the sum, but a symmetric
        # matrix is easier to work with for the solver.
        
        return {'a': a, 'b': b}