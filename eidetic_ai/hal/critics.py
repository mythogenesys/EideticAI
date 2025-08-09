import torch
import torch.nn as nn
from typing import Set

class Critic_L0_Factual:
    """
    A simple, rule-based critic for Level 0 (Factual Definition).
    It checks if the core, essential concepts for a definition are present.
    This critic is not learned.
    """
    def get_fragility(self, chosen_concepts: Set[str], ideal_concepts: Set[str]) -> float:
        """
        Calculates fragility based on missing essential concepts.
        Fragility is the proportion of ideal concepts that were missed.
        
        Returns:
            float: A fragility score between 0.0 (perfect) and 1.0 (completely wrong).
        """
        if not ideal_concepts:
            return 0.0
        
        missing_concepts = ideal_concepts.difference(chosen_concepts)
        fragility = len(missing_concepts) / len(ideal_concepts)
        return fragility

class Critic_L1_Procedural(nn.Module):
    """
    A neural critic for Level 1 (Procedural Application).
    This model learns to predict if a set of concepts is sufficient
    to solve a simple procedural problem.
    """
    def __init__(self, embedding_dim: int):
        super(Critic_L1_Procedural, self).__init__()
        # The architecture is similar to our previous critic
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, concept_embeddings_policy: torch.Tensor) -> torch.Tensor:
        """
        Takes a "soft" embedding representing the chosen concepts and predicts fragility.
        A high output means the concept set is likely insufficient.
        
        Args:
            concept_embeddings_policy (torch.Tensor): A weighted average of all concept
                embeddings, based on the policy probabilities. Shape (embedding_dim,).
        """
        # The input is already a single vector, so we just pass it through the net
        return torch.sigmoid(self.net(concept_embeddings_policy)) # Output a score between 0 and 1

# For now, the L2 critic can share the same architecture as the L1 critic.
# In a full implementation, it might have a more complex, attention-based architecture
# to better handle interactions between many concepts.
Critic_L2_Modeling = Critic_L1_Procedural