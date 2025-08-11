import torch
import torch.nn as nn


class DifferentiablePolicy(nn.Module):
    """
    A direct policy network that learns to output the optimal continuous
    representation of a concept selection.

    This architecture is defined completely in the __init__ method, ensuring
    that the optimizer correctly tracks all trainable parameters.
    """
    def __init__(self, embedding_dim: int, num_concepts: int):
        super(DifferentiablePolicy, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_concepts = num_concepts

        # This is a per-concept processing network. Each concept embedding
        # is passed through this MLP independently to get a logit.
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.LayerNorm(64),
            nn.Linear(64, 1) # Output a single logit per concept
        )

    def forward(self, concept_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Takes concept embeddings and directly outputs the policy, i.e.,
        the probability of including each concept in the final explanation.

        Args:
            concept_embeddings (torch.Tensor): Shape (num_concepts, embedding_dim)

        Returns:
            torch.Tensor: A tensor of shape (num_concepts,) with values
                          between 0 and 1, representing the policy.
        """
        # Pass each concept embedding through the network
        logits = self.net(concept_embeddings).squeeze(-1)
        
        # The policy is the sigmoid of the logits
        return torch.sigmoid(logits)