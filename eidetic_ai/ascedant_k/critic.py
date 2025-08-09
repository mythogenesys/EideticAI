import torch
import torch.nn as nn

class Critic(nn.Module):
    """
    A simple model to evaluate the quality of a conceptual explanation.

    In a real system, this would be distilled from vast student data. In our
    simulation, we will train it on a synthetic task: to penalize explanations
    that are either too sparse (missing key concepts) or too dense (full of
    irrelevant concepts).
    """
    def __init__(self, embedding_dim: int):
        super(Critic, self).__init__()
        # The input is the sum-pooled embedding of the chosen concepts
        self.layer1 = nn.Linear(embedding_dim, 128)
        self.activation = nn.ReLU()
        self.layer2 = nn.Linear(128, 64)
        # The output is a single scalar value representing "fragility" or "confusion"
        self.output_head = nn.Linear(64, 1)

    def forward(self, chosen_concept_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            chosen_concept_embeddings (torch.Tensor): A tensor of shape 
                (num_chosen_concepts, embedding_dim).

        Returns:
            torch.Tensor: A scalar tensor representing the fragility score.
        """
        if chosen_concept_embeddings.dim() == 1:
            # Handle case where only one concept is chosen
            pooled_embedding = chosen_concept_embeddings
        elif len(chosen_concept_embeddings) == 0:
            # Handle empty set of concepts (highly fragile)
            return torch.tensor(10.0) # Return a large penalty
        else:
            # Simple average pooling of the chosen concept embeddings
            pooled_embedding = torch.mean(chosen_concept_embeddings, dim=0)
        
        x = self.activation(self.layer1(pooled_embedding))
        x = self.activation(self.layer2(x))
        fragility_score = self.output_head(x)
        
        return fragility_score