import torch
import torch.nn.functional as F

class ContinuousRelaxationSolver:
    """
    Solves for the QCE ground state by optimizing a continuous relaxation
    of the Hamiltonian energy function using gradient descent.
    
    This is a more principled and efficient approach than heuristic search.
    """
    def __init__(self, a: torch.Tensor, b: torch.Tensor, steps=100, lr=0.1):
        """
        Initializes the solver with Hamiltonian coefficients as Tensors.
        Args:
            a (torch.Tensor): Linear coefficients. Shape (N,).
            b (torch.Tensor): Quadratic coefficients. Shape (N, N).
            steps (int): Number of optimization steps.
            lr (float): Learning rate for the gradient descent.
        """
        self.a = a
        self.b = b
        self.num_concepts = len(a)
        self.steps = steps
        self.lr = lr
        if self.num_concepts == 0:
            raise ValueError("Cannot solve for an empty set of concepts.")

    def _calculate_energy(self, x_relaxed: torch.Tensor) -> torch.Tensor:
        """Calculates the relaxed energy H(x) = 0.5*x^T*B*x + a^T*x"""
        # We use torch.einsum for efficient, clear tensor contractions.
        # 'i,ij,j->' computes the full quadratic term x.T * B * x
        quadratic_term = 0.5 * torch.einsum('i,ij,j->', x_relaxed, self.b, x_relaxed)
        linear_term = torch.dot(self.a, x_relaxed)
        return linear_term + quadratic_term

    def solve(self) -> torch.Tensor:
        """
        Runs the gradient-based optimization.

        Returns:
            torch.Tensor: The binary vector x representing the found low-energy state.
        """
        # Start with a random continuous configuration between 0 and 1.
        # This tensor requires gradients so we can optimize it.
        x_relaxed = torch.rand(self.num_concepts, requires_grad=True)
        
        # Use Adam, a robust gradient-based optimizer.
        optimizer = torch.optim.Adam([x_relaxed], lr=self.lr)

        for _ in range(self.steps):
            optimizer.zero_grad()
            
            # Clamp the values to the [0, 1] box constraint after each step
            with torch.no_grad():
                x_relaxed.data.clamp_(0, 1)

            energy = self._calculate_energy(x_relaxed)
            energy.backward()
            optimizer.step()

        # Final cleanup and discretization
        final_x = x_relaxed.detach()
        final_x.clamp_(0, 1)
        
        # Discretize the final continuous solution to a binary vector.
        return (final_x > 0.5).float()