import numpy as np
import random
import math

class SimulatedAnnealingSolver:
    """
    A simple QUBO solver using Simulated Annealing to find the ground state
    of a conceptual Hamiltonian.
    """
    def __init__(self, a: np.ndarray, b: np.ndarray):
        """
        Initializes the solver with Hamiltonian coefficients.
        Args:
            a (np.ndarray): Linear coefficients (vector). Shape (N,).
            b (np.ndarray): Quadratic coefficients (matrix). Shape (N, N).
        """
        self.a = a
        self.b = b
        self.num_concepts = len(a)
        if self.num_concepts == 0:
            raise ValueError("Cannot solve for an empty set of concepts.")

    def _calculate_energy(self, x: np.ndarray) -> float:
        """Calculates the energy H(x) = x^T * B * x + a^T * x"""
        linear_term = np.dot(self.a, x)
        # Note: 0.5 * x.T * B * x is for the unique pairs. Since B is symmetric
        # with zero diagonal, x.T*B*x counts each pair twice.
        quadratic_term = 0.5 * np.dot(x.T, np.dot(self.b, x))
        return linear_term + quadratic_term

    def solve(self, initial_temp=1.0, final_temp=0.01, alpha=0.995, steps_per_temp=100):
        """
        Runs the simulated annealing algorithm.

        Args:
            initial_temp (float): Starting temperature.
            final_temp (float): Ending temperature.
            alpha (float): Cooling rate (multiplicative factor).
            steps_per_temp (int): Number of iterations at each temperature.

        Returns:
            np.ndarray: The binary vector x representing the lowest energy state found.
        """
        # Start with a random configuration
        current_x = np.random.randint(2, size=self.num_concepts)
        current_energy = self._calculate_energy(current_x)
        
        best_x = np.copy(current_x)
        best_energy = current_energy
        
        temp = initial_temp

        while temp > final_temp:
            for _ in range(steps_per_temp):
                # Propose a new state by flipping a random bit
                proposal_x = np.copy(current_x)
                flip_index = random.randint(0, self.num_concepts - 1)
                proposal_x[flip_index] = 1 - proposal_x[flip_index]
                
                proposal_energy = self._calculate_energy(proposal_x)
                
                delta_energy = proposal_energy - current_energy
                
                # Acceptance condition
                if delta_energy < 0 or random.random() < math.exp(-delta_energy / temp):
                    current_x = np.copy(proposal_x)
                    current_energy = proposal_energy
                    
                    if current_energy < best_energy:
                        best_x = np.copy(current_x)
                        best_energy = current_energy
            
            temp *= alpha
            
        return best_x