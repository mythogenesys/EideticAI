import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from tqdm import tqdm

import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- Local Imports ---
from eidetic_ai.c3.causal_graph import get_kinematics_causal_graph
from eidetic_ai.c3.student_model import SimulatedStudent
from eidetic_ai.progs.dsl import ProgramInterpreter
from eidetic_ai.progs.program_synthesis import ProgrammaticTeacher, generate_random_program, crossover

# --- Configuration ---
CONCEPT_SPACE_PATH = "data/concept_spaces/physics_kinematics.json"
POPULATION_SIZE = 50
NUM_GENERATIONS = 100
NUM_ELITES = 5 # Number of best individuals to carry over to the next generation
MUTATION_RATE = 0.8 # Probability that an individual will be mutated
NUM_STUDENT_EVALS = 5 # Evaluate each teacher on 5 different random students

def evaluate_teacher(teacher: ProgrammaticTeacher, all_concepts, ground_truth_graph) -> float:
    """Evaluates a teacher's fitness by testing it on a number of students."""
    scores = []
    for _ in range(NUM_STUDENT_EVALS):
        # Create a new "confused" student
        student = SimulatedStudent(all_concepts, ground_truth_graph, noise_level=0.8)
        initial_causal_score = student.get_causal_identifiability_score()
        
        # The interpreter runs the teacher's program on the student
        interpreter = ProgramInterpreter(student, all_concepts)
        interpreter.run(teacher.program)
        
        # Fitness is the improvement in the student's causal understanding
        final_causal_score = student.get_causal_identifiability_score()
        score = final_causal_score - initial_causal_score
        scores.append(score)
    
    return np.mean(scores)

def run_popgen_training():
    """
    Executes an evolutionary training loop for programmatic teachers.
    """
    print("--- Eidetic AI: Programmatic Co-evolution (PopGen) ---")

    # --- 1. Setup Environment ---
    print("Setting up environment...")
    with open(CONCEPT_SPACE_PATH, 'r') as f:
        concepts = json.load(f)['concepts']
    
    ground_truth_graph = get_kinematics_causal_graph()

    # --- 2. Initialize Population ---
    print(f"Generating initial population of {POPULATION_SIZE} random teachers...")
    population = [ProgrammaticTeacher(generate_random_program(concepts), concepts) for _ in range(POPULATION_SIZE)]

    history = {'best_fitness': [], 'avg_fitness': []}

    # --- 3. The Evolutionary Loop ---
    print(f"Starting evolution for {NUM_GENERATIONS} generations...")
    for gen in tqdm(range(NUM_GENERATIONS), desc="Evolving Generations"):
        
        # --- A. Evaluate Fitness of the entire population ---
        for teacher in population:
            teacher.fitness = evaluate_teacher(teacher, concepts, ground_truth_graph)
        
        # Sort the population by fitness (higher is better)
        population.sort(key=lambda t: t.fitness, reverse=True)
        
        best_teacher_this_gen = population[0]
        history['best_fitness'].append(best_teacher_this_gen.fitness)
        history['avg_fitness'].append(np.mean([t.fitness for t in population]))

        if (gen + 1) % 10 == 0:
            print(f"\nGen {gen+1}/{NUM_GENERATIONS} | Best Fitness: {best_teacher_this_gen.fitness:.4f} | Avg Fitness: {history['avg_fitness'][-1]:.4f}")

        # --- B. Create the Next Generation ---
        next_generation = []
        
        # 1. Elitism: The best individuals survive automatically
        elites = population[:NUM_ELITES]
        next_generation.extend(elites)
        
        # 2. Crossover & Mutation: Fill the rest of the population
        while len(next_generation) < POPULATION_SIZE:
            # Select parents (e.g., tournament selection or just randomly from top 50%)
            parent1 = random.choice(population[:POPULATION_SIZE // 2])
            parent2 = random.choice(population[:POPULATION_SIZE // 2])
            
            child = crossover(parent1, parent2)
            
            if random.random() < MUTATION_RATE:
                child.mutate()
            
            next_generation.append(child)
            
        population = next_generation

    print("Evolution complete.")

    # --- 4. Display Best Discovered Program ---
    best_overall_teacher = population[0]
    print("\n--- Best Discovered Curriculum-Program ---")
    print(f"Fitness (Causal Improvement): {best_overall_teacher.fitness:.4f}")
    print(best_overall_teacher)
    
    # --- 5. Visualize Results ---
    plt.figure(figsize=(10, 6), dpi=100)
    plt.plot(history['avg_fitness'], label='Average Population Fitness')
    plt.plot(history['best_fitness'], label='Best Teacher Fitness', color='#a31f34', linewidth=2)
    plt.title("Evolution of Teacher Fitness over Generations", size=16)
    plt.xlabel("Generation")
    plt.ylabel("Fitness (Avg Causal Score Improvement)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.savefig("results/popgen_training_curves.png")
    print("PopGen training curves saved to results/popgen_training_curves.png")

if __name__ == "__main__":
    run_popgen_training()