import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import json
import time
import torch
import numpy as np

from mlx_lm import load, generate
from sentence_transformers import SentenceTransformer
from src.qce.hc import HamiltonianConstructor
from src.qce.solver import SimulatedAnnealingSolver

MODEL_PATH = "./Meta-Llama-3-8B-Instruct-4bit"
CONCEPT_SPACE_PATH = "data/concept_spaces/physics_kinematics.json"
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2' 

def run_qce_pipeline(user_prompt: str):
    """
    Executes the full, end-to-end QCE pipeline prototype.
    """
    print("--- Eidetic AI: QCE Pipeline Prototype ---")
    print(f"User Prompt: '{user_prompt}'")
    print("-" * 40)

    print("Stage 1: Identifying relevant concepts with LLM...")
    llm_model, tokenizer = load(MODEL_PATH)
    
    with open(CONCEPT_SPACE_PATH, 'r') as f:
        concept_data = json.load(f)
    all_concepts_str = ", ".join(concept_data['concepts'])

    identification_prompt = f"""
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an expert in physics. Your task is to identify the most relevant concepts needed to answer a user's question from a predefined list. Output only a comma-separated list of the concept names. Do not add any explanation.

Available concepts: {all_concepts_str}<|eot_id|><|start_header_id|>user<|end_header_id|>
{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
    
    identified_concepts_str = generate(llm_model, tokenizer, prompt=identification_prompt, max_tokens=100, verbose=False)
    relevant_concepts = [c.strip() for c in identified_concepts_str.split(',') if c.strip() in concept_data['concepts']]
    
    if not relevant_concepts:
        print("\033[93mWarning: Could not identify any relevant concepts. Aborting.\033[0m")
        return
        
    print(f"Identified {len(relevant_concepts)} concepts: {relevant_concepts}")
    print("-" * 40)

    print("Stage 2: Generating concept embeddings...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    concept_embeddings = embedding_model.encode(relevant_concepts, convert_to_tensor=True)
    embedding_dim = concept_embeddings.shape[1]
    num_concepts = concept_embeddings.shape[0]
    print(f"Embeddings created with dimension {embedding_dim}.")
    print("-" * 40)

    print("Stage 3: Constructing Hamiltonian with (random) HC...")
    hc_model = HamiltonianConstructor(embedding_dim=embedding_dim, num_concepts=num_concepts)
    with torch.no_grad():
        hamiltonian_coeffs = hc_model(concept_embeddings.to('cpu'))
    print("Hamiltonian coefficients a and b generated.")
    print("-" * 40)

    print("Stage 4: Solving for ground state with Simulated Annealing...")
    solver = SimulatedAnnealingSolver(
        a=hamiltonian_coeffs['a'].numpy(),
        b=hamiltonian_coeffs['b'].numpy()
    )
    start_solve_time = time.time()
    ground_state_vector = solver.solve()
    solve_time = time.time() - start_solve_time
    print(f"Solver finished in {solve_time:.4f} seconds.")

    winning_concepts = [concept for i, concept in enumerate(relevant_concepts) if ground_state_vector[i] == 1]
    
    if not winning_concepts:
        print("\033[93mWarning: Solver resulted in an empty concept set. Using all identified concepts as fallback.\033[0m")
        winning_concepts = relevant_concepts

    print(f"Winning concepts (ground state): {winning_concepts}")
    print("-" * 40)

    print("Stage 5: Synthesizing final explanation with LLM...")
    winning_concepts_str = ", ".join(winning_concepts)
    synthesis_prompt = f"""
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a master physics educator. Your task is to provide a clear, concise, and conceptually rich explanation for the user's question.

IMPORTANT: You must construct your explanation PRIMARILY using the following set of concepts. Use them as the foundation of your answer.

Concepts to use: {winning_concepts_str}<|eot_id|><|start_header_id|>user<|end_header_id|>
{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
    final_response = generate(llm_model, tokenizer, prompt=synthesis_prompt, max_tokens=300, verbose=True)
    print("\n" + "-"*40)


if __name__ == "__main__":
    prompt = "What is the difference between velocity and acceleration, and how are they related?"
    run_qce_pipeline(prompt)