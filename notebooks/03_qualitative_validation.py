import json
import torch
import numpy as np
from sentence_transformers import SentenceTransformer

import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- Local Imports ---
from mlx_lm import load, generate
from src.qce.hc import DifferentiablePolicy

# --- Configuration ---
MODEL_PATH = "./Meta-Llama-3-8B-Instruct-4bit"
CONCEPT_SPACE_PATH = "data/concept_spaces/physics_kinematics.json"
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
TRAINED_POLICY_PATH = "eidetic_ai/qce/models/trained_policy.pth"

def run_single_pipeline(policy_network, llm_model, tokenizer, concepts, concept_embeddings, user_prompt, title=""):
    """
    Runs a single pass of the QCE pipeline using a given policy network.
    """
    print("\n" + "="*20 + f" {title} " + "="*20)
    
    policy_network.eval()
    with torch.no_grad():
        # === Stage 1: Get Policy Vector ===
        # The policy network directly outputs the concept selection probabilities.
        policy_vector = policy_network(concept_embeddings)
        
        # === Stage 2: Discretize to Get Winning Concepts ===
        action_vector = (policy_vector > 0.5).float()
        chosen_indices = np.where(action_vector.numpy() == 1)[0]
        winning_concepts = [concepts[i] for i in chosen_indices]

    if not winning_concepts:
        print("\n\033[93mResult: The policy network selected ZERO concepts.\033[0m")
        # In a real app, we would have a fallback, but for this demo, we stop.
        return

    print(f"\nWinning Concepts ({len(winning_concepts)}): {winning_concepts}")

    # === Stage 3: Synthesize Final Explanation with LLM ===
    print("\nSynthesizing final explanation...")
    winning_concepts_str = ", ".join(winning_concepts)
    synthesis_prompt = f"""
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a master physics educator. Your task is to provide a clear, concise, and conceptually rich explanation for the user's question.

IMPORTANT: You must construct your explanation PRIMARILY using the following set of concepts. Use them as the foundation of your answer.

Concepts to use: {winning_concepts_str}<|eot_id|><|start_header_id|>user<|end_header_id|>
{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
    final_response = generate(llm_model, tokenizer, prompt=synthesis_prompt, max_tokens=300, verbose=True)
    print("\n" + "="* (42 + len(title)))

def run_qualitative_comparison():
    """
    Performs a side-by-side qualitative comparison of the untrained vs. trained policy.
    """
    print("--- Eidetic AI: Qualitative Before & After Validation ---")

    # --- 1. Setup Environment (once) ---
    print("\nSetting up environment...")
    user_prompt = "What is the difference between velocity and acceleration, and how are they related?"
    
    with open(CONCEPT_SPACE_PATH, 'r') as f:
        concepts = json.load(f)['concepts']
    
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    concept_embeddings = torch.tensor(embedding_model.encode(concepts)).float()
    embedding_dim = concept_embeddings.shape[1]
    num_concepts = len(concepts)
    
    print("Loading base LLM for synthesis...")
    llm_model, tokenizer = load(MODEL_PATH)

    # --- 2. Run with UNTRAINED Model (The "Before") ---
    untrained_policy_net = DifferentiablePolicy(embedding_dim, num_concepts)
    run_single_pipeline(untrained_policy_net, llm_model, tokenizer, concepts, concept_embeddings, user_prompt, title="UNTRAINED MODEL")

    # --- 3. Run with TRAINED Model (The "After") ---
    if not os.path.exists(TRAINED_POLICY_PATH):
        print(f"\n\033[91mError: Trained model not found at '{TRAINED_POLICY_PATH}'\033[0m")
        print("Please run 'notebooks/02_ascedant_k_simulation.py' first to train and save the model.")
        return
        
    trained_policy_net = DifferentiablePolicy(embedding_dim, num_concepts)
    trained_policy_net.load_state_dict(torch.load(TRAINED_POLICY_PATH))
    run_single_pipeline(trained_policy_net, llm_model, tokenizer, concepts, concept_embeddings, user_prompt, title="TRAINED MODEL")


if __name__ == "__main__":
    run_qualitative_comparison()