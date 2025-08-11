import json
import torch
import torch.optim as optim
import numpy as np
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import torch.nn.functional as F

import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.qce.hc import DifferentiablePolicy
from eidetic_ai.memory.graph import AssociativeMemoryGraph

CONCEPT_SPACE_PATH = "data/concept_spaces/physics_kinematics.json"
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
NUM_EPOCHS = 200
LEARNING_RATE = 0.001
L1_LAMBDA = 0.005

def run_principled_training():
    """
    Executes a principled training loop using a differentiable policy.
    This is the Sprowls-level version of the Phase 2 experiment.
    """
    print("--- Eidetic AI: Principled Training Simulation (Phase 2 Reforged) ---")

    # --- 1. Setup Environment ---
    print("Setting up environment...")
    with open(CONCEPT_SPACE_PATH, 'r') as f:
        concepts = json.load(f)['concepts']
    
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    concept_embeddings = torch.tensor(embedding_model.encode(concepts)).float()
    embedding_dim = concept_embeddings.shape[1]
    num_concepts = len(concepts)

    ideal_concepts = {"displacement", "velocity", "acceleration", "time", "rate of change", "vector"}
    target_vector = torch.tensor([1.0 if c in ideal_concepts else 0.0 for c in concepts])

    # --- 2. Initialize Models and Optimizers ---
    print("Initializing Differentiable Policy and Memory...")
    policy_network = DifferentiablePolicy(embedding_dim, num_concepts)
    memory = AssociativeMemoryGraph()
    memory.add_concepts(concepts)
    optimizer = optim.Adam(policy_network.parameters(), lr=LEARNING_RATE)

    history = {'loss': [], 'f1_score': []}

    # --- 3. The Differentiable Training Loop ---
    print(f"Starting training for {NUM_EPOCHS} epochs...")
    for epoch in range(NUM_EPOCHS):
        
        policy_network.train()
        optimizer.zero_grad()
        
        policy_vector = policy_network(concept_embeddings)
        
        bce_loss = F.binary_cross_entropy(policy_vector, target_vector)
        sparsity_loss = L1_LAMBDA * torch.mean(policy_vector)
        total_loss = bce_loss + sparsity_loss

        total_loss.backward()
        optimizer.step()
        
        # --- C. Logging and Metrics ---
        policy_network.eval()
        with torch.no_grad():
            action_vector = (policy_vector > 0.5).float()
            
            chosen_indices = set(np.where(action_vector.numpy() == 1)[0])
            ideal_indices_set = set(np.where(target_vector.numpy() == 1)[0])
            
            intersection = len(chosen_indices.intersection(ideal_indices_set))
            precision = intersection / len(chosen_indices) if len(chosen_indices) > 0 else 0
            recall = intersection / len(ideal_indices_set) if len(ideal_indices_set) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        # --- D. Update Memory and Log ---
        if f1_score > 0.8 and len(chosen_indices) > 1:
            chosen_concepts_list = [concepts[i] for i in chosen_indices]
            memory.potentiate(chosen_concepts_list, f1_score)

        history['loss'].append(total_loss.item())
        history['f1_score'].append(f1_score)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Loss: {total_loss.item():.4f} | F1 Score: {f1_score:.3f} | Concepts: {len(chosen_indices)}")

    print("Training complete.")

    # --- 4. Visualize Results ---
    fig, axs = plt.subplots(1, 2, figsize=(18, 6))
    
    axs[0].plot(history['f1_score'], label='F1 Score')
    f1_moving_avg = np.convolve(history['f1_score'], np.ones(10)/10, mode='valid')
    axs[0].plot(range(9, len(history['f1_score'])), f1_moving_avg, color='#a31f34', label='10-epoch MA', linewidth=2)
    axs[0].set_title("F1 Score per Epoch", size=16)
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("F1 Score")
    axs[0].grid(True, linestyle='--', alpha=0.6)
    axs[0].legend()
    
    axs[1].plot(history['loss'], label='Total Loss (BCE + L1)')
    axs[1].set_title("Loss Curve", size=16)
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Loss")
    axs[1].grid(True, linestyle='--', alpha=0.6)
    axs[1].legend()
    
    plt.tight_layout()
    plt.savefig("results/training_curves_principled.png")
    print("Training curves saved to results/training_curves_principled.png")

    memory.visualize()

    # --- 5. Save the Trained Model ---
    trained_model_path = "eidetic_ai/qce/models/trained_policy.pth"
    os.makedirs(os.path.dirname(trained_model_path), exist_ok=True)
    torch.save(policy_network.state_dict(), trained_model_path)
    print(f"\n--- Trained model weights saved to {trained_model_path} ---")

if __name__ == "__main__":
    run_principled_training()