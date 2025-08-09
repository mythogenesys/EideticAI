import json
import torch
import torch.optim as optim
import numpy as np
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import pandas as pd
import random
import torch.nn.functional as F


import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- Local Imports ---
from src.qce.hc import DifferentiablePolicy
from eidetic_ai.hal.abstactions import HIERARCHY
from eidetic_ai.hal.critics import Critic_L0_Factual, Critic_L1_Procedural, Critic_L2_Modeling

# --- Configuration ---
CONCEPT_SPACE_PATH = "data/concept_spaces/physics_kinematics.json"
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
NUM_EPOCHS = 300
LEARNING_RATE = 0.001

def run_hal_training():
    """
    Executes a training loop for the HAL framework with decoupled update steps.
    """
    print("--- Eidetic AI: HAL Framework Training ---")

    # --- 1. Setup Environment ---
    print("Setting up environment...")
    with open(CONCEPT_SPACE_PATH, 'r') as f:
        concepts = json.load(f)['concepts']
    
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    concept_embeddings = torch.tensor(embedding_model.encode(concepts)).float()
    embedding_dim = concept_embeddings.shape[1]
    num_concepts = len(concepts)

    # --- 2. Initialize Models ---
    print("Initializing Teacher and Critic Ladder...")
    teacher_policy = DifferentiablePolicy(embedding_dim, num_concepts)
    optimizer_teacher = optim.Adam(teacher_policy.parameters(), lr=LEARNING_RATE)

    critics = {
        0: Critic_L0_Factual(),
        1: Critic_L1_Procedural(embedding_dim),
        2: Critic_L2_Modeling(embedding_dim)
    }
    # Combine trainable critic parameters into a single optimizer
    trainable_critic_params = list(critics[1].parameters()) + list(critics[2].parameters())
    optimizer_critics = optim.Adam(trainable_critic_params, lr=LEARNING_RATE)

    history = {'teacher_loss': [], 'critic_loss': [], 'f1_score_l0': [], 'f1_score_l1': [], 'f1_score_l2': []}

    # --- 3. The Training Loop ---
    print(f"Starting training for {NUM_EPOCHS} epochs...")
    for epoch in range(NUM_EPOCHS):
        
        # --- A. Sample a Training Problem ---
        level_data = random.choice(HIERARCHY)
        problem = random.choice(level_data.examples)
        ideal_concepts = set(problem['ideal_concepts'])
        target_vector = torch.tensor([1.0 if c in ideal_concepts else 0.0 for c in concepts])

        # =====================================================
        #                CRITICS UPDATE STEP
        # =====================================================
        # The critics learn to predict the fragility of the *current* teacher's policy.
        # We freeze the teacher's weights during this phase.
        teacher_policy.eval()
        optimizer_critics.zero_grad()

        # Get the teacher's current policy (we don't need gradients for the teacher here)
        with torch.no_grad():
            policy_vector = teacher_policy(concept_embeddings)
            action_vector = (policy_vector > 0.5).float()
            
            # Calculate the true fragility (1 - F1 score) to use as the learning target
            chosen_indices = set(np.where(action_vector.numpy() == 1)[0])
            ideal_indices_set = set(np.where(target_vector.numpy() == 1)[0])
            intersection = len(chosen_indices.intersection(ideal_indices_set))
            precision = intersection / len(chosen_indices) if len(chosen_indices) > 0 else 0
            recall = intersection / len(ideal_indices_set) if len(ideal_indices_set) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        actual_fragility = torch.tensor(1.0 - f1_score)

        # The critics' forward pass needs to be done with gradients enabled
        soft_embedding = torch.matmul(policy_vector, concept_embeddings)
        predicted_fragility_l1 = critics[1](soft_embedding).squeeze()
        predicted_fragility_l2 = critics[2](soft_embedding).squeeze()
        
        # Calculate loss for both trainable critics
        critic_l1_loss = F.mse_loss(predicted_fragility_l1, actual_fragility)
        critic_l2_loss = F.mse_loss(predicted_fragility_l2, actual_fragility)
        total_critic_loss = critic_l1_loss + critic_l2_loss
        
        total_critic_loss.backward()
        optimizer_critics.step()
        
        # =====================================================
        #                 TEACHER UPDATE STEP
        # =====================================================
        # The teacher learns to fool the *current* critics.
        # We freeze the critics' weights during this phase.
        teacher_policy.train()
        for critic in [critics[1], critics[2]]:
            critic.eval()
        
        optimizer_teacher.zero_grad()

        # Teacher's forward pass
        policy_vector = teacher_policy(concept_embeddings)

        # Get fragility scores from the critics (no gradients needed for critics here)
        with torch.no_grad():
            chosen_concepts_for_l0 = {c for i, c in enumerate(concepts) if policy_vector[i] > 0.5}
            fragility_l0 = critics[0].get_fragility(chosen_concepts_for_l0, ideal_concepts)
        
        soft_embedding_for_teacher = torch.matmul(policy_vector, concept_embeddings)
        fragility_l1 = critics[1](soft_embedding_for_teacher).squeeze()
        fragility_l2 = critics[2](soft_embedding_for_teacher).squeeze()
        
        # The teacher is penalized based on the fragility of the current problem's level
        if level_data.level == 0:
            fragility_penalty = torch.tensor(fragility_l0)
        elif level_data.level == 1:
            fragility_penalty = fragility_l1
        else: # level 2
            fragility_penalty = fragility_l2
        
        # Teacher's Loss
        bce_loss = F.binary_cross_entropy(policy_vector, target_vector)
        teacher_loss = bce_loss + fragility_penalty

        teacher_loss.backward()
        optimizer_teacher.step()

        # --- E. Logging ---
        history['teacher_loss'].append(teacher_loss.item())
        history['critic_loss'].append(total_critic_loss.item())
        for i in range(3):
            key = f'f1_score_l{i}'
            if i == level_data.level:
                history[key].append(f1_score)
            else:
                history[key].append(np.nan)

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Level: {level_data.level} | T_Loss: {teacher_loss.item():.4f} | C_Loss: {total_critic_loss.item():.4f} | F1: {f1_score:.3f}")

    print("Training complete.")

    # --- 4. Visualize Results ---
    fig, axs = plt.subplots(1, 2, figsize=(18, 6), dpi=100)
    
    axs[0].plot(pd.Series(history['teacher_loss']).rolling(20, min_periods=1).mean(), label="Teacher Loss")
    axs[0].plot(pd.Series(history['critic_loss']).rolling(20, min_periods=1).mean(), label="Critic Loss")
    axs[0].set_title("Loss Curves (20-epoch MA)", size=16)
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].grid(True, linestyle='--', alpha=0.6)
    axs[0].legend()

    for i in range(3):
        key = f'f1_score_l{i}'
        s = pd.Series(history[key])
        axs[1].plot(s.rolling(40, min_periods=1).mean(), label=f'Level {i} F1 (MA)')

    axs[1].set_title("F1 Score per Abstraction Level (40-epoch MA)", size=16)
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("F1 Score")
    axs[1].grid(True, linestyle='--', alpha=0.6)
    axs[1].legend()

    plt.tight_layout()
    plt.savefig("results/hal_training_curves.png")
    print("HAL training curves saved to results/hal_training_curves.png")


if __name__ == "__main__":
    run_hal_training()