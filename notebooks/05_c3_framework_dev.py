import json
import torch
import torch.optim as optim
import numpy as np
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import pandas as pd

import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- Local Imports ---
from src.qce.hc import DifferentiablePolicy
from eidetic_ai.c3.causal_graph import get_kinematics_causal_graph
from eidetic_ai.c3.student_model import SimulatedStudent

# --- Configuration ---
CONCEPT_SPACE_PATH = "data/concept_spaces/physics_kinematics.json"
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
NUM_EPOCHS = 400 # Causal learning is harder, needs more epochs
LEARNING_RATE = 0.001
# Weights for the combined reward function
ALPHA_IG = 0.7 # Weight for Information Gain (causal improvement)
BETA_NLG = 0.3 # Weight for Normalized Learning Gain (knowledge improvement)

def run_c3_training():
    """
    Executes a training loop for the Causal Contrastive Curriculum (C³) framework.
    """
    print("--- Eidetic AI: C³ Framework Training ---")

    # --- 1. Setup Environment ---
    print("Setting up environment...")
    with open(CONCEPT_SPACE_PATH, 'r') as f:
        concepts = json.load(f)['concepts']
    
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    concept_embeddings = torch.tensor(embedding_model.encode(concepts)).float()
    embedding_dim = concept_embeddings.shape[1]
    num_concepts = len(concepts)

    ground_truth_graph = get_kinematics_causal_graph()

    # --- 2. Initialize Models ---
    print("Initializing Teacher and Simulated Student...")
    teacher_policy = DifferentiablePolicy(embedding_dim, num_concepts)
    optimizer = optim.Adam(teacher_policy.parameters(), lr=LEARNING_RATE)
    
    # We create a new student for each epoch to get a clean measure of progress
    # In a real system, the student would be persistent.
    
    history = {'total_reward': [], 'causal_score': [], 'knowledge_gain': []}
    reward_baseline = 0.0
    baseline_alpha = 0.95

    # --- 3. The Training Loop ---
    print(f"Starting training for {NUM_EPOCHS} epochs...")
    for epoch in range(NUM_EPOCHS):
        
        # --- A. Initialize a new "confused" student ---
        student = SimulatedStudent(concepts, ground_truth_graph)
        initial_causal_score = student.get_causal_identifiability_score()
        initial_knowledge = torch.sum(student.knowledge_vector)

        # --- B. Teacher Proposes a Policy (an "intervention") ---
        teacher_policy.train()
        policy_vector = teacher_policy(concept_embeddings)
        
        # --- C. Student Receives the Lesson ---
        # Discretize policy to get the set of concepts for the lesson
        chosen_concepts = {c for i, c in enumerate(concepts) if policy_vector[i] > 0.5}
        student.receive_lesson(chosen_concepts)

        # --- D. Calculate Causal-Informed Reward ---
        # Reward 1: Information Gain (IG), approximated by the change in causal score
        final_causal_score = student.get_causal_identifiability_score()
        reward_ig = final_causal_score - initial_causal_score
        
        # Reward 2: Normalized Learning Gain (NLG)
        final_knowledge = torch.sum(student.knowledge_vector)
        knowledge_gain = (final_knowledge - initial_knowledge).item()
        max_possible_gain = num_concepts - initial_knowledge
        reward_nlg = knowledge_gain / max_possible_gain if max_possible_gain > 0 else 0.0

        # Total reward is a weighted sum
        total_reward = ALPHA_IG * reward_ig + BETA_NLG * reward_nlg
        
        # --- E. Teacher Update Step ---
        advantage = total_reward - reward_baseline
        reward_baseline = baseline_alpha * reward_baseline + (1 - baseline_alpha) * total_reward
        
        # We need a loss function. We'll use a simple REINFORCE-style loss where
        # the "action" is the continuous policy vector itself.
        # We want to push the policy vector in directions that yield high reward.
        # Loss = - (sum of policy probabilities) * advantage
        policy_sum_prob = torch.sum(policy_vector)
        teacher_loss = -policy_sum_prob * advantage
        
        optimizer.zero_grad()
        teacher_loss.backward()
        optimizer.step()
        
        # --- F. Logging ---
        history['total_reward'].append(total_reward)
        history['causal_score'].append(final_causal_score)
        history['knowledge_gain'].append(reward_nlg)

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Avg Reward: {np.mean(history['total_reward'][-20:]):.4f} | Causal Score: {final_causal_score:.3f}")

    print("Training complete.")

    # --- 4. Visualize Results ---
    fig, axs = plt.subplots(1, 2, figsize=(18, 6), dpi=100)
    
    reward_ma = pd.Series(history['total_reward']).rolling(20, min_periods=1).mean()
    axs[0].plot(reward_ma)
    axs[0].set_title("Total Reward (Causal + Knowledge) (20-epoch MA)", size=16)
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Weighted Reward")
    axs[0].grid(True, linestyle='--', alpha=0.6)

    causal_ma = pd.Series(history['causal_score']).rolling(20, min_periods=1).mean()
    knowledge_ma = pd.Series(history['knowledge_gain']).rolling(20, min_periods=1).mean()
    axs[1].plot(causal_ma, label='Causal Score (MA)')
    axs[1].plot(knowledge_ma, label='Knowledge Gain (MA)')
    axs[1].set_title("Student Improvement Metrics (20-epoch MA)", size=16)
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Score")
    axs[1].grid(True, linestyle='--', alpha=0.6)
    axs[1].legend()

    plt.tight_layout()
    plt.savefig("results/c3_training_curves.png")
    print("C³ training curves saved to results/c3_training_curves.png")

    # Plot the final student graph from the last epoch
    student.internal_graph.plot(save_path="results/final_student_causal_graph.png")

if __name__ == "__main__":
    run_c3_training()