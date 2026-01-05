"""
train.py
Run with:
    python -m train
"""

import os
import torch

# ---------------- DEVICE ----------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# ---------------- IMPORT ENV & AGENT ----------------
from env import GomokuEnv
from agent import PGAgent

# ---------------- GOOGLE DRIVE PATH ----------------
DRIVE_SAVE_PATH = "/content/drive/MyDrive/Gomoku_Models"
os.makedirs(DRIVE_SAVE_PATH, exist_ok=True)

# ---------------- TRAINING CONFIG ----------------
EPISODES = 50000
SAVE_EVERY = 10000

# ---------------- ENV & AGENT ----------------
env = GomokuEnv()
agent = PGAgent()
agent.model.to(DEVICE)

# ---------------- TRAIN LOOP ----------------
for ep in range(1, EPISODES + 1):
    state = env.reset()
    done = False

    while not done:
        # Convert state to tensor if needed
        if not torch.is_tensor(state):
            state_tensor = torch.tensor(state, dtype=torch.float32, device=DEVICE)
        else:
            state_tensor = state.to(DEVICE)

        action, _ = agent.act(state_tensor)

        state, reward, done = env.step(action)
        agent.rewards.append(reward)

    agent.learn()

    # -------- SAVE CHECKPOINT --------
    if ep % SAVE_EVERY == 0:
        ckpt_path = os.path.join(
            DRIVE_SAVE_PATH, f"gomoku_policy_ep{ep}.pt"
        )
        torch.save(agent.model.state_dict(), ckpt_path)
        print(f"[Checkpoint saved] Episode {ep} → {ckpt_path}")

    # -------- LOG --------
    if ep % 500 == 0:
        print(f"Episode {ep} finished")

# ---------------- FINAL SAVE ----------------
final_path = os.path.join(DRIVE_SAVE_PATH, "gomoku_policy_final.pt")
torch.save(agent.model.state_dict(), final_path)
print("Training complete & final model saved at:", final_path)
