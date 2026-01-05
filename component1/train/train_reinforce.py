import torch
from env.gomoku_env import GomokuEnv
from agents.reinforce import REINFORCEAgent

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

env = GomokuEnv()
agent = REINFORCEAgent()
agent.policy.to(DEVICE)

EPISODES = 300000

for ep in range(EPISODES):
    state, _ = env.reset()
    done = False
    rewards = []

    while not done:
        action = agent.act(state, env.board)
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        rewards.append(reward)

    agent.update(rewards)

    if ep % 1000 == 0:
        print(f"REINFORCE | Ep {ep}")

torch.save(agent.policy.state_dict(), "models/reinforce.pth")
print("REINFORCE model saved.")
