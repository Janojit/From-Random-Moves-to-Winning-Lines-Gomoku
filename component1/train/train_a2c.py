import torch
from env.gomoku_env import GomokuEnv
from agents.a2c import A2CAgent

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

env = GomokuEnv()
agent = A2CAgent().to(DEVICE)

EPISODES = 300000

for ep in range(EPISODES):
    state, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        action, logp, value = agent.act(state, env.board)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        with torch.no_grad():
            next_value = agent.get_value(next_state)

        agent.update(logp, value, reward, next_value, done)
        state = next_state
        total_reward += reward

    if ep % 1000 == 0:
        print(f"A2C | Episode {ep} | Reward {total_reward:.2f}")

torch.save(agent.state_dict(), "models/a2c.pth")
print("A2C model saved.")
