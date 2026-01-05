import torch
from env.gomoku_env import GomokuEnv
from agents.dqn import DQNAgent
from utils.replay_buffer import ReplayBuffer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

env = GomokuEnv()
agent = DQNAgent()
agent.q.to(DEVICE)
agent.target.to(DEVICE)
buffer = ReplayBuffer()

EPISODES = 300000
BATCH = 128
EPS = 1.0

for ep in range(EPISODES):
    state, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = agent.act(state, env.board, EPS)

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        buffer.push(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        if len(buffer) > BATCH:
            s, a, r, ns, d = buffer.sample(BATCH)
            agent.update(s, a, r, ns, d)

    EPS = max(0.05, EPS * 0.9995)

    if ep % 1000 == 0:
        agent.sync()
        print(f"DQN | Ep {ep} | Reward {total_reward:.2f}")

torch.save(agent.q.state_dict(), "models/dqn.pth")
print("DQN model saved.")
