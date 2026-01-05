import torch
from env.gomoku_env import GomokuEnv
from agents.double_dqn import DoubleDQNAgent
from utils.replay_buffer import ReplayBuffer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

env = GomokuEnv()
agent = DoubleDQNAgent()
agent.q.to(DEVICE)
agent.target.to(DEVICE)
buffer = ReplayBuffer()

EPISODES = 300000
BATCH = 128

for ep in range(EPISODES):
    state, _ = env.reset()
    done = False

    while not done:
        action = agent.act(state, env.board)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        buffer.push(state, action, reward, next_state, done)
        state = next_state

        if len(buffer) > BATCH:
            s, a, r, ns, d = buffer.sample(BATCH)
            agent.update(s, a, r, ns, d)

    if ep % 2000 == 0:
        agent.sync()
        print(f"Double-DQN | Ep {ep}")

torch.save(agent.q.state_dict(), "models/double_dqn.pth")
print("Double-DQN model saved.")
