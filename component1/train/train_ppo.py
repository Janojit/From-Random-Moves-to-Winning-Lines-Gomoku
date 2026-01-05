import torch
from env.gomoku_env import GomokuEnv
from agents.ppo import PPOAgent

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

env = GomokuEnv()
agent = PPOAgent().to(DEVICE)

EPISODES = 300000

for ep in range(EPISODES):
    state, _ = env.reset()
    done = False

    states = []
    actions = []
    log_probs = []
    rewards = []
    values = []
    boards = []

    while not done:
        action, logp, value = agent.act(state, env.board)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        states.append(torch.tensor(state, dtype=torch.float32))
        actions.append(action)
        log_probs.append(logp)
        rewards.append(reward)
        values.append(value)
        boards.append(env.board.copy())

        state = next_state

    # Compute returns
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + agent.gamma * R
        returns.insert(0, R)

    returns = torch.tensor(returns, device=DEVICE)
    values = torch.stack(values).to(DEVICE)
    advantages = returns - values

    agent.update(
        states=torch.stack(states).to(DEVICE),
        actions=torch.tensor(actions, device=DEVICE),
        old_log_probs=torch.stack(log_probs).to(DEVICE),
        returns=returns,
        advantages=advantages,
        boards=boards
    )

    if ep % 500 == 0:
        print(f"PPO | Episode {ep}")

torch.save(agent.state_dict(), "models/ppo.pth")
print("PPO model saved.")
