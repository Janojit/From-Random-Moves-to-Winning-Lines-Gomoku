import torch
import numpy as np

from env.gomoku_env import GomokuEnv
from agents.dqn import DQNAgent
from agents.double_dqn import DoubleDQNAgent
from agents.ppo import PPOAgent
from agents.a2c import A2CAgent
from agents.reinforce import REINFORCEAgent

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

EPISODES = 100

def evaluate(agent, agent_type):
    env = GomokuEnv()
    wins = losses = draws = 0

    for _ in range(EPISODES):
        state, _ = env.reset()
        done = False

        while not done:
            with torch.no_grad():
                if agent_type == "DQN":
                    action = agent.act(state, env.board, eps=0.0)

                elif agent_type == "Double-DQN":
                    action = agent.act(state, env.board)

                elif agent_type in ["PPO", "A2C"]:
                    action, _, _ = agent.act(state, env.board)

                elif agent_type == "REINFORCE":
                    action = agent.act(state, env.board)

                else:
                    raise ValueError("Unknown agent type")

            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

        if reward == 1:
            wins += 1
        elif reward == -1:
            losses += 1
        else:
            draws += 1

    return wins, losses, draws


if __name__ == "__main__":
    results = {}

    # -------- DQN --------
    dqn = DQNAgent()
    dqn.q.to(DEVICE)
    dqn.target.to(DEVICE)
    dqn.q.load_state_dict(
        torch.load("models/dqn.pth", map_location=DEVICE, weights_only=True)
    )
    results["DQN"] = evaluate(dqn, "DQN")

    # ----- Double-DQN -----
    ddqn = DoubleDQNAgent()
    ddqn.q.to(DEVICE)
    ddqn.target.to(DEVICE)
    ddqn.q.load_state_dict(
        torch.load("models/double_dqn.pth", map_location=DEVICE, weights_only=True)
    )
    results["Double-DQN"] = evaluate(ddqn, "Double-DQN")

    # -------- PPO --------
    ppo = PPOAgent().to(DEVICE)
    ppo.load_state_dict(
        torch.load("models/ppo.pth", map_location=DEVICE, weights_only=True)
    )
    results["PPO"] = evaluate(ppo, "PPO")

    # -------- A2C --------
    a2c = A2CAgent().to(DEVICE)
    a2c.load_state_dict(
        torch.load("models/a2c.pth", map_location=DEVICE, weights_only=True)
    )
    results["A2C"] = evaluate(a2c, "A2C")

    # ----- REINFORCE -----
    reinforce = REINFORCEAgent()
    reinforce.policy.to(DEVICE)
    reinforce.policy.load_state_dict(
        torch.load("models/reinforce.pth", map_location=DEVICE, weights_only=True)
    )
    results["REINFORCE"] = evaluate(reinforce, "REINFORCE")

    np.save("eval/eval_results.npy", results)
    print("Evaluation completed.")
    print(results)
