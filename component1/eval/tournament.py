import torch
import numpy as np
import random
from itertools import combinations

from env.gomoku_env import GomokuEnv
from agents.dqn import DQNAgent
from agents.double_dqn import DoubleDQNAgent
from agents.ppo import PPOAgent
from agents.a2c import A2CAgent
from agents.reinforce import REINFORCEAgent

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

GAMES_PER_PAIR = 4   # increase later if needed

# --------------------------------------------------
# Agent Loader
# --------------------------------------------------
def load_agent(name):
    if name == "DQN":
        a = DQNAgent()
        a.q.to(DEVICE)
        a.target.to(DEVICE)
        a.q.load_state_dict(torch.load("models/dqn.pth", map_location=DEVICE, weights_only=True))
        return a

    if name == "Double-DQN":
        a = DoubleDQNAgent()
        a.q.to(DEVICE)
        a.target.to(DEVICE)
        a.q.load_state_dict(torch.load("models/double_dqn.pth", map_location=DEVICE, weights_only=True))
        return a

    if name == "PPO":
        a = PPOAgent().to(DEVICE)
        a.load_state_dict(torch.load("models/ppo.pth", map_location=DEVICE, weights_only=True))
        return a

    if name == "A2C":
        a = A2CAgent().to(DEVICE)
        a.load_state_dict(torch.load("models/a2c.pth", map_location=DEVICE, weights_only=True))
        return a

    if name == "REINFORCE":
        a = REINFORCEAgent()
        a.policy.to(DEVICE)
        a.policy.load_state_dict(torch.load("models/reinforce.pth", map_location=DEVICE, weights_only=True))
        return a

    raise ValueError("Unknown agent")


# --------------------------------------------------
# Action Selector
# --------------------------------------------------
def select_action(agent, name, state, board):
    with torch.no_grad():
        if name == "DQN":
            return agent.act(state, board, eps=0.0)
        elif name == "Double-DQN":
            return agent.act(state, board)
        elif name in ["PPO", "A2C"]:
            return agent.act(state, board)[0]
        elif name == "REINFORCE":
            return agent.act(state, board)
        else:
            raise ValueError("Unknown agent type")


# --------------------------------------------------
# Single Match
# --------------------------------------------------
def play_match(agent_A, name_A, agent_B, name_B):
    env = GomokuEnv()
    state, _ = env.reset()
    done = False
    current_player = 1  # A starts as black

    while not done:
        if current_player == 1:
            action = select_action(agent_A, name_A, state, env.board)
        else:
            action = select_action(agent_B, name_B, state, env.board)

        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        current_player *= -1

    return reward  # +1: black win, -1: red win, 0: draw


# --------------------------------------------------
# TOURNAMENT
# --------------------------------------------------
if __name__ == "__main__":
    agent_names = ["DQN", "Double-DQN", "PPO", "A2C", "REINFORCE"]
    agents = {name: load_agent(name) for name in agent_names}

    results = {name: {"win": 0, "loss": 0, "draw": 0} for name in agent_names}

    for A, B in combinations(agent_names, 2):
        print(f"\n{A} vs {B}")

        agent_A = agents[A]
        agent_B = agents[B]

        for g in range(GAMES_PER_PAIR):
            # A starts
            r = play_match(agent_A, A, agent_B, B)
            if r == 1:
                results[A]["win"] += 1
                results[B]["loss"] += 1
            elif r == -1:
                results[B]["win"] += 1
                results[A]["loss"] += 1
            else:
                results[A]["draw"] += 1
                results[B]["draw"] += 1

            # B starts
            r = play_match(agent_B, B, agent_A, A)
            if r == 1:
                results[B]["win"] += 1
                results[A]["loss"] += 1
            elif r == -1:
                results[A]["win"] += 1
                results[B]["loss"] += 1
            else:
                results[A]["draw"] += 1
                results[B]["draw"] += 1

    # --------------------------------------------------
    # SAVE & DISPLAY
    # --------------------------------------------------
    np.save("eval/tournament_results.npy", results)

    print("\n===== TOURNAMENT RESULTS =====")
    for name, r in results.items():
        print(f"{name:12s} | W: {r['win']:3d} | L: {r['loss']:3d} | D: {r['draw']:3d}")
