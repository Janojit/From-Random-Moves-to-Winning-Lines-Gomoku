import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PolicyNet(nn.Module):
    def __init__(self, board_size=9):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(board_size * board_size, 128),
            nn.ReLU(),
            nn.Linear(128, board_size * board_size)
        )

    def forward(self, x):
        return self.net(x)

class PGAgent:
    def __init__(self, board_size=9, lr=1e-3):
        self.model = PolicyNet(board_size).to(DEVICE)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.log_probs = []
        self.rewards = []

    def act(self, state):
        state = torch.tensor(state.flatten(), dtype=torch.float32).unsqueeze(0).to(DEVICE)
        logits = self.model(state)
        probs = torch.softmax(logits, dim=1)

        action = torch.multinomial(probs, 1).item()
        self.log_probs.append(torch.log(probs[0, action]))

        return action, probs.detach().cpu().numpy().reshape(9, 9)

    def learn(self):
        R = 0
        loss = 0
        for log_prob, reward in zip(reversed(self.log_probs), reversed(self.rewards)):
            R = reward + 0.99 * R
            loss -= log_prob * R

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.log_probs.clear()
        self.rewards.clear()
