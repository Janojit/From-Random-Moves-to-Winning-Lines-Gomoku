import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.helpers import masked_softmax
from utils.networks import device

class A2CAgent(nn.Module):
    def __init__(self, lr=1e-4, gamma=0.99):
        super().__init__()
        self.gamma = gamma

        self.shared = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 9 * 9, 256),
            nn.ReLU()
        )
        self.policy = nn.Linear(256, 81)
        self.value_head = nn.Linear(256, 1)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, state):
        x = self.shared(state)
        return x

    def act(self, state, board):
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        x = self.forward(state)

        logits = self.policy(x)[0]
        probs = masked_softmax(logits, board)

        dist = torch.distributions.Categorical(probs)
        action = dist.sample()

        log_prob = dist.log_prob(action)
        value = self.value_head(x).squeeze(0)

        return action.item(), log_prob, value

    def get_value(self, state):
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        x = self.forward(state)
        return self.value_head(x).squeeze(0)

    def update(self, log_prob, value, reward, next_value, done):
        target = reward + self.gamma * next_value * (1 - done)
        advantage = target - value

        policy_loss = -log_prob * advantage.detach()
        value_loss = advantage.pow(2)

        loss = policy_loss + 0.5 * value_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
