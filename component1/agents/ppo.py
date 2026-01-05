import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.helpers import masked_softmax
from utils.networks import device

class PPOAgent(nn.Module):
    def __init__(self, lr=3e-4, gamma=0.99, clip_eps=0.2):
        super().__init__()
        self.gamma = gamma
        self.clip_eps = clip_eps

        self.shared = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 9 * 9, 256),
            nn.ReLU()
        )
        self.policy = nn.Linear(256, 81)
        self.value = nn.Linear(256, 1)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def act(self, state, board):
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        x = self.shared(state)

        logits = self.policy(x)[0]
        probs = masked_softmax(logits, board)

        dist = torch.distributions.Categorical(probs)
        action = dist.sample()

        log_prob = dist.log_prob(action)
        value = self.value(x).squeeze(0)

        return action.item(), log_prob.detach(), value.detach()

    def evaluate_actions(self, states, actions, boards):
        x = self.shared(states)
        logits = self.policy(x)

        # Mask invalid actions per state
        probs = []
        for i in range(len(boards)):
            probs.append(masked_softmax(logits[i], boards[i]))
        probs = torch.stack(probs)

        dist = torch.distributions.Categorical(probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        values = self.value(x).squeeze(-1)
        return log_probs, entropy, values

    def update(self, states, actions, old_log_probs, returns, advantages, boards, epochs=4):
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(epochs):
            log_probs, entropy, values = self.evaluate_actions(states, actions, boards)

            ratio = torch.exp(log_probs - old_log_probs)

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = F.mse_loss(values, returns)

            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy.mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
