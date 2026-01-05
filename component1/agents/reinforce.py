import torch
from utils.networks import CNN, device
from utils.helpers import masked_softmax

class REINFORCEAgent:
    def __init__(self):
        self.policy = CNN(81).to(device)
        self.optim = torch.optim.Adam(self.policy.parameters(), lr=1e-4)
        self.saved = []

    def act(self, state, board):
        logits = self.policy(torch.tensor(state).unsqueeze(0).to(device))[0]
        probs = masked_softmax(logits, board)
        dist = torch.distributions.Categorical(probs)
        a = dist.sample()
        self.saved.append(dist.log_prob(a))
        return a.item()

    def update(self, rewards):
        R = 0
        loss = []
        for logp, r in zip(reversed(self.saved), reversed(rewards)):
            R = r + 0.99 * R
            loss.append(-logp * R)
        loss = torch.stack(loss).sum()
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        self.saved.clear()
