import torch
import torch.nn.functional as F
from utils.networks import CNN, device
from utils.helpers import mask_logits

class DoubleDQNAgent:
    def __init__(self, lr=1e-4):
        self.q = CNN(81).to(device)
        self.target = CNN(81).to(device)
        self.target.load_state_dict(self.q.state_dict())
        self.optim = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.gamma = 0.99

    def act(self, state, board):
        with torch.no_grad():
            q = self.q(torch.tensor(state).unsqueeze(0).to(device))[0]
            q = mask_logits(q, board)
        return q.argmax().item()

    def update(self, s, a, r, ns, done):
        q = self.q(s).gather(1, a.unsqueeze(1)).squeeze()
        with torch.no_grad():
            next_a = self.q(ns).argmax(1)
            next_q = self.target(ns).gather(1, next_a.unsqueeze(1)).squeeze()
        target = r + self.gamma * next_q * (1 - done)
        loss = F.mse_loss(q, target)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

    def sync(self):
        self.target.load_state_dict(self.q.state_dict())
