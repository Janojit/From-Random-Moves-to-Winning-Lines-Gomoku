import torch
import torch.nn.functional as F
from utils.networks import CNN, device
from utils.helpers import mask_logits

class DQNAgent:
    def __init__(self, lr=1e-4):
        self.q = CNN(81).to(device)
        self.target = CNN(81).to(device)
        self.target.load_state_dict(self.q.state_dict())
        self.optim = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.gamma = 0.99

    def act(self, state, board, eps=0.1):
        if torch.rand(1).item() < eps:
            return int(torch.tensor(board.flatten() == 0).nonzero().flatten()[torch.randint(0, (board.flatten()==0).sum(), (1,))])

        with torch.no_grad():
            q = self.q(torch.tensor(state).unsqueeze(0).to(device))[0]
            q = mask_logits(q, board)
        return q.argmax().item()

    def update(self, s, a, r, ns, done):
        q = self.q(s).gather(1, a.unsqueeze(1)).squeeze()
        with torch.no_grad():
            nq = self.target(ns).max(1)[0]
        target = r + self.gamma * nq * (1 - done)
        loss = F.mse_loss(q, target)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

    def sync(self):
        self.target.load_state_dict(self.q.state_dict())
