import random
import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.pos] = (state, action, reward, next_state, done)
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))

        return (
            torch.tensor(states, dtype=torch.float32).cuda(),
            torch.tensor(actions, dtype=torch.long).cuda(),
            torch.tensor(rewards, dtype=torch.float32).cuda(),
            torch.tensor(next_states, dtype=torch.float32).cuda(),
            torch.tensor(dones, dtype=torch.float32).cuda()
        )

    def __len__(self):
        return len(self.buffer)
