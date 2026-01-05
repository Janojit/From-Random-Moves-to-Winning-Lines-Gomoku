import gymnasium as gym
import numpy as np
from gymnasium import spaces

BOARD_SIZE = 9

class GomokuEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)

        self.action_space = spaces.Discrete(BOARD_SIZE * BOARD_SIZE)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(2, BOARD_SIZE, BOARD_SIZE), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board[:] = 0
        return self._get_obs(), {}

    def _get_obs(self):
        obs = np.zeros((2, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        obs[0] = (self.board == 1)
        obs[1] = (self.board == -1)
        return obs

    def legal_actions(self):
        return np.where(self.board.flatten() == 0)[0]

    def step(self, action):
        if action not in self.legal_actions():
            return self._get_obs(), -1.0, True, False, {}

        r, c = divmod(action, BOARD_SIZE)
        self.board[r, c] = 1

        if self._check_win(1):
            return self._get_obs(), 1.0, True, False, {}

        if np.all(self.board != 0):
            return self._get_obs(), 0.0, True, False, {}

        self.board *= -1
        return self._get_obs(), 0.01, False, False, {}

    def _check_win(self, p):
        b = self.board == p
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if j+4 < BOARD_SIZE and b[i, j:j+5].all(): return True
                if i+4 < BOARD_SIZE and b[i:i+5, j].all(): return True
                if i+4 < BOARD_SIZE and j+4 < BOARD_SIZE and all(b[i+k, j+k] for k in range(5)): return True
                if i+4 < BOARD_SIZE and j-4 >= 0 and all(b[i+k, j-k] for k in range(5)): return True
        return False
