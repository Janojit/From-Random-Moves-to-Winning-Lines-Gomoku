import numpy as np

class GomokuEnv:
    def __init__(self, size=9):
        self.size = size
        self.reset()

    def reset(self):
        self.board = np.zeros((self.size, self.size), dtype=np.int8)
        self.done = False
        self.winner = None
        return self.board.copy()

    def step(self, action):
        if self.done:
            return self.board.copy(), 0.0, True

        r, c = divmod(action, self.size)

        # -------- ILLEGAL MOVE --------
        if self.board[r, c] != 0:
            # Penalize but DO NOT end episode
            return self.board.copy(), -0.5, False

        # -------- VALID MOVE --------
        self.board[r, c] = 1

        # -------- WIN CHECK --------
        if self.check_win(1):
            self.done = True
            self.winner = 1
            return self.board.copy(), 1.0, True

        # -------- DRAW CHECK --------
        if np.all(self.board != 0):
            self.done = True
            self.winner = 0
            return self.board.copy(), 0.2, True

        # -------- SWITCH PLAYER (SYMMETRY) --------
        self.board *= -1
        return self.board.copy(), 0.0, False

    def check_win(self, player):
        b = (self.board == player)
        s = self.size

        for r in range(s):
            for c in range(s):
                if c <= s - 5 and np.all(b[r, c:c+5]):
                    return True
                if r <= s - 5 and np.all(b[r:r+5, c]):
                    return True
                if r <= s - 5 and c <= s - 5 and np.all([b[r+i, c+i] for i in range(5)]):
                    return True
                if r >= 4 and c <= s - 5 and np.all([b[r-i, c+i] for i in range(5)]):
                    return True
        return False
