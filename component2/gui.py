import pygame
import numpy as np
import torch
from env import GomokuEnv
from agent import PGAgent

pygame.init()

# ---------------- CONFIG ----------------
SIZE = 9
CELL = 60
WIDTH = HEIGHT = SIZE * CELL + 80
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Explainable Gomoku")

font = pygame.font.SysFont(None, 36)

# ---------------- ENV & AGENT ----------------
env = GomokuEnv()
agent = PGAgent()
agent.model.load_state_dict(
    torch.load("checkpoints/gomoku_policy_ep50000.pt", map_location="cpu")
)
agent.model.eval()

state = env.reset()

winner_text = ""
heatmap = None
last_legal_actor = None  # "Human" or "AI"

# ---------------- DRAW FUNCTION ----------------
def draw(board, heatmap=None):
    screen.fill((240, 240, 240))

    for r in range(SIZE):
        for c in range(SIZE):
            rect = pygame.Rect(c * CELL, r * CELL, CELL, CELL)
            pygame.draw.rect(screen, (0, 0, 0), rect, 1)

            if heatmap is not None:
                val = heatmap[r, c]
                if val > 0:
                    overlay = pygame.Surface((CELL, CELL))
                    overlay.set_alpha(int(160 * val))
                    overlay.fill((255, 0, 0))
                    screen.blit(overlay, (c * CELL, r * CELL))

            if board[r, c] == 1:
                pygame.draw.circle(screen, (0, 0, 0), rect.center, 20)
            elif board[r, c] == -1:
                pygame.draw.circle(screen, (180, 0, 0), rect.center, 20)

    txt = font.render(winner_text, True, (0, 0, 0))
    screen.blit(txt, (10, HEIGHT - 60))
    pygame.display.flip()

# ---------------- MAIN LOOP ----------------
running = True

while running:
    draw(env.board, heatmap)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        # ---------- HUMAN MOVE ----------
        if event.type == pygame.MOUSEBUTTONDOWN and not env.done:
            x, y = pygame.mouse.get_pos()
            c, r = x // CELL, y // CELL

            if r < SIZE and c < SIZE:
                prev_board = env.board.copy()

                new_state, _, done = env.step(r * SIZE + c)

                # ---- ILLEGAL HUMAN MOVE → IGNORE ----
                if np.array_equal(prev_board, env.board):
                    continue  # human retries

                # ---- LEGAL HUMAN MOVE ----
                state = new_state
                last_legal_actor = "Human"
                heatmap = None
                winner_text = ""

                # ---------- AI MOVE LOOP ----------
                while not done:
                    prev_board = env.board.copy()

                    with torch.no_grad():
                        action, probs = agent.act(state)

                    new_state, _, done = env.step(action)

                    # ---- ILLEGAL AI MOVE → RETRY ----
                    if np.array_equal(prev_board, env.board):
                        continue  # AI retries same turn

                    # ---- LEGAL AI MOVE ----
                    state = new_state
                    last_legal_actor = "AI"

                    heatmap = probs.reshape(SIZE, SIZE)
                    heatmap = heatmap / (heatmap.max() + 1e-8)
                    break

                # ---------- GAME OVER ----------
                if done:
                    if env.winner == 1:
                        winner_text = f"{last_legal_actor} Won!"
                    else:
                        winner_text = "Draw!"

        # ---------- RESET ----------
        if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
            state = env.reset()
            winner_text = ""
            heatmap = None
            last_legal_actor = None

pygame.quit()
