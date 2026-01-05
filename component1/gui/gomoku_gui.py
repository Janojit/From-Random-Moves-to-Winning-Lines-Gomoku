import pygame
import sys
import torch
import numpy as np

from env.gomoku_env import GomokuEnv
from agents.dqn import DQNAgent
from agents.double_dqn import DoubleDQNAgent
from agents.ppo import PPOAgent
from agents.a2c import A2CAgent
from agents.reinforce import REINFORCEAgent

# ================= DEVICE =================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pygame.init()

# ================= CONFIG =================
BOARD_SIZE = 9
CELL_SIZE = 60
MARGIN = 40
WIDTH = HEIGHT = BOARD_SIZE * CELL_SIZE + 2 * MARGIN

FONT = pygame.font.SysFont("arial", 22)
BIG_FONT = pygame.font.SysFont("arial", 32)

WHITE = (245, 245, 245)
BLACK = (20, 20, 20)
RED = (200, 50, 50)
BLUE = (50, 50, 200)

screen = pygame.display.set_mode((WIDTH, HEIGHT + 180))
pygame.display.set_caption("Gomoku DRL Arena")

# ================= AGENT LOADER =================
def load_agent(name):
    if name == "DQN":
        agent = DQNAgent()
        agent.q.to(DEVICE)
        agent.target.to(DEVICE)
        agent.q.load_state_dict(torch.load("models/dqn.pth", map_location=DEVICE))
        agent.eval = lambda: None
        return agent

    if name == "Double-DQN":
        agent = DoubleDQNAgent()
        agent.q.to(DEVICE)
        agent.target.to(DEVICE)
        agent.q.load_state_dict(torch.load("models/double_dqn.pth", map_location=DEVICE))
        agent.eval = lambda: None
        return agent

    if name == "PPO":
        agent = PPOAgent().to(DEVICE)
        agent.load_state_dict(torch.load("models/ppo.pth", map_location=DEVICE))
        agent.eval()
        return agent

    if name == "A2C":
        agent = A2CAgent().to(DEVICE)
        agent.load_state_dict(torch.load("models/a2c.pth", map_location=DEVICE))
        agent.eval()
        return agent

    if name == "REINFORCE":
        agent = REINFORCEAgent()
        agent.policy.to(DEVICE)
        agent.policy.load_state_dict(torch.load("models/reinforce.pth", map_location=DEVICE))
        agent.eval = lambda: None
        return agent

# ================= HELPERS =================
def is_legal_move(board, action):
    r = action // BOARD_SIZE
    c = action % BOARD_SIZE
    return 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and board[r, c] == 0

def normalize_action(action):
    if isinstance(action, tuple):
        action = action[0]
    if torch.is_tensor(action):
        action = action.item()
    return int(action)

def get_ai_action(agent, state, board):
    try:
        return agent.act(state, board, 0.0)
    except TypeError:
        return agent.act(state, board)

def get_player_description(pid, player_type, player_agent):
    label = "Player 1" if pid == 1 else "Player 2"
    if player_type[pid] == "human":
        return f"{label} (Human)"
    agent_name = player_agent[pid].__class__.__name__.replace("Agent", "")
    return f"{label} (AI – {agent_name})"

def draw_text(text, y, color=BLACK):
    label = FONT.render(text, True, color)
    screen.blit(label, (20, y))

def draw_board(board):
    screen.fill(WHITE)
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            rect = pygame.Rect(
                MARGIN + j * CELL_SIZE,
                MARGIN + i * CELL_SIZE,
                CELL_SIZE,
                CELL_SIZE
            )
            pygame.draw.rect(screen, BLACK, rect, 1)
            if board[i, j] == 1:
                pygame.draw.circle(screen, BLACK, rect.center, CELL_SIZE // 2 - 4)
            elif board[i, j] == -1:
                pygame.draw.circle(screen, RED, rect.center, CELL_SIZE // 2 - 4)

# ================= MAIN =================
def main():
    env = GomokuEnv()
    state, _ = env.reset()

    setup_stage = 0
    player_type = {}
    player_agent = {}

    current_player = 1
    game_over = False
    winner_text = ""

    ai_waiting = False
    ai_start_time = 0
    AI_DELAY_MS = 10

    clock = pygame.time.Clock()
    agent_names = ["DQN", "Double-DQN", "PPO", "A2C", "REINFORCE"]

    while True:
        draw_board(env.board)

        # ---------- UI ----------
        if setup_stage < 4:
            draw_text("PLAYER SETUP", HEIGHT + 10)
            if setup_stage == 0:
                draw_text("Player 1: H = Human | A = AI", HEIGHT + 40)
            elif setup_stage == 1:
                draw_text("Select AI for Player 1:", HEIGHT + 40)
                for i, name in enumerate(agent_names):
                    draw_text(f"{i+1}: {name}", HEIGHT + 70 + i * 25)
            elif setup_stage == 2:
                draw_text("Player 2: H = Human | A = AI", HEIGHT + 40)
            elif setup_stage == 3:
                draw_text("Select AI for Player 2:", HEIGHT + 40)
                for i, name in enumerate(agent_names):
                    draw_text(f"{i+1}: {name}", HEIGHT + 70 + i * 25)
        else:
            draw_text("Game Running", HEIGHT + 10)
            if player_type[current_player] == "ai" and ai_waiting:
                draw_text("AI is thinking...", HEIGHT + 40, BLUE)
            if game_over:
                msg = BIG_FONT.render(winner_text, True, BLUE)
                screen.blit(msg, (WIDTH // 2 - 180, HEIGHT + 80))

        pygame.display.flip()
        clock.tick(30)

        # ---------- EVENTS ----------
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            # Setup input
            if setup_stage < 4 and event.type == pygame.KEYDOWN:
                if setup_stage in [0, 2]:
                    p = 1 if setup_stage == 0 else -1
                    if event.key == pygame.K_h:
                        player_type[p] = "human"
                        setup_stage += 2
                    elif event.key == pygame.K_a:
                        player_type[p] = "ai"
                        setup_stage += 1
                elif setup_stage in [1, 3]:
                    p = 1 if setup_stage == 1 else -1
                    idx = event.key - pygame.K_1
                    if 0 <= idx < len(agent_names):
                        player_agent[p] = load_agent(agent_names[idx])
                        setup_stage += 1

            # Human move
            if setup_stage >= 4 and not game_over:
                if player_type[current_player] == "human" and event.type == pygame.MOUSEBUTTONDOWN:
                    x, y = event.pos
                    r = (y - MARGIN) // CELL_SIZE
                    c = (x - MARGIN) // CELL_SIZE
                    action = r * BOARD_SIZE + c
                    if is_legal_move(env.board, action):
                        state, reward, terminated, truncated, _ = env.step(action)
                        game_over = terminated or truncated
                        current_player *= -1
                        ai_waiting = False

                        if game_over:
                            if reward != 0:
                                winner_pid = -current_player
                                winner_text = f"{get_player_description(winner_pid, player_type, player_agent)} Wins!"
                            else:
                                winner_text = "Draw"

        # AI move
        if setup_stage >= 4 and not game_over and player_type[current_player] == "ai":
            if not ai_waiting:
                ai_waiting = True
                ai_start_time = pygame.time.get_ticks()
            elif pygame.time.get_ticks() - ai_start_time >= AI_DELAY_MS:
                agent = player_agent[current_player]
                with torch.no_grad():
                    while True:
                        action = normalize_action(get_ai_action(agent, state, env.board))
                        if is_legal_move(env.board, action):
                            break
                state, reward, terminated, truncated, _ = env.step(action)
                game_over = terminated or truncated
                current_player *= -1
                ai_waiting = False

                if game_over:
                    if reward != 0:
                        winner_pid = -current_player
                        winner_text = f"{get_player_description(winner_pid, player_type, player_agent)} Wins!"
                    else:
                        winner_text = "Draw"

if __name__ == "__main__":
    main()
