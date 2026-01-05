# 🧠 From Random Moves to Winning Lines

## Deep Reinforcement Learning for Gomoku

**Team Name:** NashCraft
**Program:** MSc Big Data Analytics, RKMVERI

**Team Members:**

* **Janojit Chakraborty** (B2430050)
* **Radheshyam Routh** (B2430053)

---

## 📌 Project Overview

This project investigates how **Deep Reinforcement Learning (DRL)** combined with **self-play** can learn strategic behavior in a **competitive, zero-sum board game**—**Gomoku (Five-in-a-Row)**.

The repository is organized into **two distinct components**, each addressing different academic objectives:

### 🔹 Component-1: Algorithmic Comparison

* Gym-style Gomoku environment
* Multiple DRL algorithms trained via self-play
* Quantitative comparison of learning stability and performance

### 🔹 Component-2: Custom Environment & Explainable AI

* Gomoku environment built from scratch (no Gym dependency)
* Policy-gradient agent trained offline
* Interactive GUI with **explainable decision heatmaps**

---

## 🗂️ Repository Structure

```
From-Random-Moves-to-Winning-Lines-Gomoku/
│
├── component1/                 # Algorithm comparison (Gym-style)
│   ├── env/                    # Gomoku Gym environment
│   ├── agents/                 # DRL agents (DQN, PPO, A2C, etc.)
│   ├── train/                  # Training scripts (self-play)
│   ├── eval/                   # Evaluation & comparison tools
│   ├── gui/                    # Gomoku GUI
│   ├── utils/                  # Shared utilities
│   └── requirements.txt
│
├── component2/                 # Custom explainable Gomoku system
│   ├── env.py                  # Custom environment
│   ├── agent.py                # Policy Gradient agent
│   ├── train.py                # Offline self-play training
│   ├── gui.py                  # Explainable GUI (heatmaps)
│   └── requirements.txt
│
└── README.md
```

---

# ⚙️ Installation & Setup

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/Janojit/From-Random-Moves-to-Winning-Lines-Gomoku.git
cd From-Random-Moves-to-Winning-Lines-Gomoku
```

---

# 🚀 COMPONENT 1

## DRL Algorithm Comparison in Gomoku

### 🎯 Objective

To **compare multiple DRL algorithms** in a controlled, self-play Gomoku environment and analyze:

* Convergence behavior
* Stability
* Strategic emergence

---

## 🔧 Setup (Component-1)

```bash
cd component1
python -m venv gomoku_env
source gomoku_env/bin/activate     # Linux / macOS
gomoku_env\Scripts\activate        # Windows
pip install -r requirements.txt
```

> GPU acceleration is automatically enabled if CUDA is available.

---

## 🧪 Verify the Environment

```bash
python -m env.gomoku_env
```

Ensure:

* Board initializes correctly
* Moves are validated
* Win detection functions properly

---

## 🏋️ Training the Agents (Self-Play)

Run the following scripts **one at a time**:

```bash
python -m train.train_dqn
python -m train.train_double_dqn
python -m train.train_reinforce
python -m train.train_a2c
python -m train.train_ppo
```

📌 Trained models are saved as `.pth` files.

---

## 📊 Evaluation & Comparison

### Run Evaluation

```bash
python -m eval.evaluate
```

### Generate Comparison Plots

```bash
python -m eval.compare_models
```

Generated plots:

* `win_count.png`
* `loss_count.png`
* `draw_count.png`

### Tournament Analysis

```bash
python -m eval.tournament
```

---

## 🤖 Algorithms Used

| Algorithm  | Category     | Purpose                |
| ---------- | ------------ | ---------------------- |
| DQN        | Value-based  | Baseline               |
| Double-DQN | Value-based  | Bias reduction         |
| REINFORCE  | Policy-based | Monte-Carlo            |
| A2C        | Actor-Critic | Faster learning        |
| PPO        | Policy-based | Stability & robustness |

---

## 📈 Key Findings (Component-1)

* Value-based methods struggle under non-stationary self-play
* Policy-gradient methods are more stable
* PPO shows the best convergence behavior
* Strategic play emerges without heuristics

---

# 🎮 COMPONENT 2

## Custom Explainable Gomoku Environment

### 🎯 Objective

To design a **novel, interpretable Gomoku system**, emphasizing:

* Environment design
* Explainability
* Human-AI interaction

---

## ✨ Features

* Custom Gomoku environment (no Gym)
* Policy-Gradient agent trained via self-play
* GPU-accelerated training
* Interactive GUI using Pygame
* **Move-probability heatmaps for explainability**
* Explicit win / loss / draw feedback

---

## 🔧 Setup (Component-2)

```bash
cd ..
cd component2
python -m venv gomoku_env
gomoku_env\Scripts\activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy pygame
```

(Optional GPU check)

```python
import torch
print(torch.cuda.is_available())
```

---

## 🏋️ Training the Agent

```bash
python train.py
```

* Training time: ~3–10 minutes (GPU)
* Output model: `gomoku_policy.pt`

---

## 🖥️ Running the GUI

```bash
python gui.py
```

### Controls

* **Mouse Click:** Place move
* **R key:** Reset game
* **Close Window:** Exit

🎨 The heatmap visualizes the AI’s confidence for every board position.

---

## 🎓 Academic Justification

> Training is performed offline using self-play with a policy-gradient method.
> The GUI is used solely for visualization and interaction, emphasizing interpretability and originality over competitive optimality.

---

## 🏁 Final Conclusion

This project demonstrates how **Deep Reinforcement Learning**, combined with **self-play**, can transform random actions into structured strategy in competitive games.
By separating **algorithmic benchmarking** and **explainable system design**, the work satisfies both **technical rigor** and **creative originality**.

---

## 📬 Contact

**Team NashCraft**
MSc Big Data Analytics
Ramakrishna Mission Vivekananda Educational and Research Institute (RKMVERI)


Just tell me.
