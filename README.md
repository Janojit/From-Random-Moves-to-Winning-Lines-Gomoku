# 🧠 **From Random Moves to Winning Lines: DRL for Gomoku**

**Team Name:** NashCraft
**Members:**

* Janojit Chakraborty (B2430050)
* Radheshyam Routh (B2430053)

---

## 📌 Project Overview

This project explores **Deep Reinforcement Learning (DRL)** in a competitive, zero-sum board game environment—**Gomoku (Five-in-a-Row)**.

The project is divided into **two major components**:

1. **Component-1:** Applying and comparing multiple DRL algorithms in a moderately difficult Gym-style environment using self-play.
2. **Component-2:** Designing a custom Gomoku environment with a graphical user interface (GUI) for interactive demonstration.

This repository contains **complete implementations**, **training scripts**, **comparative evaluation tools**, and a **visual GUI**.

---

## 🗂️ Repository Structure

```
gomoku_drl/
│
├── env/                # Gym-style Gomoku environment
│   └── gomoku_env.py
│
├── agents/             # DRL agents
│   ├── __init__.py
│   ├── dqn.py
│   ├── double_dqn.py
│   ├── ppo.py
│   ├── a2c.py
│   └── reinforce.py
│
├── train/              # Training scripts (self-play)
│   ├── __init__.py
│   ├── train_dqn.py
│   ├── train_double_dqn.py
│   ├── train_ppo.py
│   ├── train_a2c.py
│   └── train_reinforce.py
│
├── eval/               # Evaluation & comparison
│   ├── __init__.py
│   ├── evaluate.py
│   ├── compare_models.py
│   └── analysis.txt
│
├── gui/                # Graphical user interface
│   ├── __init__.py
│   └── gomoku_gui.py
│
├── utils/              # Shared utilities
│   ├── __init__.py
│   ├── networks.py
│   ├── replay_buffer.py
│   └── helpers.py
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Step-by-Step Instructions

---

## 🔹 STEP 1: Environment Setup

### 1. Create a Python virtual environment

```bash
python -m venv gomoku_env
source gomoku_env/bin/activate     # Linux / macOS
gomoku_env\Scripts\activate        # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** GPU support is automatically enabled if CUDA is available (RTX 3050 Ti).

---

## 🔹 STEP 2: Verify the Gomoku Environment

Before training any model, verify the environment logic.

```bash
python -m env.gomoku_env
```

Ensure:

* Board initializes correctly
* Moves are applied properly
* Win detection works

---

## 🔹 STEP 3: Train DRL Agents (Component-1)

Each algorithm is trained using **self-play** in the same environment to ensure a fair comparison.

### Recommended training order:

#### 1️⃣ DQN

```bash
python -m train.train_dqn
```

#### 2️⃣ Double DQN

```bash
python -m train.train_double_dqn
```

#### 3️⃣ REINFORCE

```bash
python -m train.train_reinforce
```

#### 4️⃣ A2C

```bash
python -m train.train_a2c
```

#### 5️⃣ PPO (Main Algorithm)

```bash
python -m train.train_ppo
```

📌 **Output:**
Trained models are saved as `.pth` files in the project root.

---

## 🔹 STEP 4: Evaluate and Compare Models

### 1. Run evaluation matches

```bash
python -m eval.evaluate
```

This script:

* Runs fixed evaluation episodes
* Measures wins, losses, and draws
* Saves results to disk

### 2. Generate comparison plots

```bash
python -m eval.compare_models
```

📊 **Generated Outputs:**

* `win_count.png`
* `loss_count.png`
* `draw_count.png`

### 3. Play Tournament

```
python -m eval.tournament
```

This file explains:

* Training stability
* Algorithmic strengths and weaknesses
* Self-play behavior differences

---

## 🔹 STEP 5: Launch the GUI (Component-2)

Run the interactive Gomoku interface:

```bash
python -m gui.gomoku_gui
```

### 🎮 Available Modes

* **Human vs Human**
* **Human vs Agent**
* **Agent vs Agent**

### 🤖 AI Selection

* DQN
* Double-DQN
* PPO
* A2C
* REINFORCE

The GUI visually displays:

* Game board
* Player turns
* Winning condition

---

## 🧪 Algorithms Used

| Algorithm  | Type         | Purpose              |
| ---------- | ------------ | -------------------- |
| DQN        | Value-based  | Baseline             |
| Double-DQN | Value-based  | Reduced bias         |
| REINFORCE  | Policy-based | Monte-Carlo learning |
| A2C        | Actor-Critic | Faster convergence   |
| PPO        | Policy-based | Stable & robust      |

---

## 📈 Evaluation Metrics

* Win rate
* Loss rate
* Draw frequency
* Training stability
* Convergence behavior

---

## 🎯 Key Takeaways

* Value-based methods struggle with non-stationary self-play.
* Policy-gradient methods (PPO, A2C) show superior stability.
* Strategic behaviors emerge without handcrafted heuristics.
* Custom environment + GUI adds originality beyond benchmarks.

---

## 🏁 Conclusion

This project demonstrates how **Deep Reinforcement Learning**, combined with **self-play**, can transform random actions into strategic decision-making in competitive board games. The modular design allows easy extension to larger boards or additional algorithms.

---

## 📬 Contact

For questions or collaboration:

* **Team NashCraft**
* MSc Big Data Analytics, RKMVERI

---
Just tell me.
