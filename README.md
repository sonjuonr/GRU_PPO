# GRU-PPO for Dynamic Obstacle Avoidance 🤖🐟

![Simulation Demo](https://github.com/sonjuonr/GRU_PPO/blob/main/results_1121/gru_dynamic_avoidance_final.gif?raw=true)

## 📖 Introduction

This project implements a **Deep Reinforcement Learning (DRL)** framework for autonomous navigation in environments with **dynamic obstacles** and **physical inertia** (e.g., robotic fish underwater navigation).

The core algorithm combines **PPO (Proximal Policy Optimization)** with a **GRU (Gated Recurrent Unit)** network. Unlike standard MLP-based approaches, the GRU architecture provides the agent with **memory**, enabling it to handle **POMDP (Partially Observable Markov Decision Process)** scenarios. By encoding historical state information, the agent can implicitly infer the velocity and trajectory of moving obstacles, resulting in smooth, anticipatory, and safe avoidance behaviors.

## ✨ Key Features

* **Memory-Augmented Agent:** Utilizes GRU hidden states to capture temporal dependencies, allowing the agent to "see" speed and acceleration from raw positional data.
* **High-Fidelity Control:** Optimized for high-frequency control (`DT=0.1s`) with a long planning horizon (`GAMMA=0.99`), suitable for robots with physical inertia.
* **Safety-First Reward Structure:** Implements a strictly prioritized reward mechanism that solves the "suicidal greed" problem common in RL, forcing the agent to prioritize safety over speed.
* **Visualization:** Integrated with TensorBoard for real-time monitoring of Loss, Entropy, Collision Rate, and Success Rate.

## 🚀 Key Improvements (Dec 1 Update)
I found that i made a big mistake. I mistakenly set 0.08 to 0.8, but i didn’t find it in past two weeks. I believe that’s the main reason why the succ rate is so low. I have fixed it and run again. Let's see what the results will change.
## 🚀 Key Improvements (Nov 21 Update)

This version introduces significant improvements over baseline implementations:

1.  **Logic Fix:** Implemented strict mutual exclusivity between `Collision`, `Success`, and `Timeout` events to prevent "Collision-Success" logic bugs.
2.  **Physics Tuning:** Reduced simulation time step (`DT`) to **0.1s** and increased obstacle detection radius to **0.8m-1.0m** to account for the robot's turning radius and physical inertia.
3.  **Reward Shaping:** Inverted the "Greed vs. Fear" ratio.
    * Increased Obstacle Penalty (`-1.2`) to strictly outweigh Shaping Reward (`0.4`).
    * This forces the agent to choose safe detours instead of risky shortcuts.
4.  **Architecture:** Increased GRU hidden dimension to **128** to better capture complex dynamic environments.

## 📊 Training Results

The model demonstrates stable convergence and robust obstacle avoidance capabilities (Results from `results_1121`).

| **Training Metrics** | **Description** |
| :---: | :--- |
| ![Success Rate](https://github.com/sonjuonr/GRU_PPO/blob/main/results_1121/figure3.png?raw=true) | **Success Rate & Rewards:** Aggressive Breakthrough: The agent achieves a stable ~42% Success Rate, a 2x improvement over baseline. The Mean Length drops to ~133 steps, indicating a "Speed-Run" strategy: the agent learns to minimize exposure to the high-penalty danger zones by navigating through them as quickly as possible.|
| ![Losses](https://github.com/sonjuonr/GRU_PPO/blob/main/results_1121/figure4.png?raw=true) | **Actor & Critic Loss:** Capacity Bottleneck Identified: While the Critic Loss stabilizes around ~20, its high variance suggests that the current 64-unit GRU is reaching its memory capacity limit in this complex dynamic environment. Future iterations will scale to 128/256 units to further reduce prediction error.|
| ![Entropy](https://github.com/sonjuonr/GRU_PPO/blob/main/results_1121/figure2.png?raw=true) | **Entropy & SPS:** Active Adaptation: Entropy remains moderately high (~0.65), reflecting that the agent is still actively exploring fine-grained control policies to handle the delicate balance between the high-speed inertia and the strict obstacle penalties.|
| ![Reward Components](https://github.com/sonjuonr/GRU_PPO/blob/main/results_1121/figure1.png?raw=true) | **Detailed Rewards:** Effective Constraints: The Obstacle Penalty graph confirms the agent is heavily penalized for risky proximity, validating that the new safety constraints are actively shaping the policy, preventing the agent from loitering in dangerous areas.|

## 🛠️ Requirements

This project is developed and tested with **Python 3.9**.

* **Python:** 3.9.19
* **PyTorch:** 2.2.0 (CUDA 12.1)
* **CUDA:** 12.1

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/sonjuonr/GRU_PPO.git](https://github.com/sonjuonr/GRU_PPO.git)
    cd GRU_PPO
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment (Conda/venv).
    
    ```bash
    # Install PyTorch compatible with CUDA 12.1
    pip install torch==2.2.0 torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
    
    # Install other required packages
    pip install numpy matplotlib tensorboard tqdm scipy imageio
    ```

## 📂 File Structure

```text
├── config.py           # [Core] Configuration for hyperparameters, environment physics, and rewards
├── PPO_GRU.py          # Main training script for the GRU-PPO agent
├── test_gru.py         # Evaluation script to visualize trained models with dynamic obstacles
├── test_gru_fixed.py   # Simulation script with fixed obstacle scenarios for debugging
├── rl_utils.py         # Utility functions (GAE calculation, etc.)
├── runs/               # Directory for TensorBoard logs
├── results_1121/       # Training result figures and gifs
└── README.md           # Project documentation
