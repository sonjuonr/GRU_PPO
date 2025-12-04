# GRU-PPO for Dynamic Obstacle Avoidance 🤖🐟

![Simulation Demo](https://github.com/sonjuonr/GRU_PPO/blob/main/gru_dynamic_avoidance_FIXED_Extended.gif?raw=true)

## 📖 Introduction

This project implements a **Deep Reinforcement Learning (DRL)** framework for autonomous navigation in environments with **dynamic obstacles** and **physical inertia** (e.g., robotic fish underwater navigation).

The core algorithm combines **PPO (Proximal Policy Optimization)** with a **GRU (Gated Recurrent Unit)** network. Unlike standard MLP-based approaches, the GRU architecture provides the agent with **memory**, enabling it to handle **POMDP (Partially Observable Markov Decision Process)** scenarios. By encoding historical state information, the agent can implicitly infer the velocity and trajectory of moving obstacles, resulting in smooth, anticipatory, and safe avoidance behaviors.

## ✨ Key Features

* **Memory-Augmented Agent:** Utilizes GRU hidden states to capture temporal dependencies, allowing the agent to "see" speed and acceleration from raw positional data.
* **High-Fidelity Control:** Optimized for high-frequency control (`DT=0.1s`) with a long planning horizon (`GAMMA=0.99`), suitable for robots with physical inertia.
* **Robust Reward Shaping:** Features a carefully tuned reward function that balances target-seeking "greed" with obstacle-avoidance "safety," resolving issues of local optima and sparse rewards.
* **Visualization:** Integrated with TensorBoard for real-time monitoring of Loss, Entropy, Collision Rate, and Success Rate.

## 🚀 Key Improvements (Dec 1 Update)

**Critical Physics Correction:** We identified and fixed a major discrepancy in the simulation environment. The dynamic obstacle speed range was previously set to `0.8 - 0.15 m/s` (significantly faster than the robot's `0.1 m/s`), making collision avoidance physically impossible in many scenarios. This has been corrected to `0.08 - 0.15 m/s`. This adjustment has fundamentally resolved the low success rate issue and validated the algorithm's effectiveness.

**Previous Improvements (Nov 21):**
1.  **Logic Fix:** Implemented strict mutual exclusivity between collision, success, and timeout events.
2.  **Physics Tuning:** Reduced simulation time step (`DT`) to **0.1s** and adjusted detection radius.
3.  **Reward Shaping:** Prioritized obstacle penalties (`-1.0`) over shaping rewards.
4.  **Architecture:** Increased GRU hidden dimension to **128**.

## 🏆 Model Selection

We provide two trained model versions with distinct behavioral characteristics. 

### 🌟 **Recommended: Aggressive Model (Results12_4)**
* **Location:** `results12_4/`
* **Behavior:** Highly efficient and agile. It takes tighter turns and optimized paths to reach the target quickly.
* **Use Case:** **Strongly Recommended** for general tasks where efficiency is a priority.

### 🛡️ **Alternative: Conservative Model (Results 12_2)**
* **Location:** `results_12_2/`
* **Behavior:** Prioritizes safety above all else. It tends to maintain a larger buffer distance from obstacles and may take longer, wider paths to ensure zero collisions.
* **Use Case:** Scenarios requiring maximum safety margins.

## 📊 Training Results (Conservative Model - 12_2)

The following visualizations analyze the performance of the **Conservative Model (Dec 2)**.

| **Training Metrics** | **Analysis & Interpretation** |
| :---: | :--- |
| ![Episode Stats](https://github.com/sonjuonr/GRU_PPO/blob/main/results%2012_2/length_reward_succ_rate.png?raw=true) | **Stability & Safety:** The *Mean Success Rate* shows a robust upward trend, stabilizing at a high level. The *Mean Length* is consistent, reflecting the model's tendency to choose longer, safer detours rather than risky shortcuts. |
| ![Losses](https://github.com/sonjuonr/GRU_PPO/blob/main/results%2012_2/actor_critic_loss.png?raw=true) | **Convergence:** Both *Actor* and *Critic* losses show healthy convergence patterns. The Critic Loss stabilizes effectively, indicating the GRU has successfully learned to predict the value of safe states in a dynamic environment. |
| ![Entropy](https://github.com/sonjuonr/GRU_PPO/blob/main/results%2012_2/entropy_SPS.png?raw=true) | **Policy Confidence:** *Entropy* decreases steadily, proving the agent has transitioned from random exploration to a confident, deterministic policy focused on hazard avoidance. |
| ![Reward Components](https://github.com/sonjuonr/GRU_PPO/blob/main/results%2012_2/collision_heading_obstacle.png?raw=true) | **Behavioral Guidance:** The breakdown shows that *Obstacle Penalties* are effectively minimized over time, confirming the agent is actively learning to stay clear of the defined danger zones. |
| ![Reward Components](https://github.com/sonjuonr/GRU_PPO/blob/main/results%2012_2/shaping_steppenalty_succ.png?raw=true) | **Behavioral Guidance:** These two reward components parts show that the parameter of reward and penalty of each part has been debuged well. Just check the exact number and the converge trend. |

## ⚠️ Known Limitations & Future Work

### The "Coincident Arrival" Edge Case
We have identified a specific corner case where the dynamic obstacle and the robotic fish arrive at the target coordinates simultaneously. Due to the current reward structure prioritizing target reach, the agent may fail to yield in this specific "head-on" scenario.

![Edge Case Demo](https://github.com/sonjuonr/GRU_PPO/blob/main/results12_1/gru_dynamic_special_cond.gif?raw=true)

**Current Research:** We are developing a new predictive mechanism to detect and resolve these inevitable collision states (ICS) by factoring in target occupancy prediction.

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
├── results_1121/       # Archived results (Nov 21)
├── results12_4/        # Aggressive Model Results (Recommended)
├── results_12_2/       # Conservative Model Results
└── README.md           # Project documentation
