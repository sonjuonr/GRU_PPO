# GRU_PPO
# GRU-PPO for Dynamic Obstacle Avoidance 🤖🐟

![Simulation Demo](https://github.com/sonjuonr/GRU_PPO/blob/main/gru_dynamic_avoidance_FIXED_Extended.gif?raw=true)

## 📖 Introduction

This project implements a Deep Reinforcement Learning (DRL) framework for autonomous navigation in environments with **dynamic obstacles** and **physical inertia** (e.g., robotic fish underwater navigation).

The core algorithm combines **PPO (Proximal Policy Optimization)** with a **GRU (Gated Recurrent Unit)** network. Unlike standard MLP-based approaches, the GRU architecture provides the agent with **memory**, enabling it to handle **POMDP (Partially Observable Markov Decision Process)** scenarios. By encoding historical state information, the agent can implicitly infer the velocity and trajectory of moving obstacles, resulting in smooth, anticipatory, and safe avoidance behaviors.

## ✨ Key Features

* **Memory-Augmented Agent:** Utilizes GRU hidden states to capture temporal dependencies, allowing the agent to "see" speed and acceleration from raw positional data.
* **High-Fidelity Control:** Optimized for high-frequency control (`DT=0.1s`) with a long planning horizon (`GAMMA=0.99`), suitable for robots with physical inertia.
* **Robust Reward Shaping:** Features a carefully tuned reward function that balances target-seeking "greed" with obstacle-avoidance "safety," resolving issues of local optima and sparse rewards.
* **Visualization:** Integrated with TensorBoard for real-time monitoring of Loss, Entropy, Collision Rate, and Success Rate.

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
    pip install numpy matplotlib tensorboard tqdm scipy
    ```

## 📂 File Structure

```text
├── config.py           # [Core] Configuration for hyperparameters, environment physics, and rewards
├── PPO_GRU.py        # Main training script for the GRU-PPO agent
├── test_gru.py         # Evaluation script to visualize trained models
├── test_gru——fixed.py  # simulation file with fixed obstacles
├── rl_utils.py         # Utility functions (GAE calculation, etc.)
├── runs/               # Directory for TensorBoard logs
└── README.md           # Project documentation

