import torch

# --- Core Training Parameters ---
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
TOTAL_TIMESTEPS = 10_000_000  
ROLLOUT_STEPS = 4096
MINIBATCH_SIZE = 512        
EPOCHS = 5

# --- PPO Algorithm Parameters ---
ACTOR_LR = 1e-4
CRITIC_LR = 1e-4
GAMMA = 0.99               
LMBDA = 0.95
EPS = 0.2
GRAD_CLIP_MAX_NORM = 0.5
ENTROPY_COEF = 0.01

# --- Network Dimensions ---
GRU_HIDDEN_DIM = 64
STATE_DIM = 11
ACTION_DIM = 3

# --- Environment Parameters ---
STEPS_PER_EPISODE = 1500
TARGET_REACH_THRESH = 0.2
DT = 0.1
D = [0.06, 0.1, 0.06]
A = [-0.25, 0, 0.25]
R_OBSTACLE = 0.15
R_ROBOT = 0.05
NUM_RANDOM_OBSTACLES = 2
SPAWN_BOX = [0.0, 3.0, 0.0, 3.0]
OBSTACLE_SPAWN_BOX = [0.5, 2.5, 0.5, 2.5]

# --- 🌟 Critical Fix: Reward Weights ---
# Logic: Greatly weaken attraction (Shaping), greatly strengthen warning (Obstacle)

REWARD_SUCCESS = 10.0
REWARD_COLLISION = -50.0

REWARD_SHAPING_WEIGHT = 0.4    # ⬇️ Reduced (was 0.8). Don't let it be too greedy.
REWARD_HEADING_WEIGHT = -0.05  # ⬆️ Tweaked (was -0.01). Slightly stronger sense of direction.
REWARD_OBSTACLE_WEIGHT = -1.2  # ⬆️ Significantly increased absolute value (was -0.4). Make it afraid to get close!
REWARD_STEP_WEIGHT = -0.005    # ⬆️ Tweaked (was -0.002). Prevent it from taking excessively long detours.

# --- Logging ---
LOG_DIR = "runs/gru_ppo_optimized_v2"
