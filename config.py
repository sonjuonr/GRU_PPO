import torch

# --- CORE TRAINING PARAMETERS ---
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
TOTAL_TIMESTEPS = 5_000_000  # Suggested increase to 10M for GRU
ROLLOUT_STEPS = 4096
MINIBATCH_SIZE = 512         # Recommended to increase MiniBatch size for speed
EPOCHS = 5

# --- PPO ALGORITHM PARAMETERS ---
ACTOR_LR = 1e-4
CRITIC_LR = 1e-4
GAMMA = 0.99                 # Maintain 0.99
LMBDA = 0.95
EPS = 0.2
GRAD_CLIP_MAX_NORM = 0.5
ENTROPY_COEF = 0.01

# --- NETWORK DIMENSIONS ---
GRU_HIDDEN_DIM = 128
STATE_DIM = 11
ACTION_DIM = 3

# --- ENVIRONMENT PARAMETERS ---
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

# --- 🌟 CRITICAL REWARD ADJUSTMENTS ---
# Logic: Drastically reduce temptation (Shaping) and drastically increase warning (Obstacle)

REWARD_SUCCESS = 10.0
REWARD_COLLISION = -50.0

REWARD_SHAPING_WEIGHT = 0.4    # ⬇️ Decrease (was 0.8). Reduce greed.
REWARD_HEADING_WEIGHT = -0.02  # ⬆️ Micro-adjustment (was -0.05). Slightly reinforce direction.
REWARD_OBSTACLE_WEIGHT = -0.5  # ⬆️ Increase magnitude (was -0.4). Instill fear of proximity.
REWARD_STEP_WEIGHT = -0.003    # ⬆️ Micro-adjustment (was -0.002). Discourage excessive loitering/detours.

# --- LOGGING ---
LOG_DIR = "runs/gru_ppo_optimized_v2"
