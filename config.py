import torch

# --- 核心训练参数 ---
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
EPISODES = 20000  # 备注：这个参数在 N-Step-Update 中将不再被直接使用
TOTAL_TIMESTEPS = 2_000_000  # 训练总步数 (替代 episodes)
ROLLOUT_STEPS = 2048         # 🌟 N-Step: 每次更新前收集的步数
MINIBATCH_SIZE = 128         # 🌟 N-Step: 每次更新时使用的批大小 (必须能整除 ROLLOUT_STEPS)
EPOCHS = 10                  # 每次更新时, 训练数据的重播次数

# --- PPO 算法参数 ---
ACTOR_LR = 1e-4
CRITIC_LR = 1e-4
GAMMA = 0.96
LMBDA = 0.95
EPS = 0.2
GRAD_CLIP_MAX_NORM = 0.5  # 梯度裁剪 (来自之前的分析)
ENTROPY_COEF = 0.01       # 熵奖励系数

# --- 网络维度 ---
GRU_HIDDEN_DIM = 64
STATE_DIM = 11
ACTION_DIM = 3

# --- 环境参数 ---
STEPS_PER_EPISODE = 150  # 每轮最大步数
TARGET_REACH_THRESH = 0.2
DT = 1
D = [0.06, 0.1, 0.06]
A = [-0.25, 0, 0.25]
R_OBSTACLE = 0.15
R_ROBOT = 0.05
NUM_RANDOM_OBSTACLES = 2
SPAWN_BOX = [0.0, 3.0, 0.0, 3.0]
OBSTACLE_SPAWN_BOX = [0.5, 2.5, 0.5, 2.5]

# --- 🌟 奖励权重 (解决了你的问题2) ---
REWARD_SUCCESS = 10.0
REWARD_COLLISION = -10.0
REWARD_SHAPING_WEIGHT = 0.5
REWARD_HEADING_WEIGHT = -0.1
REWARD_OBSTACLE_WEIGHT = -0.06
REWARD_STEP_WEIGHT = -0.02

# --- 🌟 TensorBoard 日志 ---
LOG_DIR = "runs/gru_ppo_n_step_v1"