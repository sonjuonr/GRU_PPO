import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("TkAgg")
import torch.nn.functional as F
import matplotlib.patches as patches
import rl_utils  # 确保 rl_utils.py 在同一文件夹
import scipy.special
import imageio  # 用于创建 GIF
import os  # 用于管理文件/文件夹

# 🌟 1. 导入配置 (与训练代码完全一致)
import config

# ---------------------------------------------------------------------
# --- 全局配置与参数 (从 config 导入) ---
# ---------------------------------------------------------------------
device = config.DEVICE
print(f"Using device: {device}")


# ---------------------------------------------------------------------
# --- 辅助函数 (与训练代码一致) ---
# ---------------------------------------------------------------------
def wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def update_movement(x, y, theta, v_D, v_A, DT):
    theta += v_A * DT
    theta = wrap_to_pi(theta)
    x += v_D * np.cos(theta) * DT
    y += v_D * np.sin(theta) * DT
    vx = v_D * np.cos(theta)
    vy = v_D * np.sin(theta)
    return x, y, theta, vx, vy


# ---------------------------------------------------------------------
# 🌟 障碍物类 (与训练代码一致)
# ---------------------------------------------------------------------
class Obstacle:
    def __init__(self, x, y, vx, vy, speed_range=(0.1, 0.3)):  # 使用训练时的速度
        self.x_init, self.y_init = x, y
        self.vx_init, self.vy_init = vx, vy
        self.speed_range = speed_range
        self.reset()

    def reset(self, x_init=None, y_init=None):
        if x_init is not None and y_init is not None:
            self.x, self.y = x_init, y_init
        else:
            self.x = np.random.uniform(config.OBSTACLE_SPAWN_BOX[0], config.OBSTACLE_SPAWN_BOX[1])
            self.y = np.random.uniform(config.OBSTACLE_SPAWN_BOX[2], config.OBSTACLE_SPAWN_BOX[3])

        angle = np.random.uniform(0, 2 * np.pi)
        speed = np.random.uniform(*self.speed_range)
        self.vx = speed * np.cos(angle)
        self.vy = speed * np.sin(angle)

    def update(self):
        self.x += self.vx * config.DT
        self.y += self.vy * config.DT
        if not (config.SPAWN_BOX[0] < self.x < config.SPAWN_BOX[1]):
            self.vx *= -1
        if not (config.SPAWN_BOX[2] < self.y < config.SPAWN_BOX[3]):
            self.vy *= -1


# ---------------------------------------------------------------------
# 🌟 状态计算函数 (与训练代码一致)
# ---------------------------------------------------------------------
def calculate_state(robot_x, robot_y, robot_theta, robot_vx, robot_vy,
                    target_x, target_y, obstacles):
    dx_goal = target_x - robot_x
    dy_goal = target_y - robot_y
    goal_dist = np.hypot(dx_goal, dy_goal)
    angle_to_goal = np.arctan2(dy_goal, dx_goal)
    relative_goal_angle = wrap_to_pi(angle_to_goal - robot_theta)
    goal_cos = np.cos(relative_goal_angle)
    goal_sin = np.sin(relative_goal_angle)
    state = [goal_dist, goal_cos, goal_sin]
    cos_theta = np.cos(-robot_theta)
    sin_theta = np.sin(-robot_theta)

    for obs in obstacles:
        rel_x_world = obs.x - robot_x
        rel_y_world = obs.y - robot_y
        rel_vx_world = obs.vx - robot_vx
        rel_vy_world = obs.vy - robot_vy
        rel_x_robot = rel_x_world * cos_theta - rel_y_world * sin_theta
        rel_y_robot = rel_x_world * sin_theta + rel_y_world * cos_theta
        rel_vx_robot = rel_vx_world * cos_theta - rel_vy_world * sin_theta
        rel_vy_robot = rel_vx_world * sin_theta + rel_vy_world * cos_theta
        state.extend([rel_x_robot, rel_y_robot, rel_vx_robot, rel_vy_robot])

    return np.array(state), relative_goal_angle


# ---------------------------------------------------------------------
# 🌟 Actor-Critic 网络 (从 GRU 训练代码中复制)
# ---------------------------------------------------------------------
class ActorGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, action_dim):
        super(ActorGRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.gru_cell = nn.GRUCell(hidden_dim, hidden_dim)
        self.action_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, x, hidden_state):
        x = F.relu(self.fc1(x))
        next_hidden_state = self.gru_cell(x, hidden_state)
        logits = self.action_head(next_hidden_state)
        dist = torch.distributions.Categorical(logits=logits)
        return dist, next_hidden_state


class CriticGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(CriticGRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.gru_cell = nn.GRUCell(hidden_dim, hidden_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x, hidden_state):
        x = F.relu(self.fc1(x))
        next_hidden_state = self.gru_cell(x, hidden_state)
        value = self.value_head(next_hidden_state)
        return value, next_hidden_state


# ---------------------------------------------------------------------
# 🌟 PPO 算法类 (测试版本)
# ---------------------------------------------------------------------
class PPO_GRU:
    def __init__(self, state_dim, action_dim, gru_hidden_dim,
                 actor_lr, critic_lr, lmbda, epochs, eps, gamma, device):
        self.device = device
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = eps

        # 实例化 GRU 网络
        self.actor = ActorGRU(state_dim, gru_hidden_dim, action_dim).to(self.device)
        self.critic = CriticGRU(state_dim, gru_hidden_dim).to(self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

    def take_action(self, state, actor_hidden):
        state = torch.as_tensor(state, dtype=torch.float).to(self.device).unsqueeze(0)

        # 🌟 关键：在测试时，我们使用确定性动作 (deterministic)
        # 即选择概率最高的动作 (argmax)，而不是采样 (sample)
        with torch.no_grad():
            dist, next_actor_hidden = self.actor(state, actor_hidden)
            action = torch.argmax(dist.probs)  # 确定性动作
            log_prob = dist.log_prob(action)

        return action.item(), log_prob.item(), next_actor_hidden

    def update(self, transition_dict):
        # 此函数在测试中不被调用
        pass


# --- Agent 和障碍物初始化 ---
# 🌟 全部使用 config
agent = PPO_GRU(config.STATE_DIM, config.ACTION_DIM, config.GRU_HIDDEN_DIM,
                config.ACTOR_LR, config.CRITIC_LR, config.LMBDA, config.EPOCHS,
                config.EPS, config.GAMMA, device)

obstacles = []
for _ in range(config.NUM_RANDOM_OBSTACLES):
    obstacles.append(Obstacle(0, 0, 0, 0))  # 用虚拟值初始化

# =====================================================================
# 🎮 Phase 2: GRU-PPO 模拟 (生成 GIF)
# =====================================================================
print("\n--- STARTING GRU-PPO DYNAMIC SIMULATION ---")

# 1. 加载训练好的 GRU 模型
MODEL_NAME_ACTOR = 'gru_ppo_actor_dynamic_final.pth'
MODEL_NAME_CRITIC = 'gru_ppo_critic_dynamic_final.pth'
try:
    agent.actor.load_state_dict(torch.load(MODEL_NAME_ACTOR, map_location=device))
    agent.critic.load_state_dict(torch.load(MODEL_NAME_CRITIC, map_location=device))
    agent.actor.eval()
    agent.critic.eval()
    print(f"✅ Successfully loaded '{MODEL_NAME_ACTOR}' and '{MODEL_NAME_CRITIC}'.")
except FileNotFoundError:
    print(f"⚠️ Model files not found. Ensure models exist.")
    exit()

# --- 2. 模拟环境设置 ---
start_x = np.random.uniform(config.SPAWN_BOX[0], config.SPAWN_BOX[1])
start_y = np.random.uniform(config.SPAWN_BOX[2], config.SPAWN_BOX[3])
start_yaw = np.random.uniform(-np.pi, np.pi)

target_x = np.random.uniform(config.SPAWN_BOX[0], config.SPAWN_BOX[1])
target_y = np.random.uniform(config.SPAWN_BOX[2], config.SPAWN_BOX[3])
while np.hypot(start_x - target_x, start_y - target_y) < 1.0:
    target_x = np.random.uniform(config.SPAWN_BOX[0], config.SPAWN_BOX[1])
    target_y = np.random.uniform(config.SPAWN_BOX[2], config.SPAWN_BOX[3])

for obs in obstacles:
    obs.reset()  # 重置障碍物
    while np.hypot(obs.x - start_x, obs.y - start_y) < config.R_OBSTACLE * 4:
        obs.reset()

robot_x, robot_y, robot_theta = start_x, start_y, start_yaw
robot_vx, robot_vy = 0.0, 0.0
trajectory_x, trajectory_y = [robot_x], [robot_y]
collision_flag = False

# --- 3. GIF 设置 ---
FRAME_DIR = "simulation_frames_gru"  # 更改文件夹名
if not os.path.exists(FRAME_DIR):
    os.makedirs(FRAME_DIR)

frame_files = []
fig, ax = plt.subplots(figsize=[8, 8])

print(f"Starting Simulation... Will save frames to '{FRAME_DIR}'")

# 🌟 关键：初始化GRU的隐藏状态
actor_hidden = torch.zeros(1, config.GRU_HIDDEN_DIM).to(device)

# --- 4. 模拟循环 ---
# 🌟 使用 config 中的最大步数
for sim_step in range(config.STEPS_PER_EPISODE):
    if collision_flag: break

    # --- PPO 动作执行 (GRU 版本) ---
    # 1. 计算当前状态
    ppo_state, _ = calculate_state(
        robot_x, robot_y, robot_theta, robot_vx, robot_vy,
        target_x, target_y, obstacles
    )

    # 2. 传入状态和 *隐藏状态* 来获取动作
    action_index, _, next_actor_hidden = agent.take_action(ppo_state, actor_hidden)

    # 3. 更新隐藏状态以备下一步使用
    actor_hidden = next_actor_hidden

    # 🌟 使用 config 中的动作
    v_A = config.A[action_index]
    v_D = config.D[action_index]

    # 4. 更新机器人位置和速度
    new_x, new_y, new_theta, new_vx, new_vy = update_movement(
        robot_x, robot_y, robot_theta, v_D, v_A, config.DT
    )

    # --- 更新障碍物 ---
    for obs in obstacles:
        obs.update()

    # --- 碰撞检测 ---
    for obs in obstacles:
        if np.hypot(new_x - obs.x, new_y - obs.y) < (config.R_OBSTACLE + config.R_ROBOT):
            collision_flag = True
            print(f"Simulation Ended at step {sim_step}: Collision!")
            break
    if collision_flag:
        break

    # 更新机器人状态
    robot_x, robot_y, robot_theta = new_x, new_y, new_theta
    robot_vx, robot_vy = new_vx, new_vy  # 必须更新速度

    trajectory_x.append(robot_x)
    trajectory_y.append(robot_y)

    # 检查是否到达终点
    if np.hypot(robot_x - target_x, robot_y - target_y) < config.TARGET_REACH_THRESH:
        print(f"Simulation Ended at step {sim_step}: Target Reached!")
        break

    # --- 5. 绘图和保存帧 (每 2 帧保存一次以加快速度) ---
    if sim_step % 2 == 0:
        ax.cla()  # 清除上一帧

        # 绘制参考线 (起点到终点)
        ax.plot([start_x, target_x], [start_y, target_y], 'g--', linewidth=1.5, label="Reference Line")

        # 绘制障碍物
        for i, obs in enumerate(obstacles):
            circle = plt.Circle((obs.x, obs.y), config.R_OBSTACLE, color='b', alpha=0.5,
                                label="Obstacle" if i == 0 else "")
            ax.add_patch(circle)
            # 绘制障碍物速度矢量
            ax.arrow(obs.x, obs.y, obs.vx, obs.vy, head_width=0.05, head_length=0.1, fc='blue', ec='blue')

        # 绘制轨迹
        ax.plot(trajectory_x, trajectory_y, 'r-', linewidth=2, label="GRU-PPO Trajectory")

        # 绘制机器人
        arrow = patches.Arrow(robot_x, robot_y,
                              0.05 * 2 * np.cos(robot_theta),  # fish_len = 0.05
                              0.05 * 2 * np.sin(robot_theta),
                              width=0.05, color='k', label="Robot")
        ax.add_patch(arrow)

        # 绘制起点和终点
        ax.plot(start_x, start_y, 'go', markersize=8, label="Start")
        ax.plot(target_x, target_y, 'r*', markersize=12, label="Target")

        # 设置图表属性
        ax.axis('equal')
        ax.set_xlabel("X [m]", fontsize=14)
        ax.set_ylabel("Y [m]", fontsize=14)
        ax.set_title(f"GRU-PPO Simulation (Step: {sim_step + 1}/{config.STEPS_PER_EPISODE})")

        # 🌟 固定绘图范围
        ax.set_xlim(config.SPAWN_BOX[0] - 0.2, config.SPAWN_BOX[1] + 0.2)
        ax.set_ylim(config.SPAWN_BOX[2] - 0.2, config.SPAWN_BOX[3] + 0.2)

        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right')
        ax.grid(True)

        # 保存帧
        frame_filename = os.path.join(FRAME_DIR, f"frame_{sim_step:04d}.png")
        fig.savefig(frame_filename)
        frame_files.append(frame_filename)

# --- 循环结束 ---
plt.close(fig)

if not collision_flag and sim_step == config.STEPS_PER_EPISODE - 1:
    print(f"Simulation Ended: Reached max steps ({config.STEPS_PER_EPISODE}).")

# --- 6. 创建 GIF ---
if not frame_files:
    print("No frames were saved. Skipping GIF creation.")
else:
    print(f"\nCreating GIF from {len(frame_files)} frames...")
    images = []
    for filename in tqdm(frame_files):
        images.append(imageio.imread(filename))

    # 🌟 更改 GIF 路径
    gif_path = 'D:/code for obstacle avoidance/animation/gru_dynamic_avoidance.gif'
    # 确保文件夹存在
    os.makedirs(os.path.dirname(gif_path), exist_ok=True)

    imageio.mimsave(gif_path, images, duration=0.1)  # 10 FPS
    print(f"✅ GIF saved as '{gif_path}'")

    # --- 7. 清理 ---
    print("Cleaning up frame files...")
    for filename in frame_files:
        os.remove(filename)
    os.rmdir(FRAME_DIR)
    print("Done.")