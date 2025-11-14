import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.patches as patches
import rl_utils
import scipy.special

matplotlib.use("TkAgg")


# ---------------------------------------------------------------------
# 🌟 LTC 网络定义 (与之前相同，保持不变)
# ---------------------------------------------------------------------

class RandomWiring:
    def __init__(self, input_dim, output_dim, neuron_count):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.neuron_count = neuron_count
        self.adjacency_matrix = np.random.uniform(0, 1, (neuron_count, neuron_count))
        self.sensory_adjacency_matrix = np.random.uniform(0, 1, (input_dim, neuron_count))

    def erev_initializer(self):
        return np.random.uniform(-0.2, 0.2, (self.neuron_count, self.neuron_count))

    def sensory_erev_initializer(self):
        return np.random.uniform(-0.2, 0.2, (self.input_dim, self.neuron_count))


class LIFNeuronLayer(nn.Module):
    def __init__(self, wiring, ode_unfolds=12, epsilon=1e-8):
        super(LIFNeuronLayer, self).__init__()
        self.wiring = wiring
        self.ode_unfolds = ode_unfolds
        self.epsilon = epsilon
        self.softplus = nn.Softplus()

        # Initialization ranges
        GLEAK_MIN, GLEAK_MAX = 0.001, 1.0
        VLEAK_MIN, VLEAK_MAX = -0.2, 0.2
        CM_MIN, CM_MAX = 0.4, 0.6
        W_MIN, W_MAX = 0.001, 1.0
        SIGMA_MIN, SIGMA_MAX = 3, 8
        MU_MIN, MU_MAX = 0.3, 0.8
        SENSORY_W_MIN, SENSORY_W_MAX = 0.001, 1.0
        SENSORY_SIGMA_MIN, SENSORY_SIGMA_MAX = 3, 8
        SENSORY_MU_MIN, SENSORY_MU_MAX = 0.3, 0.8

        # Initialize neuron parameters
        self.gleak = nn.Parameter(torch.rand(wiring.neuron_count) * (GLEAK_MAX - GLEAK_MIN) + GLEAK_MIN)
        self.vleak = nn.Parameter(torch.rand(wiring.neuron_count) * (VLEAK_MAX - VLEAK_MIN) + VLEAK_MIN)
        self.cm = nn.Parameter(torch.rand(wiring.neuron_count) * (CM_MAX - CM_MIN) + CM_MIN)
        self.w = nn.Parameter(torch.rand(wiring.neuron_count, wiring.neuron_count) * (W_MAX - W_MIN) + W_MIN)
        self.sigma = nn.Parameter(
            torch.rand(wiring.neuron_count, wiring.neuron_count) * (SIGMA_MAX - SIGMA_MIN) + SIGMA_MIN)
        self.mu = nn.Parameter(torch.rand(wiring.neuron_count, wiring.neuron_count) * (MU_MAX - MU_MIN) + MU_MIN)
        self.erev = nn.Parameter(torch.Tensor(wiring.erev_initializer()))

        # Initialize sensory parameters
        self.sensory_w = nn.Parameter(
            torch.rand(wiring.input_dim, wiring.neuron_count) * (SENSORY_W_MAX - SENSORY_W_MIN) + SENSORY_W_MIN)
        self.sensory_sigma = nn.Parameter(torch.rand(wiring.input_dim, wiring.neuron_count) * (
                SENSORY_SIGMA_MAX - SENSORY_SIGMA_MIN) + SENSORY_SIGMA_MIN)
        self.sensory_mu = nn.Parameter(
            torch.rand(wiring.input_dim, wiring.neuron_count) * (SENSORY_MU_MAX - SENSORY_MU_MIN) + SENSORY_MU_MIN)
        self.sensory_erev = nn.Parameter(torch.Tensor(wiring.sensory_erev_initializer()))

        # Sparsity masks
        self.sparsity_mask = nn.Parameter(torch.Tensor(np.abs(wiring.adjacency_matrix)), requires_grad=False)
        self.sensory_sparsity_mask = nn.Parameter(torch.Tensor(np.abs(wiring.sensory_adjacency_matrix)),
                                                  requires_grad=False)

    def forward(self, inputs, state, elapsed_time=1.0):
        return self.ode_solver(inputs, state, elapsed_time)

    def ode_solver(self, inputs, state, elapsed_time):
        v_pre = state

        # Pre-compute sensory effects
        sensory_activation = self.softplus(self.sensory_w) * self.sigmoid(inputs, self.sensory_mu, self.sensory_sigma)
        sensory_activation = sensory_activation * self.sensory_sparsity_mask
        sensory_reversal_activation = sensory_activation * self.sensory_erev

        w_numerator_sensory = torch.sum(sensory_reversal_activation, dim=1)
        w_denominator_sensory = torch.sum(sensory_activation, dim=1)

        cm_t = self.softplus(self.cm) / (elapsed_time / self.ode_unfolds)
        w_param = self.softplus(self.w)

        for _ in range(self.ode_unfolds):
            w_activation = w_param * self.sigmoid(v_pre, self.mu, self.sigma)
            w_activation = w_activation * self.sparsity_mask
            reversal_activation = w_activation * self.erev

            w_numerator = torch.sum(reversal_activation, dim=1) + w_numerator_sensory
            w_denominator = torch.sum(w_activation, dim=1) + w_denominator_sensory

            gleak = self.softplus(self.gleak)
            numerator = cm_t * v_pre + gleak * self.vleak + w_numerator
            denominator = cm_t + gleak + w_denominator

            v_pre = numerator / (denominator + self.epsilon)

        return v_pre

    def sigmoid(self, v_pre, mu, sigma):
        v_pre = torch.unsqueeze(v_pre, -1)
        mu = mu.unsqueeze(0)
        sigma = sigma.unsqueeze(0)
        activation = sigma * (v_pre - mu)
        return torch.sigmoid(activation)


class LTCCell(nn.Module):
    def __init__(self, wiring, in_features=None, ode_unfolds=6, epsilon=1e-8):
        super(LTCCell, self).__init__()
        self.wiring = wiring
        self.neuron = LIFNeuronLayer(wiring, ode_unfolds, epsilon)

    def forward(self, inputs, states, elapsed_time=1.0):
        next_state = self.neuron(inputs, states, elapsed_time)
        outputs = next_state[:, :self.wiring.output_dim]
        return outputs, next_state


# ---------------------------------------------------------------------
# --- 全局配置与参数 (已修改) ---
# ---------------------------------------------------------------------
steps = 150  # 每轮最大步数
TARGET_REACH_THRESH = 0.2  # 到达目标的判定距离

# RL Parameters
actor_lr = 1e-4
critic_lr = 1e-4
episodes = 20000

# 🌟 LTC 网络维度
ltc_hidden_dim = 64  # LTC 神经元数量 (记忆)
ltc_output_dim = 16  # LTC 输出维度

gamma = 0.96
lmbda = 0.95
epochs = 10
eps = 0.2
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using device: {device}")

# 🌟 状态与动作维度 (已修改)
state_dim = 11  # 3 (goal) + 2*4 (obstacles)
action_dim = 3  # 离散动作空间
DT = 1  # 时间步长
D = [0.06, 0.1, 0.06]  # 动作对应的线速度
A = [-0.25, 0, 0.25]  # 动作对应的角速度

# 环境参数
R_obstacle = 0.15  # 障碍物半径
R_robot = 0.05  # 机器人半径 (用于碰撞检测)
NUM_RANDOM_OBSTACLES = 2
# 🌟 随机生成区域
SPAWN_BOX = [0.0, 3.0, 0.0, 3.0]  # 机器人和目标的活动区域
OBSTACLE_SPAWN_BOX = [0.5, 2.5, 0.5, 2.5]  # 障碍物随机生成的区域


# ---------------------------------------------------------------------
# --- 辅助函数 (已修改) ---
# ---------------------------------------------------------------------
def wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def update_movement(x, y, theta, v_D, v_A, DT):
    theta += v_A * DT
    theta = wrap_to_pi(theta)
    x += v_D * np.cos(theta) * DT
    y += v_D * np.sin(theta) * DT
    # 返回新的 x, y, theta, 以及用于状态计算的 vx, vy
    vx = v_D * np.cos(theta)
    vy = v_D * np.sin(theta)
    return x, y, theta, vx, vy


# ---------------------------------------------------------------------
# 🌟 障碍物类 (已按讨论修改)
# ---------------------------------------------------------------------
class Obstacle:
    def __init__(self, x, y, vx, vy, speed_range=(0.1, 0.5)):
        self.x_init, self.y_init = x, y
        self.vx_init, self.vy_init = vx, vy
        self.speed_range = speed_range
        self.reset()

    def reset(self, x_init=None, y_init=None):
        """ 重置障碍物到初始位置和速度，或随机新位置/速度 """
        if x_init is not None and y_init is not None:
            self.x, self.y = x_init, y_init
        else:
            # 在指定区域内随机生成
            self.x = np.random.uniform(OBSTACLE_SPAWN_BOX[0], OBSTACLE_SPAWN_BOX[1])
            self.y = np.random.uniform(OBSTACLE_SPAWN_BOX[2], OBSTACLE_SPAWN_BOX[3])

        # 🌟 恒定速度模型：只在 reset 时随机化一次
        angle = np.random.uniform(0, 2 * np.pi)
        speed = np.random.uniform(*self.speed_range)
        self.vx = speed * np.cos(angle)
        self.vy = speed * np.sin(angle)

    def update(self):
        """ 恒定速度运动 """
        self.x += self.vx * DT
        self.y += self.vy * DT

        # 简易边界反弹
        if not (SPAWN_BOX[0] < self.x < SPAWN_BOX[1]):
            self.vx *= -1
        if not (SPAWN_BOX[2] < self.y < SPAWN_BOX[3]):
            self.vy *= -1


# ---------------------------------------------------------------------
# 🌟 状态计算函数 (全新)
# ---------------------------------------------------------------------
def calculate_state(robot_x, robot_y, robot_theta, robot_vx, robot_vy,
                    target_x, target_y, obstacles):
    """
    计算新的 11D 状态:
    [goal_dist, goal_cos, goal_sin,
     obs1_rel_x, obs1_rel_y, obs1_rel_vx, obs1_rel_vy,
     obs2_rel_x, obs2_rel_y, obs2_rel_vx, obs2_rel_vy]
    """

    # --- 1. 目标状态 (以机器人为中心) ---
    dx_goal = target_x - robot_x
    dy_goal = target_y - robot_y

    goal_dist = np.hypot(dx_goal, dy_goal)

    # 目标在机器人坐标系下的角度
    angle_to_goal = np.arctan2(dy_goal, dx_goal)
    # 目标相对于机器人航向的角度
    relative_goal_angle = wrap_to_pi(angle_to_goal - robot_theta)

    goal_cos = np.cos(relative_goal_angle)
    goal_sin = np.sin(relative_goal_angle)

    state = [goal_dist, goal_cos, goal_sin]

    # --- 2. 障碍物状态 (以机器人为中心) ---

    # 旋转矩阵：将世界坐标系转为机器人坐标系
    cos_theta = np.cos(-robot_theta)
    sin_theta = np.sin(-robot_theta)

    for obs in obstacles:
        # 世界坐标系下的相对位置和速度
        rel_x_world = obs.x - robot_x
        rel_y_world = obs.y - robot_y
        rel_vx_world = obs.vx - robot_vx
        rel_vy_world = obs.vy - robot_vy

        # 转换为机器人坐标系
        # (x', y') = (x*cos - y*sin, x*sin + y*cos)
        rel_x_robot = rel_x_world * cos_theta - rel_y_world * sin_theta
        rel_y_robot = rel_x_world * sin_theta + rel_y_world * cos_theta

        rel_vx_robot = rel_vx_world * cos_theta - rel_vy_world * sin_theta
        rel_vy_robot = rel_vx_world * sin_theta + rel_vy_world * cos_theta

        state.extend([rel_x_robot, rel_y_robot, rel_vx_robot, rel_vy_robot])

    # 补齐 (如果障碍物少于 NUM_RANDOM_OBSTACLES)
    # (在我们的例子中 NUM_RANDOM_OBSTACLES=2，所以这部分是固定的)
    # (如果障碍物是动态数量，这里需要补零)

    return np.array(state), relative_goal_angle


# ---------------------------------------------------------------------
# 🌟 LTC Actor-Critic 网络 (与之前相同，保持不变)
# ---------------------------------------------------------------------

class ActorLTC(nn.Module):
    def __init__(self, wiring, input_dim, action_dim):
        super(ActorLTC, self).__init__()
        self.hidden_dim = wiring.neuron_count
        self.cell = LTCCell(wiring)
        self.action_head = nn.Linear(wiring.output_dim, action_dim)

    def forward(self, x, hidden_state):
        ltc_output, next_hidden_state = self.cell(x, hidden_state)
        logits = self.action_head(ltc_output)
        dist = torch.distributions.Categorical(logits=logits)
        return dist, next_hidden_state


class CriticLTC(nn.Module):
    def __init__(self, wiring, input_dim):
        super(CriticLTC, self).__init__()
        self.hidden_dim = wiring.neuron_count
        self.cell = LTCCell(wiring)
        self.value_head = nn.Linear(wiring.output_dim, 1)

    def forward(self, x, hidden_state):
        ltc_output, next_hidden_state = self.cell(x, hidden_state)
        value = self.value_head(ltc_output)
        return value, next_hidden_state


# ---------------------------------------------------------------------
# 🌟 PPO 算法类 (与之前相同，保持不变)
# ---------------------------------------------------------------------

class PPO_LTC:
    def __init__(self, state_dim, action_dim, ltc_output_dim, ltc_hidden_dim,
                 actor_lr, critic_lr, lmbda, epochs, eps, gamma, device):

        self.device = device
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = eps

        actor_wiring = RandomWiring(state_dim, ltc_output_dim, ltc_hidden_dim)
        critic_wiring = RandomWiring(state_dim, ltc_output_dim, ltc_hidden_dim)

        self.actor = ActorLTC(actor_wiring, state_dim, action_dim).to(self.device)
        self.critic = CriticLTC(critic_wiring, state_dim).to(self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

    def take_action(self, state, actor_hidden):
        state = torch.as_tensor(state, dtype=torch.float).to(self.device).unsqueeze(0)
        dist, next_actor_hidden = self.actor(state, actor_hidden)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob.item(), next_actor_hidden

    def update(self, transition_dict):
        """
        关键修改：PPO 更新现在必须按顺序处理整个轨迹
        """
        # 🌟 步骤1: 先将列表高效地转换为单个Numpy数组
        np_states = np.array(transition_dict['states'])
        np_actions = np.array(transition_dict['actions'])
        np_rewards = np.array(transition_dict['rewards'])
        np_next_states = np.array(transition_dict['next_states'])
        np_dones = np.array(transition_dict['dones'])
        np_log_probs = np.array(transition_dict['log_probs'])

        # 🌟 步骤2: 现在从Numpy高效地转换为Tensor
        states = torch.as_tensor(np_states, dtype=torch.float).to(self.device)
        actions = torch.as_tensor(np_actions).view(-1, 1).to(self.device)
        rewards = torch.as_tensor(np_rewards, dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.as_tensor(np_next_states, dtype=torch.float).to(self.device)
        dones = torch.as_tensor(np_dones, dtype=torch.float).view(-1, 1).to(self.device)
        old_log_probs = torch.as_tensor(np_log_probs, dtype=torch.float).view(-1, 1).to(self.device)

        T = len(states)
        values = torch.zeros(T, 1).to(self.device)
        with torch.no_grad():
            h_c = torch.zeros(1, self.critic.hidden_dim).to(self.device)
            for t in range(T):
                v, h_c = self.critic(states[t].unsqueeze(0), h_c)
                values[t] = v
            v_T_plus_1, _ = self.critic(next_states[-1].unsqueeze(0), h_c)

        td_target = rewards + self.gamma * values.roll(-1, 0) * (1 - dones)
        td_target[-1] = rewards[-1] + self.gamma * v_T_plus_1 * (1 - dones[-1])

        td_delta = td_target - values
        advantage = rl_utils.compute_advantage(self.gamma, self.lmbda, td_delta.cpu()).to(self.device)

        for _ in range(self.epochs):
            h_a = torch.zeros(1, self.actor.hidden_dim).to(self.device)
            new_log_probs = []
            entropies = []
            for t in range(T):
                dist, h_a = self.actor(states[t].unsqueeze(0), h_a)
                new_log_probs.append(dist.log_prob(actions[t]))
                entropies.append(dist.entropy())

            new_log_probs = torch.stack(new_log_probs)
            entropy_loss = torch.stack(entropies).mean()

            h_c = torch.zeros(1, self.critic.hidden_dim).to(self.device)
            new_values = []
            for t in range(T):
                v, h_c = self.critic(states[t].unsqueeze(0), h_c)
                new_values.append(v.squeeze(0))
            new_values = torch.stack(new_values)

            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage

            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = F.mse_loss(new_values, td_target)

            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy_loss

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()


# --- 实例化 PPO_LTC Agent ---
agent = PPO_LTC(state_dim, action_dim, ltc_output_dim, ltc_hidden_dim,
                actor_lr, critic_lr, lmbda, epochs, eps, gamma, device)

# --- 初始化障碍物列表 ---
obstacles = []
for _ in range(NUM_RANDOM_OBSTACLES):
    obstacles.append(Obstacle(0, 0, 0, 0))  # 用虚拟值初始化

episode_rewards = []
recent_average_rewards_e = []
recent_average_rewards_e50 = []
reward_pre = 0
recent_average_reward_e50 = 0

# ---------------------------------------------------------------------
# 🌟 PPO 训练主循环 (已按新任务修改)
# ---------------------------------------------------------------------
for episode in tqdm(range(episodes)):

    # --- 1. 环境初始化 (随机起点、终点、障碍物) ---
    start_x = np.random.uniform(SPAWN_BOX[0], SPAWN_BOX[1])
    start_y = np.random.uniform(SPAWN_BOX[2], SPAWN_BOX[3])
    start_yaw = np.random.uniform(-np.pi, np.pi)

    target_x = np.random.uniform(SPAWN_BOX[0], SPAWN_BOX[1])
    target_y = np.random.uniform(SPAWN_BOX[2], SPAWN_BOX[3])

    # 确保障碍物和目标不会太近
    while np.hypot(start_x - target_x, start_y - target_y) < 1.0:
        target_x = np.random.uniform(SPAWN_BOX[0], SPAWN_BOX[1])
        target_y = np.random.uniform(SPAWN_BOX[2], SPAWN_BOX[3])

    for obs in obstacles:
        obs.reset()  # 重置障碍物
        # 确保障碍物不会刷在起点上
        while np.hypot(obs.x - start_x, obs.y - start_y) < R_obstacle * 4:
            obs.reset()

    x, y, theta = start_x, start_y, start_yaw
    vx, vy = 0.0, 0.0  # 初始速度为 0

    state, relative_goal_angle = calculate_state(x, y, theta, vx, vy, target_x, target_y, obstacles)

    # 记录上一帧到目标的距离，用于计算 shaping reward
    last_goal_dist = state[0]  # state[0] 就是 goal_dist

    episode_reward = 0
    transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': [], 'log_probs': []}
    success = 0

    # 初始化隐藏状态
    actor_hidden = torch.zeros(1, ltc_hidden_dim).to(device)

    # --- 2. 步进环境 ---
    for step in range(steps):
        done = False

        # 动作选择
        action, log_prob, next_actor_hidden = agent.take_action(state, actor_hidden)
        v_A = A[action]
        v_D = D[action]

        # 运动更新
        x, y, theta, vx, vy = update_movement(x, y, theta, v_D, v_A, DT)

        # 更新障碍物位置
        for obs in obstacles:
            obs.update()

        # 计算下一个状态
        next_state, relative_goal_angle = calculate_state(x, y, theta, vx, vy, target_x, target_y, obstacles)

        # 🌟 全新的奖励函数 🌟

        # 1. 成功检测
        dist_to_target = next_state[0]  # goal_dist
        if dist_to_target < TARGET_REACH_THRESH:
            reward_success = 10.0
            success = 1
            done = True
        else:
            reward_success = 0.0

        # 2. 碰撞检测
        collided = False
        min_dist_to_obs = float('inf')
        for obs in obstacles:
            dist = np.hypot(x - obs.x, y - obs.y)
            min_dist_to_obs = min(min_dist_to_obs, dist)
            if dist < (R_obstacle + R_robot):
                collided = True
                break

        if collided:
            reward_collision = -10.0
            done = True
        else:
            reward_collision = 0.0

        # 3. 目标塑形奖励 (关键！)
        # 奖励 = (上一帧的距离 - 这一帧的距离)，靠近则为正
        reward_shaping = (last_goal_dist - dist_to_target) * 0.5

        # 4. 航向奖励 (惩罚没有朝向目标)
        # relative_goal_angle 是 state 计算时返回的
        reward_heading = -0.1 * abs(relative_goal_angle)

        # 5. 避障奖励 (惩罚靠近障碍物)
        if min_dist_to_obs < 0.5:
            reward_obstacle = -0.05 * (1 - min_dist_to_obs / 0.5)
        else:
            reward_obstacle = 0.0

        # 6. 步骤惩罚 (时间成本)
        reward_step = -0.02

        # 总奖励
        reward = (reward_success + reward_collision + reward_shaping +
                  reward_heading + reward_obstacle + reward_step)

        # 收集经验
        transition_dict['states'].append(state)
        transition_dict['actions'].append(action)
        transition_dict['next_states'].append(next_state)
        transition_dict['rewards'].append(reward)
        transition_dict['dones'].append(done)
        transition_dict['log_probs'].append(log_prob)

        episode_reward += reward
        state = np.copy(next_state)
        last_goal_dist = dist_to_target  # 更新上一帧距离
        actor_hidden = next_actor_hidden  # 更新隐藏状态

        if done:
            actor_hidden = torch.zeros(1, ltc_hidden_dim).to(device)  # 重置记忆

        if done or step == (steps - 1):
            if success == 1:
                reward_e = 1
            else:
                reward_e = 0
            recent_average_rewards_e.append(reward_e)
            break

    # --- 3. PPO 更新 ---
    if len(transition_dict['states']) > 0:
        agent.update(transition_dict)

    # --- 4. 训练日志记录 ---
    episode_rewards.append(episode_reward)
    if episode <= 50:
        recent_average_reward_e50 = sum(recent_average_rewards_e) / (episode + 1)
        recent_average_rewards_e50.append(recent_average_reward_e50)
    else:
        recent_average_reward_e50 = sum(recent_average_rewards_e[-50:]) / 50
        recent_average_rewards_e50.append(recent_average_reward_e50)

    if episode % 100 == 0 and episode > 0:
        mean_reward = np.mean(recent_average_rewards_e[-100:])
        print(f"Episode: {episode}, Mean Success Rate (last 100): {mean_reward:.2f}")

# --- 训练结束后的结果绘图 ---
plt.figure(figsize=(6, 5))
episodes_count = len(recent_average_rewards_e50)
plt.plot(range(episodes_count), recent_average_rewards_e50, 'b', label='recent_average_rewards (50)')
plt.plot([0, episodes_count], [0, 0], 'k--')
plt.plot([0, episodes_count], [1, 1], 'k--')
plt.plot([0, episodes_count], [0.95, 0.95], 'r--')
plt.xlabel('Episode', fontsize=14)
plt.ylabel('Success rate', fontsize=14)
plt.title('LTC-PPO Success Rate (Dynamic Obstacle Avoidance)')
plt.tick_params(axis='x', labelsize=12)
plt.tick_params(axis='y', labelsize=12)
plt.show()

# --- 保存模型 ---
torch.save(agent.actor.state_dict(), 'ltc_ppo_actor_dynamic.pth')
torch.save(agent.critic.state_dict(), 'ltc_ppo_critic_dynamic.pth')
print("Trained LTC-PPO models saved.")