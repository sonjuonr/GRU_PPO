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
import time

# 🌟 1. Import config and TensorBoard
import config
from torch.utils.tensorboard import SummaryWriter

matplotlib.use("TkAgg")


# ---------------------------------------------------------------------
# 🌟 Auxiliary Class: Rollout Buffer (Necessary for N-Step Update)
# ---------------------------------------------------------------------
class RolloutBuffer:
    def __init__(self, buffer_size, state_dim, hidden_dim, device):
        self.buffer_size = buffer_size
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.device = device
        self.clear()

    def clear(self):
        # Stores N steps of data
        self.states = torch.zeros((self.buffer_size, self.state_dim), dtype=torch.float32)
        self.actions = torch.zeros((self.buffer_size, 1), dtype=torch.int64)
        self.log_probs = torch.zeros((self.buffer_size, 1), dtype=torch.float32)
        self.rewards = torch.zeros((self.buffer_size, 1), dtype=torch.float32)
        self.dones = torch.zeros((self.buffer_size, 1), dtype=torch.float32)
        self.values = torch.zeros((self.buffer_size, 1), dtype=torch.float32)
        # 🌟 Must store the RNN's hidden state
        self.h_actor = torch.zeros((self.buffer_size, self.hidden_dim), dtype=torch.float32)

        self.ptr = 0  # Buffer pointer

    def add(self, state, action, log_prob, reward, done, value, h_actor):
        if self.ptr >= self.buffer_size:
            print("Warning: Rollout buffer overflow")
            return

        self.states[self.ptr] = torch.as_tensor(state, device='cpu')
        self.actions[self.ptr] = torch.as_tensor([action], device='cpu')
        self.log_probs[self.ptr] = torch.as_tensor([log_prob], device='cpu')
        self.rewards[self.ptr] = torch.as_tensor([reward], device='cpu')
        self.dones[self.ptr] = torch.as_tensor([done], device='cpu')
        self.values[self.ptr] = torch.as_tensor([value], device='cpu')
        self.h_actor[self.ptr] = h_actor.squeeze(0).to('cpu')

        self.ptr += 1

    def compute_returns_and_advantages(self, last_value, gamma, lmbda):
        """Calculates GAE (Generalized Advantage Estimation)"""
        last_value = last_value.to('cpu')
        last_gae_lam = 0

        # We need advantages and returns
        self.advantages = torch.zeros_like(self.rewards)
        self.returns = torch.zeros_like(self.rewards)

        for t in reversed(range(self.buffer_size)):
            if t == self.buffer_size - 1:
                next_non_terminal = 1.0 - self.dones[t]  # Check if the last step is done
                next_values = last_value
            else:
                next_non_terminal = 1.0 - self.dones[t + 1]
                next_values = self.values[t + 1]

            # GAE Calculation
            delta = self.rewards[t] + gamma * next_values * next_non_terminal - self.values[t]
            self.advantages[t] = last_gae_lam = delta + gamma * lmbda * next_non_terminal * last_gae_lam

        # Returns = Advantages + Values
        self.returns = self.advantages + self.values

    def get_batches(self, minibatch_size):
        """Creates sequential mini-batches for RNN"""
        # N-Step Update: We divide the entire 2048 steps of data into N chunks
        # (This is a simplified implementation, not handling cross-episode boundaries)
        num_minibatches = self.buffer_size // minibatch_size

        # Send all data to device
        self.states = self.states.to(self.device)
        self.actions = self.actions.to(self.device)
        self.log_probs = self.log_probs.to(self.device)
        self.advantages = self.advantages.to(self.device)
        self.returns = self.returns.to(self.device)
        self.h_actor = self.h_actor.to(self.device)

        indices = np.arange(self.buffer_size)
        # Note: For RNNs, we should not fully shuffle (randomize)
        # We "randomly" select the starting points here, but maintain order within the minibatch
        # (This is a complex topic, here we use a simplified "randomly ordered sequential chunk")
        np.random.shuffle(indices.reshape(-1, minibatch_size))

        for start in range(0, self.buffer_size, minibatch_size):
            end = start + minibatch_size
            batch_indices = indices[start:end]

            yield (
                self.states[batch_indices],
                self.actions[batch_indices],
                self.log_probs[batch_indices],
                self.advantages[batch_indices],
                self.returns[batch_indices],
                self.h_actor[batch_indices[0]].unsqueeze(0)  # 🌟 Use the first hidden state of this chunk as the initial state
            )


# ---------------------------------------------------------------------
# --- Auxiliary Functions, Obstacle, State Calculation (imported config) ---
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


class Obstacle:
    def __init__(self, x, y, vx, vy, speed_range=(0.8, 0.15)):
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
# 🌟 Actor-Critic Networks 
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
# 🌟 PPO Algorithm Class (Refactored to support N-Step Update)
# ---------------------------------------------------------------------
class PPO_GRU:
    def __init__(self, state_dim, action_dim, gru_hidden_dim,
                 actor_lr, critic_lr, lmbda, epochs, eps, gamma, device):
        self.device = device
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = eps

        self.actor = ActorGRU(state_dim, gru_hidden_dim, action_dim).to(self.device)
        self.critic = CriticGRU(state_dim, gru_hidden_dim).to(self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

    def take_action(self, state, actor_hidden):
        """(Used for Rollout) - Outputs action with gradients"""
        state = torch.as_tensor(state, dtype=torch.float).to(self.device).unsqueeze(0)

        # Evaluate Actor
        self.actor.eval()  # Ensure eval mode during rollout
        with torch.no_grad():
            dist, next_actor_hidden = self.actor(state, actor_hidden)
        self.actor.train()  # Switch back to train mode

        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob.item(), next_actor_hidden

    def get_value(self, state, critic_hidden):
        """(Used for Rollout) - Gets the value of the current state"""
        state = torch.as_tensor(state, dtype=torch.float).to(self.device).unsqueeze(0)

        self.critic.eval()
        with torch.no_grad():
            value, next_critic_hidden = self.critic(state, critic_hidden)
        self.critic.train()

        return value.item(), next_critic_hidden

    def update(self, buffer, writer, global_step):
        """
        N-Step Update:
        Uses N steps of buffer data, trains for K Epochs, each Epoch divided into M Mini-Batches
        """

        # Stores loss for TensorBoard logging
        actor_losses = []
        critic_losses = []
        entropies = []

        for _ in range(self.epochs):
            for batch in buffer.get_batches(config.MINIBATCH_SIZE):
                (
                    mb_states,
                    mb_actions,
                    mb_old_log_probs,
                    mb_advantages,
                    mb_returns,
                    mb_h_actor_initial,  # Initial hidden state
                ) = batch

                # --- Recalculate Actor (Replay) ---
                # We need to replay the entire mini-batch sequence
                T = len(mb_states)
                new_log_probs = []
                new_entropies = []
                h_a = mb_h_actor_initial
                for t in range(T):
                    dist, h_a = self.actor(mb_states[t].unsqueeze(0), h_a)
                    new_log_probs.append(dist.log_prob(mb_actions[t]))
                    new_entropies.append(dist.entropy())

                new_log_probs = torch.stack(new_log_probs)
                entropy_loss = torch.stack(new_entropies).mean()

                # --- Recalculate Critic (Replay) ---
                h_c = mb_h_actor_initial.detach()  # Assume Actor/Critic share state
                new_values = []
                for t in range(T):
                    v, h_c = self.critic(mb_states[t].unsqueeze(0), h_c)
                    new_values.append(v.squeeze(0))
                new_values = torch.stack(new_values)

                # --- PPO Loss Calculation ---
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * mb_advantages

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = F.mse_loss(new_values, mb_returns)

                loss = (actor_loss +
                        0.5 * critic_loss -
                        config.ENTROPY_COEF * entropy_loss)

                # --- Gradient Update ---
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), config.GRAD_CLIP_MAX_NORM)
                nn.utils.clip_grad_norm_(self.critic.parameters(), config.GRAD_CLIP_MAX_NORM)
                self.actor_optimizer.step()
                self.critic_optimizer.step()

                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())
                entropies.append(entropy_loss.item())

        # 🌟 Log (Average Value)
        writer.add_scalar("Loss/Actor_Loss", np.mean(actor_losses), global_step)
        writer.add_scalar("Loss/Critic_Loss", np.mean(critic_losses), global_step)
        writer.add_scalar("Metrics/Entropy", np.mean(entropies), global_step)


# --- Instantiate Agent and Environment ---
# 🌟 All using config
device = config.DEVICE
agent = PPO_GRU(config.STATE_DIM, config.ACTION_DIM, config.GRU_HIDDEN_DIM,
                config.ACTOR_LR, config.CRITIC_LR, config.LMBDA, config.EPOCHS,
                config.EPS, config.GAMMA, device)

buffer = RolloutBuffer(config.ROLLOUT_STEPS, config.STATE_DIM, config.GRU_HIDDEN_DIM, device)

obstacles = []
for _ in range(config.NUM_RANDOM_OBSTACLES):
    obstacles.append(Obstacle(0, 0, 0, 0))

# 🌟 Initialize TensorBoard
writer = SummaryWriter(config.LOG_DIR)
print(f"Logging to {config.LOG_DIR}, using device {device}")
print(f"Total timesteps: {config.TOTAL_TIMESTEPS}, Rollout size: {config.ROLLOUT_STEPS}")

# ---------------------------------------------------------------------
# 🌟 Step 4: PPO Training Main Loop (N-Step Version)
# ---------------------------------------------------------------------

# --- Initialize Environment State ---
start_x = np.random.uniform(config.SPAWN_BOX[0], config.SPAWN_BOX[1])
start_y = np.random.uniform(config.SPAWN_BOX[2], config.SPAWN_BOX[3])
start_yaw = np.random.uniform(-np.pi, np.pi)
target_x = np.random.uniform(config.SPAWN_BOX[0], config.SPAWN_BOX[1])
target_y = np.random.uniform(config.SPAWN_BOX[2], config.SPAWN_BOX[3])
for obs in obstacles:
    obs.reset()
    while np.hypot(obs.x - start_x, obs.y - start_y) < config.R_OBSTACLE * 4:
        obs.reset()

x, y, theta = start_x, start_y, start_yaw
vx, vy = 0.0, 0.0
state, _ = calculate_state(x, y, theta, vx, vy, target_x, target_y, obstacles)

# --- Initialize RNN Hidden States ---
actor_hidden = torch.zeros(1, config.GRU_HIDDEN_DIM).to(device)
critic_hidden = torch.zeros(1, config.GRU_HIDDEN_DIM).to(device)  # We need it to get the value

# --- Initialize Log Trackers ---
global_step = 0
num_updates = config.TOTAL_TIMESTEPS // config.ROLLOUT_STEPS
start_time = time.time()

# --- Main Training Loop ---
for update_num in tqdm(range(1, num_updates + 1)):

    # Clear the buffer, prepare to collect N steps of data
    buffer.clear()

    # Temporary trackers to record reward during rollout
    ep_rewards = []
    ep_successes = []
    ep_lengths = []
    # 🌟 Fix 1: Add new lists
    ep_rew_success = []
    ep_rew_collision = []
    ep_rew_shaping = []
    ep_rew_heading = []
    ep_rew_obstacle = []
    ep_rew_step = []

    current_episode_reward = 0
    current_episode_len = 0
    # 🌟 Fix 2: Add new accumulators
    current_ep_rew_success = 0
    current_ep_rew_collision = 0
    current_ep_rew_shaping = 0
    current_ep_rew_heading = 0
    current_ep_rew_obstacle = 0
    current_ep_rew_step = 0

    # ---------------------------------
    # 1. ROLLOUT (Data Collection)
    # ---------------------------------
    for step in range(config.ROLLOUT_STEPS):
        global_step += 1
        current_episode_len += 1

        # --- Action Selection ---
        action, log_prob, next_actor_hidden = agent.take_action(state, actor_hidden)
        value, next_critic_hidden = agent.get_value(state, critic_hidden)  # Must get the value

        v_A = config.A[action]
        v_D = config.D[action]

        # --- Environment Step ---
        x, y, theta, vx, vy = update_movement(x, y, theta, v_D, v_A, config.DT)
        for obs in obstacles:
            obs.update()
        next_state, relative_goal_angle = calculate_state(x, y, theta, vx, vy, target_x, target_y, obstacles)

        # --- Reward Calculation (Using config) ---
        dist_to_target = next_state[0]
        last_goal_dist = state[0]

        # --- Initialize default values ---
        done = False
        success = 0
        reward_collision = 0.0
        reward_success = 0.0

        # --- 1. Priority Check for Collision ---
        collided = False
        min_dist_to_obs = float('inf')
        for obs in obstacles:
            dist = np.hypot(x - obs.x, y - obs.y)
            min_dist_to_obs = min(min_dist_to_obs, dist)
            if dist < (config.R_OBSTACLE + config.R_ROBOT):
                collided = True
                break

        if collided:
            # Failure: Collision occurred (Highest priority)
            reward_collision = config.REWARD_COLLISION
            done = True
            success = 0  # Explicitly mark as unsuccessful

            # [!!! Crucial Fix !!!]
            # When the episode ends, all shaping rewards must be 0
            reward_shaping = 0.0
            reward_heading = 0.0
            reward_obstacle = 0.0
            reward_step = 0.0

        elif dist_to_target < config.TARGET_REACH_THRESH:
            # Success: Reached target (Second priority)
            reward_success = config.REWARD_SUCCESS
            done = True
            success = 1  # Explicitly mark as successful

            # [!!! Crucial Fix !!!]
            reward_shaping = 0.0
            reward_heading = 0.0
            reward_obstacle = 0.0
            reward_step = 0.0

        elif current_episode_len >= config.STEPS_PER_EPISODE:
            # Failure: Timeout (Third priority)
            done = True
            success = 0  # Explicitly mark as unsuccessful

            # [!!! Crucial Fix !!!]
            reward_shaping = 0.0
            reward_heading = 0.0
            reward_obstacle = 0.0
            reward_step = 0.0

        else:
            # Episode continues: Calculate all shaping rewards
            done = False
            success = 0

            # [!!! Crucial Fix: All shaping rewards are calculated only here !!!]
            reward_shaping = (last_goal_dist - dist_to_target) * config.REWARD_SHAPING_WEIGHT
            reward_heading = config.REWARD_HEADING_WEIGHT * abs(relative_goal_angle)
            if min_dist_to_obs < 1.0:  # Ensure the danger zone radius is 1.0
                reward_obstacle = config.REWARD_OBSTACLE_WEIGHT * (1.0 - min_dist_to_obs / 1.0)
            else:
                reward_obstacle = 0.0
            reward_step = config.REWARD_STEP_WEIGHT

            # Total Reward (This sum is now logically correct)
        reward = (reward_success + reward_collision + reward_shaping +
                  reward_heading + reward_obstacle + reward_step)

        # 🌟 Fix 3: Accumulate separately
        current_ep_rew_success += reward_success
        current_ep_rew_collision += reward_collision
        current_ep_rew_shaping += reward_shaping
        current_ep_rew_heading += reward_heading
        current_ep_rew_obstacle += reward_obstacle
        current_ep_rew_step += reward_step

        current_episode_reward += reward

        # --- Store to Buffer ---
        buffer.add(state, action, log_prob, reward, done, value, actor_hidden)

        # Update state
        state = np.copy(next_state)
        actor_hidden = next_actor_hidden
        critic_hidden = next_critic_hidden  # Critic hidden state must also be updated

        # --- If episode ends (Done) ---
        if done:
            # 🌟 Fix 4a: Log all components
            ep_rewards.append(current_episode_reward)
            ep_successes.append(success)
            ep_lengths.append(current_episode_len)

            ep_rew_success.append(current_ep_rew_success)
            ep_rew_collision.append(current_ep_rew_collision)
            ep_rew_shaping.append(current_ep_rew_shaping)
            ep_rew_heading.append(current_ep_rew_heading)
            ep_rew_obstacle.append(current_ep_rew_obstacle)
            ep_rew_step.append(current_ep_rew_step)

            # --- Reset Environment ---
            start_x = np.random.uniform(config.SPAWN_BOX[0], config.SPAWN_BOX[1])
            start_y = np.random.uniform(config.SPAWN_BOX[2], config.SPAWN_BOX[3])
            start_yaw = np.random.uniform(-np.pi, np.pi)
            target_x = np.random.uniform(config.SPAWN_BOX[0], config.SPAWN_BOX[1])
            target_y = np.random.uniform(config.SPAWN_BOX[2], config.SPAWN_BOX[3])
            for obs in obstacles:
                obs.reset()

            x, y, theta = start_x, start_y, start_yaw
            vx, vy = 0.0, 0.0
            state, _ = calculate_state(x, y, theta, vx, vy, target_x, target_y, obstacles)

            # 🌟 Reset RNN Hidden States
            actor_hidden = torch.zeros(1, config.GRU_HIDDEN_DIM).to(device)
            critic_hidden = torch.zeros(1, config.GRU_HIDDEN_DIM).to(device)

            # 🌟 Fix 4b: Reset all accumulators
            current_episode_reward = 0
            current_episode_len = 0

            current_ep_rew_success = 0
            current_ep_rew_collision = 0
            current_ep_rew_shaping = 0
            current_ep_rew_heading = 0
            current_ep_rew_obstacle = 0
            current_ep_rew_step = 0

    # ---------------------------------
    # 2. GAE Calculation and PPO Update
    # ---------------------------------

    # Get the value of the last step in N steps, used for GAE
    with torch.no_grad():
        last_value, _ = agent.get_value(state, critic_hidden)

    # Calculate GAE and Returns
    buffer.compute_returns_and_advantages(torch.tensor([last_value]).to(device), config.GAMMA, config.LMBDA)

    # Perform PPO Update
    agent.update(buffer, writer, global_step)

    # --- 3. Log (Rollout Level) ---
    sps = int(global_step / (time.time() - start_time))
    writer.add_scalar("Metrics/SPS (Steps Per Second)", sps, global_step)

    if len(ep_rewards) > 0:  # Only log if episodes ended during the rollout
        writer.add_scalar("Episode/Mean_Reward", np.mean(ep_rewards), global_step)
        writer.add_scalar("Episode/Mean_Success_Rate", np.mean(ep_successes), global_step)
        writer.add_scalar("Episode/Mean_Length", np.mean(ep_lengths), global_step)

        # 🌟 Fix 5: Add logs for all reward components here
        writer.add_scalar("Reward_Components/Mean_Success", np.mean(ep_rew_success), global_step)
        writer.add_scalar("Reward_Components/Mean_Collision", np.mean(ep_rew_collision), global_step)
        writer.add_scalar("Reward_Components/Mean_Shaping", np.mean(ep_rew_shaping), global_step)
        writer.add_scalar("Reward_Components/Mean_Heading", np.mean(ep_rew_heading), global_step)
        writer.add_scalar("Reward_Components/Mean_Obstacle", np.mean(ep_rew_obstacle), global_step)
        writer.add_scalar("Reward_Components/Mean_Step_Penalty", np.mean(ep_rew_step), global_step)

    # Save model every 100 updates
    if update_num % 100 == 0:
        torch.save(agent.actor.state_dict(), f'gru_ppo_actor_{update_num}.pth')
        torch.save(agent.critic.state_dict(), f'gru_ppo_critic_{update_num}.pth')

# --- Training Ends ---
writer.close()
print("Training finished. Saving final models.")
torch.save(agent.actor.state_dict(), 'gru_ppo_actor_dynamic_final.pth')
torch.save(agent.critic.state_dict(), 'gru_ppo_critic_dynamic_final.pth')
