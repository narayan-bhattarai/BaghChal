# %%
%pip install matplotlib

# %%
# Enhanced BaaghChal Training with Visualization and Balanced Learning
import numpy as np
import random
import pickle
import time
import matplotlib.pyplot as plt
import random
from collections import defaultdict
# -------------------------------
# 1. Define the BaaghChal Environment (Improved Reward System)
# -------------------------------
class BaaghChalEnv:
    def __init__(self):
        self.reset()

    def reset(self):
        self.board = np.zeros((5, 5), dtype=int)
        self.tiger_positions = [(0, 0), (0, 4), (4, 0), (4, 4)]
        for i, j in self.tiger_positions:
            self.board[i, j] = 2
        self.goat_positions = []
        self.remaining_goats = 20
        self.phase = 'placement'
        self.current_player = 'goat'
        return self.get_state()

    def get_state(self):
        return self.board.copy()

    def get_valid_actions(self, player):
        actions = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                      (-1, -1), (-1, 1), (1, -1), (1, 1)]

        if player == 'goat':
            if self.phase == 'placement' and self.remaining_goats > 0:
                for i in range(5):
                    for j in range(5):
                        if self.board[i, j] == 0:
                            actions.append(('place', (i, j)))
            else:
                for i, j in self.goat_positions:
                    for di, dj in directions:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < 5 and 0 <= nj < 5 and self.board[ni, nj] == 0:
                            actions.append(('move', (i, j), (ni, nj)))
        elif player == 'tiger':
            for i, j in self.tiger_positions:
                for di, dj in directions:
                    ni, nj = i + di, j + dj
                    ni2, nj2 = i + 2*di, j + 2*dj
                    if 0 <= ni < 5 and 0 <= nj < 5 and self.board[ni, nj] == 0:
                        actions.append(('move', (i, j), (ni, nj)))
                    elif (0 <= ni2 < 5 and 0 <= nj2 < 5 and
                          self.board[ni, nj] == 1 and self.board[ni2, nj2] == 0):
                        actions.insert(0, ('jump', (i, j), (ni2, nj2)))  # prioritize jumps
        return actions

    def step(self, action):
        reward, done = 0, False
        player = self.current_player

        if player == 'goat':
            if action[0] == 'place':
                _, (i, j) = action
                self.board[i, j] = 1
                self.goat_positions.append((i, j))
                self.remaining_goats -= 1
                if self.remaining_goats == 0:
                    self.phase = 'movement'
            elif action[0] == 'move':
                _, (i, j), (ni, nj) = action
                self.board[i, j] = 0
                self.board[ni, nj] = 1
                self.goat_positions.remove((i, j))
                self.goat_positions.append((ni, nj))

            if not self.get_valid_actions('tiger'):
                reward = 1  # block reward
                done = True

        elif player == 'tiger':
            if action[0] == 'move':
                _, (i, j), (ni, nj) = action
                self.board[i, j] = 0
                self.board[ni, nj] = 2
                self.tiger_positions.remove((i, j))
                self.tiger_positions.append((ni, nj))
            elif action[0] == 'jump':
                _, (i, j), (ni, nj) = action
                ji, jj = i + (ni - i)//2, j + (nj - j)//2
                self.board[i, j] = 0
                self.board[ji, jj] = 0
                self.board[ni, nj] = 2
                self.tiger_positions.remove((i, j))
                self.tiger_positions.append((ni, nj))
                self.goat_positions.remove((ji, jj))
                reward = 2  # strong incentive to jump

        self.current_player = 'tiger' if player == 'goat' else 'goat'
        return self.get_state(), reward, done, {}

        reward = 0
        done = False
        player = self.current_player

        if player == 'goat':
            if action[0] == 'place':
                _, pos = action
                i, j = pos
                self.board[i, j] = 1
                self.goat_positions.append((i, j))
                self.remaining_goats -= 1
                if self.remaining_goats == 0:
                    self.phase = 'movement'
                reward = 0.05
            elif action[0] == 'move':
                _, from_pos, to_pos = action
                i, j = from_pos
                ni, nj = to_pos
                self.board[i, j] = 0
                self.board[ni, nj] = 1
                self.goat_positions.remove(from_pos)
                self.goat_positions.append((ni, nj))
                reward = 0.05
        elif player == 'tiger':
            if action[0] == 'move':
                _, from_pos, to_pos = action
                i, j = from_pos
                ni, nj = to_pos
                self.board[i, j] = 0
                self.board[ni, nj] = 2
                self.tiger_positions.remove(from_pos)
                self.tiger_positions.append((ni, nj))
                reward = -0.1
            elif action[0] == 'jump':
                _, from_pos, to_pos = action
                i, j = from_pos
                ni, nj = to_pos
                jumped_i = i + (ni - i) // 2
                jumped_j = j + (nj - j) // 2
                self.board[i, j] = 0
                self.board[ni, nj] = 2
                self.tiger_positions.remove(from_pos)
                self.tiger_positions.append((ni, nj))
                if (jumped_i, jumped_j) in self.goat_positions:
                    self.goat_positions.remove((jumped_i, jumped_j))
                    self.board[jumped_i, jumped_j] = 0
                    reward = 1

        tiger_moves = self.get_valid_actions('tiger')
        if not tiger_moves:
            done = True
            reward = 10 if player == 'goat' else -10

        self.current_player = 'tiger' if player == 'goat' else 'goat'
        return self.get_state(), reward, done, {}

# -------------------------------
# 2. QLearningAgent (same as before)
# -------------------------------
class QLearningAgent:
    def __init__(self, role, learning_rate=0.2, discount_factor=0.9, epsilon=0.5):
        self.q_table = {}
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.role = role

    def get_q(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def choose_action(self, state, valid_actions):
        if random.random() < self.epsilon:
            return random.choice(valid_actions)
        qs = [self.get_q(state, a) for a in valid_actions]
        max_q = max(qs) if qs else 0
        best_actions = [a for a, q in zip(valid_actions, qs) if q == max_q]
        return random.choice(best_actions) if best_actions else random.choice(valid_actions)

    def learn(self, state, action, reward, next_state, next_valid_actions, done):
        current_q = self.get_q(state, action)
        if done:
            target = reward
        else:
            next_qs = [self.get_q(next_state, a) for a in next_valid_actions]
            target = reward + self.discount_factor * (max(next_qs) if next_qs else 0)
        self.q_table[(state, action)] = current_q + self.learning_rate * (target - current_q)




# %%
import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple Neural Network for DQN

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    def __init__(self, role, state_size, action_space_size, lr=1e-3, gamma=0.9, epsilon=1.0, epsilon_decay=0.997, epsilon_min=0.01):
        self.role = role
        self.state_size = state_size
        self.action_space_size = action_space_size
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.gamma = gamma

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN(state_size, action_space_size).to(self.device)
        self.target_net = DQN(state_size, action_space_size).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.update_target_network()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def choose_action(self, state, valid_actions):
        if random.random() < self.epsilon:
            return random.choice(valid_actions)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.policy_net(state_tensor).detach().cpu().numpy().flatten()
        action_indices = {i: a for i, a in enumerate(valid_actions)}
        sorted_indices = sorted(action_indices, key=lambda i: q_values[i], reverse=True)
        return action_indices[sorted_indices[0]]

    def learn(self, state, action_idx, reward, next_state, done):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        target = reward + (0 if done else self.gamma * torch.max(self.target_net(next_state_tensor)).item())
        target_tensor = torch.tensor(target, dtype=torch.float32).to(self.device)

        current_q = self.policy_net(state_tensor)[0, action_idx]
        loss = self.loss_fn(current_q, target_tensor)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# %%
# Initialize Environment and Agents
env = BaaghChalEnv()
state_size = 25  # 5x5 board flattened
action_space_size = 50  # Arbitrary, will map dynamically

goat_agent = DQNAgent(role='goat', state_size=state_size, action_space_size=action_space_size)
tiger_agent = DQNAgent(role='tiger', state_size=state_size, action_space_size=action_space_size)

num_episodes = 8000
target_update_frequency = 50

reward_history_goat = []
reward_history_tiger = []
episode_numbers = []
start_time = time.time()

for episode in range(1, num_episodes + 1):
    state = env.reset()
    state_repr = tuple(state.flatten())
    done = False
    total_reward_goat = 0
    total_reward_tiger = 0

    while not done:
        current_player = env.current_player
        valid_actions = env.get_valid_actions(current_player)
        if not valid_actions:
            break

        action = goat_agent.choose_action(state_repr, valid_actions) if current_player == 'goat' else tiger_agent.choose_action(state_repr, valid_actions)

        next_state, reward, done, _ = env.step(action)
        next_state_repr = tuple(next_state.flatten())
        
        action_idx = random.randint(0, len(valid_actions)-1)  # crude mapping (optimize later)

        if current_player == 'goat':
            goat_agent.learn(state_repr, action_idx, reward, next_state_repr, done)
            total_reward_goat += reward
        else:
            tiger_agent.learn(state_repr, action_idx, reward, next_state_repr, done)
            total_reward_tiger += reward

        state_repr = next_state_repr

    # Update target network every few episodes
    if episode % target_update_frequency == 0:
        goat_agent.update_target_network()
        tiger_agent.update_target_network()

    reward_history_goat.append(total_reward_goat)
    reward_history_tiger.append(total_reward_tiger)
    episode_numbers.append(episode)

    if episode % 100 == 0:
        elapsed = time.time() - start_time
        print(f"[DQN] Episode {episode}/{num_episodes} completed. Elapsed time: {elapsed:.2f}s")

print("✅ DQN Training Complete")

# %%
# Evaluation Run after loading saved DQN models
env = BaaghChalEnv()
state_size = 25
action_space_size = 50

# Load trained models
goat_agent = DQNAgent(role='goat', state_size=state_size, action_space_size=action_space_size)
tiger_agent = DQNAgent(role='tiger', state_size=state_size, action_space_size=action_space_size)

goat_agent.policy_net.load_state_dict(torch.load('goat_dqn_model.pt'))
tiger_agent.policy_net.load_state_dict(torch.load('tiger_dqn_model.pt'))

goat_agent.policy_net.eval()
tiger_agent.policy_net.eval()

# Collect metrics
reward_history_goat = []
reward_history_tiger = []
win_history = {'goat': 0, 'tiger': 0, 'draw': 0}
episode_numbers = []

num_eval_episodes = 8000

for episode in range(1, num_eval_episodes + 1):
    state = env.reset()
    state_repr = tuple(state.flatten())
    done = False
    total_reward_goat = 0
    total_reward_tiger = 0

    while not done:
        current_player = env.current_player
        valid_actions = env.get_valid_actions(current_player)
        if not valid_actions:
            break

        action = goat_agent.choose_action(state_repr, valid_actions) if current_player == 'goat' else tiger_agent.choose_action(state_repr, valid_actions)

        next_state, reward, done, _ = env.step(action)
        next_state_repr = tuple(next_state.flatten())

        if current_player == 'goat':
            total_reward_goat += reward
        else:
            total_reward_tiger += reward

        state_repr = next_state_repr

    # Update metrics
    if total_reward_goat > total_reward_tiger:
        win_history['goat'] += 1
    elif total_reward_tiger > total_reward_goat:
        win_history['tiger'] += 1
    else:
        win_history['draw'] += 1

    reward_history_goat.append(total_reward_goat)
    reward_history_tiger.append(total_reward_tiger)
    episode_numbers.append(episode)

print("✅ Evaluation finished. You can now plot metrics!")

# %%
# Save the trained DQN models
torch.save(goat_agent.policy_net.state_dict(), 'goat_dqn_model.pt')
torch.save(tiger_agent.policy_net.state_dict(), 'tiger_dqn_model.pt')

print("✅ DQN models saved successfully!")

# %%
def plot_rewards_dqn():
    plt.figure()
    plt.plot(episode_numbers, reward_history_goat, label='Goat Rewards')
    plt.plot(episode_numbers, reward_history_tiger, label='Tiger Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Reward Trend - DQN')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_win_distribution_dqn():
    roles = list(win_history.keys())
    wins = list(win_history.values())
    plt.figure()
    plt.bar(roles, wins, color=['green', 'orange', 'gray'])
    plt.title('Win Distribution After Training - DQN')
    plt.xlabel('Role')
    plt.ylabel('Number of Wins')
    plt.grid(True, axis='y')
    plt.show()

def plot_average_rewards_dqn():
    avg_goat = [np.mean(reward_history_goat[max(0, i - 100):i+1]) for i in range(len(reward_history_goat))]
    avg_tiger = [np.mean(reward_history_tiger[max(0, i - 100):i+1]) for i in range(len(reward_history_tiger))]
    plt.figure()
    plt.plot(episode_numbers, avg_goat, label='Goat (Moving Avg)')
    plt.plot(episode_numbers, avg_tiger, label='Tiger (Moving Avg)')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward (Last 100)')
    plt.title('Moving Average Reward Over Time - DQN')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_epsilon_decay_dqn(initial_goat=1.0, initial_tiger=1.0, decay=0.995):
    eps_goat = [max(0.01, initial_goat * (decay ** i)) for i in range(num_episodes)]
    eps_tiger = [max(0.01, initial_tiger * (decay ** i)) for i in range(num_episodes)]
    plt.figure()
    plt.plot(episode_numbers, eps_goat, label='Goat Epsilon')
    plt.plot(episode_numbers, eps_tiger, label='Tiger Epsilon')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    plt.title('Exploration Rate Decay - DQN')
    plt.legend()
    plt.grid(True)
    plt.show()

plot_rewards_dqn()
plot_win_distribution_dqn()
plot_average_rewards_dqn()
plot_epsilon_decay_dqn()

# %%

# -------------------------------
# 3. ESarsaAgent (same as before)
# -------------------------------
def default_q_values():
    return defaultdict(float)

class ESarsaAgent:
    def __init__(self, role, learning_rate=0.1, epsilon=0.1, gamma=0.9):
        self.role = role  # 'goat' or 'tiger'
        self.alpha = learning_rate
        self.epsilon = epsilon
        self.gamma = gamma
        self.q_table = defaultdict(default_q_values)  # ✅ Fix applied here

    def choose_action(self, state, valid_actions):
        if not valid_actions:
            return None
        
        if random.random() < self.epsilon:
            return random.choice(valid_actions)
        else:
            # Choose best action based on Q-values
            q_values = [self.q_table[state][a] for a in valid_actions]
            max_q = max(q_values)
            best_actions = [a for a in valid_actions if self.q_table[state][a] == max_q]
            return random.choice(best_actions)

    def learn(self, state, action, reward, next_state, next_valid_actions, done):
        q_predict = self.q_table[state][action]

        if done or not next_valid_actions:
            q_target = reward
        else:
            # Compute expected value over next state Q-values
            next_q_values = [self.q_table[next_state][a] for a in next_valid_actions]
            max_q = max(next_q_values)

            expected_value = 0
            for a in next_valid_actions:
                if self.q_table[next_state][a] == max_q:
                    prob = 1 - self.epsilon + (self.epsilon / len(next_valid_actions))
                else:
                    prob = self.epsilon / len(next_valid_actions)
                expected_value += prob * self.q_table[next_state][a]

            q_target = reward + self.gamma * expected_value

        self.q_table[state][action] += self.alpha * (q_target - q_predict)


# %%

# -------------------------------
# 3. Training with Win Balance Metrics
# -------------------------------
env = BaaghChalEnv()
goat_agent = QLearningAgent(role='goat', learning_rate=0.2, epsilon=0.5)
tiger_agent = QLearningAgent(role='tiger', learning_rate=0.1, epsilon=0.1)

num_episodes = 8000
win_history = {'goat': 0, 'tiger': 0, 'draw': 0}
reward_history_goat = []
reward_history_tiger = []
episode_numbers = []
start_time = time.time()

for episode in range(1, num_episodes + 1):
    state = env.reset()
    state_repr = tuple(state.flatten())
    done = False
    total_reward_goat = 0
    total_reward_tiger = 0

    while not done:
        current_player = env.current_player
        valid_actions = env.get_valid_actions(current_player)
        if not valid_actions:
            break

        action = goat_agent.choose_action(state_repr, valid_actions) if current_player == 'goat' else tiger_agent.choose_action(state_repr, valid_actions)
        next_state, reward, done, _ = env.step(action)
        next_state_repr = tuple(next_state.flatten())
        next_valid_actions = env.get_valid_actions(env.current_player)

        if current_player == 'goat':
            goat_agent.learn(state_repr, action, reward, next_state_repr, next_valid_actions, done)
            total_reward_goat += reward
        else:
            tiger_agent.learn(state_repr, action, reward, next_state_repr, next_valid_actions, done)
            total_reward_tiger += reward

        state_repr = next_state_repr

    if total_reward_tiger > total_reward_goat:
        win_history['tiger'] += 1
    elif total_reward_goat > total_reward_tiger:
        win_history['goat'] += 1
    else:
        win_history['draw'] += 1

    reward_history_goat.append(total_reward_goat)
    reward_history_tiger.append(total_reward_tiger)
    episode_numbers.append(episode)

    if episode > 1000:
        goat_agent.epsilon = max(0.01, goat_agent.epsilon * 0.995)
    tiger_agent.epsilon = max(0.01, tiger_agent.epsilon * 0.995)

    if episode % 100 == 0:
        elapsed = time.time() - start_time
        print(f"Episode {episode}/{num_episodes} completed. Elapsed time: {elapsed:.2f}s")



# %%

# -------------------------------
# 4. Save Trained Models
# -------------------------------
with open('goat_agent_q_table.pkl', 'wb') as f:
    pickle.dump(goat_agent.q_table, f)
with open('tiger_agent_q_table.pkl', 'wb') as f:
    pickle.dump(tiger_agent.q_table, f)

# %%
# -------------------------------
# 4. Sarsa Model Training
# -------------------------------

def default_q_values():
    return defaultdict(float)

class SarsaAgent:
    def __init__(self, role, learning_rate=0.1, epsilon=0.1, gamma=0.9):
        self.role = role
        self.alpha = learning_rate
        self.epsilon = epsilon
        self.gamma = gamma
        self.q_table = defaultdict(default_q_values)

    def choose_action(self, state, valid_actions):
        if not valid_actions:
            return None
        
        if random.random() < self.epsilon:
            return random.choice(valid_actions)
        else:
            q_values = [self.q_table[state][a] for a in valid_actions]
            max_q = max(q_values)
            best_actions = [a for a in valid_actions if self.q_table[state][a] == max_q]
            return random.choice(best_actions)

    def learn(self, state, action, reward, next_state, next_action, done):
        q_predict = self.q_table[state][action]
        q_target = reward

        if not done:
            q_target += self.gamma * self.q_table[next_state][next_action]

        self.q_table[state][action] += self.alpha * (q_target - q_predict)


# %%
env = BaaghChalEnv()
goat_agent = SarsaAgent(role='goat', learning_rate=0.2, epsilon=0.5, gamma=0.9)
tiger_agent = SarsaAgent(role='tiger', learning_rate=0.1, epsilon=0.1, gamma=0.9)

num_episodes = 8000
win_history = {'goat': 0, 'tiger': 0, 'draw': 0}
reward_history_goat = []
reward_history_tiger = []
episode_numbers = []
start_time = time.time()

for episode in range(1, num_episodes + 1):
    state = env.reset()
    state_repr = tuple(state.flatten())
    done = False
    total_reward_goat = 0
    total_reward_tiger = 0

    current_player = env.current_player
    valid_actions = env.get_valid_actions(current_player)
    action = goat_agent.choose_action(state_repr, valid_actions) if current_player == 'goat' else tiger_agent.choose_action(state_repr, valid_actions)

    while not done:
        if action is None:
            break  # No valid action to take

        next_state, reward, done, _ = env.step(action)
        next_state_repr = tuple(next_state.flatten())
        next_valid_actions = env.get_valid_actions(env.current_player)

        next_action = None
        if next_valid_actions:
            next_action = (
                goat_agent.choose_action(next_state_repr, next_valid_actions)
                if env.current_player == 'goat'
                else tiger_agent.choose_action(next_state_repr, next_valid_actions)
            )

        if current_player == 'goat':
            goat_agent.learn(state_repr, action, reward, next_state_repr, next_action, done)
            total_reward_goat += reward
        else:
            tiger_agent.learn(state_repr, action, reward, next_state_repr, next_action, done)
            total_reward_tiger += reward

        state_repr = next_state_repr
        action = next_action
        current_player = env.current_player

    if total_reward_tiger > total_reward_goat:
        win_history['tiger'] += 1
    elif total_reward_goat > total_reward_tiger:
        win_history['goat'] += 1
    else:
        win_history['draw'] += 1

    reward_history_goat.append(total_reward_goat)
    reward_history_tiger.append(total_reward_tiger)
    episode_numbers.append(episode)

    if episode > 1000:
        goat_agent.epsilon = max(0.01, goat_agent.epsilon * 0.995)
    tiger_agent.epsilon = max(0.01, tiger_agent.epsilon * 0.995)

    if episode % 100 == 0:
        elapsed = time.time() - start_time
        print(f"Episode {episode}/{num_episodes} completed. Elapsed time: {elapsed:.2f}s")


# %%
# -------------------------------
# 4. ESarsa Model Training
# -------------------------------
env = BaaghChalEnv()
goat_agent = ESarsaAgent(role='goat', learning_rate=0.2, epsilon=0.5, gamma=0.9)
tiger_agent = ESarsaAgent(role='tiger', learning_rate=0.1, epsilon=0.1, gamma=0.9)

num_episodes = 8000
win_history = {'goat': 0, 'tiger': 0, 'draw': 0}
reward_history_goat = []
reward_history_tiger = []
episode_numbers = []
start_time = time.time()

for episode in range(1, num_episodes + 1):
    state = env.reset()
    state_repr = tuple(state.flatten())
    done = False
    total_reward_goat = 0
    total_reward_tiger = 0

    while not done:
        current_player = env.current_player
        valid_actions = env.get_valid_actions(current_player)
        if not valid_actions:
            break

        action = goat_agent.choose_action(state_repr, valid_actions) if current_player == 'goat' else tiger_agent.choose_action(state_repr, valid_actions)
        next_state, reward, done, _ = env.step(action)
        next_state_repr = tuple(next_state.flatten())
        next_valid_actions = env.get_valid_actions(env.current_player)

        if current_player == 'goat':
            goat_agent.learn(state_repr, action, reward, next_state_repr, next_valid_actions, done)
            total_reward_goat += reward
        else:
            tiger_agent.learn(state_repr, action, reward, next_state_repr, next_valid_actions, done)
            total_reward_tiger += reward

        state_repr = next_state_repr

    if total_reward_tiger > total_reward_goat:
        win_history['tiger'] += 1
    elif total_reward_goat > total_reward_tiger:
        win_history['goat'] += 1
    else:
        win_history['draw'] += 1

    reward_history_goat.append(total_reward_goat)
    reward_history_tiger.append(total_reward_tiger)
    episode_numbers.append(episode)

    if episode > 1000:
        goat_agent.epsilon = max(0.01, goat_agent.epsilon * 0.995)
    tiger_agent.epsilon = max(0.01, tiger_agent.epsilon * 0.995)

    if episode % 100 == 0:
        elapsed = time.time() - start_time
        print(f"Episode {episode}/{num_episodes} completed. Elapsed time: {elapsed:.2f}s")

# %%
# -------------------------------
# 4. Save Trained Models for E-Sarsa
# -------------------------------
with open('goat_agent_e_sarsa.pkl', 'wb') as f:
    pickle.dump(goat_agent.q_table, f)

with open('tiger_agent_e_sarsa.pkl', 'wb') as f:    
    pickle.dump(tiger_agent.q_table, f)

print("Trained E-SARSA models saved successfully.")

# %%
import matplotlib.pyplot as plt

# -------------------------------
# 5. Visualization
# -------------------------------
def plot_rewards():
    plt.figure()
    plt.plot(episode_numbers, reward_history_goat, label='Goat Rewards')
    plt.plot(episode_numbers, reward_history_tiger, label='Tiger Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Reward Trend - ESarsa')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_win_distribution():
    roles = list(win_history.keys())
    wins = list(win_history.values())
    plt.figure()
    plt.bar(roles, wins, color=['green', 'orange', 'gray'])
    plt.title('Win Distribution After Training - ESarsa')
    plt.xlabel('Role')
    plt.ylabel('Number of Wins')
    plt.grid(True, axis='y')
    plt.show()

def plot_average_rewards():
    avg_goat = [np.mean(reward_history_goat[max(0, i - 100):i+1]) for i in range(len(reward_history_goat))]
    avg_tiger = [np.mean(reward_history_tiger[max(0, i - 100):i+1]) for i in range(len(reward_history_tiger))]
    plt.figure()
    plt.plot(episode_numbers, avg_goat, label='Goat (Moving Avg)')
    plt.plot(episode_numbers, avg_tiger, label='Tiger (Moving Avg)')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward (Last 100)')
    plt.title('Moving Average Reward Over Time - ESarsa')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_epsilon_decay():
    eps_goat = [max(0.01, 0.5 * (0.995 ** max(0, i - 1000))) for i in range(num_episodes)]
    eps_tiger = [max(0.01, 0.1 * (0.995 ** i)) for i in range(num_episodes)]
    plt.figure()
    plt.plot(episode_numbers, eps_goat, label='Goat Epsilon')
    plt.plot(episode_numbers, eps_tiger, label='Tiger Epsilon')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    plt.title('Exploration Rate Decay - ESarsa')
    plt.legend()
    plt.grid(True)
    plt.show()

plot_rewards()
plot_win_distribution()
plot_average_rewards()
plot_epsilon_decay()
print("Training completed and visualizations generated.")

# %%
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

true_labels = []      # Actual winner of each episode
predicted_labels = [] # Who the agent was (goat or tiger)

# During or after the SARSA training loop:
if total_reward_tiger > total_reward_goat:
    win_history['tiger'] += 1
    true_labels.append('tiger')
    predicted_labels.append('tiger')
elif total_reward_goat > total_reward_tiger:
    win_history['goat'] += 1
    true_labels.append('goat')
    predicted_labels.append('goat')
else:
    win_history['draw'] += 1
    true_labels.append('draw')
    predicted_labels.append('goat' if total_reward_goat >= total_reward_tiger else 'tiger')

# %%
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_confusion(true_labels, predicted_labels):
    labels = ['goat', 'tiger', 'draw']
    cm = confusion_matrix(true_labels, predicted_labels, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap='Blues', values_format='d')
    plt.title('Confusion Matrix: SARSA Agent')
    plt.grid(False)
    plt.show()

# %%
import numpy as np
import pickle
import random

# -------------------------------
# Environment Definition
# -------------------------------


# -------------------------------
# Q-Learning Agent
# -------------------------------

class QLearningAgent:
    def __init__(self, role, learning_rate=0.1, discount_factor=0.9, epsilon=0.0):
        self.q_table = {}
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.role = role

    def get_q(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def choose_action(self, state, valid_actions):
        qs = [self.get_q(state, a) for a in valid_actions]
        max_q = max(qs) if qs else 0
        best_actions = [a for a, q in zip(valid_actions, qs) if q == max_q]
        return random.choice(best_actions) if best_actions else random.choice(valid_actions)

# -------------------------------
# Load Trained Q-Tables
# -------------------------------

with open('goat_agent_q_table.pkl', 'rb') as f:
    goat_q_table = pickle.load(f)

with open('tiger_agent_q_table.pkl', 'rb') as f:
    tiger_q_table = pickle.load(f)

goat_agent = QLearningAgent(role='goat', epsilon=0.0)
goat_agent.q_table = goat_q_table

tiger_agent = QLearningAgent(role='tiger', epsilon=0.0)
tiger_agent.q_table = tiger_q_table

# -------------------------------
# Helper Function: Display Board with Emojis and Coordinates
# -------------------------------

def display_board_with_emojis(board):
    print("\n     0  1   2  3  4")  # Column numbers
    print("  -----------------")
    for i, row in enumerate(board):
        row_emojis = f"{i} |"  # Row number
        for cell in row:
            if cell == 0:
                row_emojis += " ⬜"
            elif cell == 1:
                row_emojis += " 🐐"
            elif cell == 2:
                row_emojis += " 🐅"
        print(row_emojis)
    print()

# -------------------------------
# Updated User Play Function (Better Action Representation)
# -------------------------------

def user_play():
    env = BaaghChalEnv()
    state = env.reset()
    state_repr = tuple(state.flatten())

    role = input("Choose your role (goat/tiger): ").strip().lower()
    if role not in ['goat', 'tiger']:
        print("Invalid role selected. Please choose 'goat' or 'tiger'.")
        return

    print("\n🎮 Game start!")
    while True:
        print(f"\nCurrent player turn: {env.current_player}")
        print("Board:")
        display_board_with_emojis(state)

        if env.current_player == role:
            valid_actions = env.get_valid_actions(role)
            if not valid_actions:
                print("No valid actions available for you. Game over.")
                break

            print("\nYour valid actions:")
            for idx, action in enumerate(valid_actions):
                if action[0] == "place":
                    print(f"{idx}: Place at {action[1]}")
                elif action[0] == "move":
                    print(f"{idx}: Move from {action[1]} to {action[2]}")
                elif action[0] == "jump":
                    print(f"{idx}: Jump from {action[1]} to {action[2]}")

            try:
                action_index = int(input("Enter the action index: "))
                action = valid_actions[action_index]
            except (ValueError, IndexError):
                print("Invalid selection. Try again.")
                continue
        else:
            valid_actions = env.get_valid_actions(env.current_player)
            if not valid_actions:
                print(f"No valid moves for {env.current_player}. Game over.")
                break

            if env.current_player == 'goat':
                action = goat_agent.choose_action(state_repr, valid_actions)
            else:
                action = tiger_agent.choose_action(state_repr, valid_actions)
            print(f"\nAgent ({env.current_player}) selects: {action}")

        state, reward, done, _ = env.step(action)
        state_repr = tuple(state.flatten())

        if done:
            print("\n🏁 Game over!")
            print("Final Board:")
            display_board_with_emojis(state)
            break


# -------------------------------
# Start Game
# -------------------------------

user_play()



