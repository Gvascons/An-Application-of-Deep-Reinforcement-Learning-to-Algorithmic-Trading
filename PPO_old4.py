# coding=utf-8

import math
import random
import copy
import datetime
import shutil
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from matplotlib import pyplot as plt
from collections import deque

# PPO hyperparameters
PPO_PARAMS = {
    'CLIP_EPSILON': 0.2,
    'VALUE_LOSS_COEF': 0.5,
    'ENTROPY_COEF': 0.01,
    'PPO_EPOCHS': 4,
    'BATCH_SIZE': 64,
    'GAMMA': 0.99,
    'GAE_LAMBDA': 0.95,
    'LEARNING_RATE': 3e-4,
    'MAX_GRAD_NORM': 0.5,
    'HIDDEN_SIZE': 256,
    'MEMORY_SIZE': 10000,
}

class PPONetwork(nn.Module):
    def __init__(self, input_size, num_actions):
        super().__init__()
        
        self.feature_dim = PPO_PARAMS['HIDDEN_SIZE']
        
        # Feature extraction layers with layer normalization
        self.shared = nn.Sequential(
            nn.Linear(input_size, self.feature_dim),
            nn.LayerNorm(self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.LayerNorm(self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, self.feature_dim // 2),
            nn.LayerNorm(self.feature_dim // 2),
            nn.ReLU(),
        )
        
        # Actor head with smaller architecture
        self.actor = nn.Sequential(
            nn.Linear(self.feature_dim // 2, self.feature_dim // 4),
            nn.ReLU(),
            nn.Linear(self.feature_dim // 4, num_actions)
        )
        
        # Critic head with smaller architecture
        self.critic = nn.Sequential(
            nn.Linear(self.feature_dim // 2, self.feature_dim // 4),
            nn.ReLU(),
            nn.Linear(self.feature_dim // 4, 1)
        )
        
        # Initialize weights with smaller values
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=0.1)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
            
        features = self.shared(x)
        action_probs = F.softmax(self.actor(features), dim=-1)
        value = self.critic(features)
        
        return action_probs, value

class PPO:
    """Implementation of PPO algorithm for trading"""
    def __init__(self, observationSpace, actionSpace, numberOfNeurons=256, dropout=0.2, marketSymbol='UNKNOWN'):
        """Initialize PPO agent with standard interface
        
        Args:
            observationSpace: Size of the observation space
            actionSpace: Size of the action space 
            numberOfNeurons: Number of neurons in hidden layers
            dropout: Dropout rate for regularization
            marketSymbol: Symbol of the market being traded
        """
        # Set random seeds
        random.seed(0)
        torch.manual_seed(0)
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Store spaces
        self.obs_space = observationSpace
        self.action_space = actionSpace
        
        # Initialize networks and optimizer with learning rate scheduling
        self.network = PPONetwork(self.obs_space, actionSpace).to(self.device)
        self.network.type(torch.float32)
        self.optimizer = optim.Adam(
            self.network.parameters(), 
            lr=PPO_PARAMS['LEARNING_RATE'],
            eps=1e-5
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='max', 
            factor=0.75,
            patience=10,
            threshold=0.01,
            min_lr=1e-5
        )
        
        # Initialize memory
        self.memory = deque(maxlen=PPO_PARAMS['MEMORY_SIZE'])
        
        # Initialize training tracking
        self.training_step = 0
        self.writer = None
        self.run_id = None
        self.figures_dir = None
        
        # Initialize performance tracking
        self.best_reward = float('-inf')
        self.trailing_rewards = deque(maxlen=100)

        # Additional tracking variables
        self.market_symbol = marketSymbol  # Store the market symbol
        self.prev_action = None
        self.prev_state = None

    def processState(self, state, coefficients):
        """Process the state according to the standard interface"""
        try:
            if torch.is_tensor(state):
                # Ensure the tensor has the correct size and dtype
                state = state.to(dtype=torch.float32)
                if state.size(-1) != self.obs_space:
                    if state.size(-1) < self.obs_space:
                        padding = torch.zeros(self.obs_space - state.size(-1), dtype=torch.float32)
                        state = torch.cat([state, padding])
                    else:
                        state = state[:self.obs_space]
                return state
                
            # Extract price data
            if isinstance(state[0], list):
                closes = np.array(state[0], dtype=np.float32)
                lows = np.array(state[1], dtype=np.float32)
                highs = np.array(state[2], dtype=np.float32)
                volumes = np.array(state[3], dtype=np.float32)
                
                # Calculate technical indicators
                returns = np.diff(closes) / closes[:-1]
                volatility = np.std(returns) if len(returns) > 1 else 0
                
                # Normalize features
                norm_closes = self.normalize(closes)
                norm_lows = self.normalize(lows)
                norm_highs = self.normalize(highs)
                norm_returns = self.normalize(np.append(returns, returns[-1]))  # Pad returns
                norm_volumes = self.normalize(volumes)
                
                # Combine features
                features = []
                features.extend(norm_closes)
                features.extend(norm_lows)
                features.extend(norm_highs)
                features.extend(norm_returns)
                features.extend([volatility])
                features.extend(norm_volumes)
                
                # Add position
                features.append(float(state[-1][0]) if isinstance(state[-1], list) else float(state[-1]))
                
                # Ensure features match observation space
                if len(features) < self.obs_space:
                    features.extend([0.0] * (self.obs_space - len(features)))
                elif len(features) > self.obs_space:
                    features = features[:self.obs_space]
                
                return torch.FloatTensor(features).to(self.device)
            else:
                state_tensor = torch.FloatTensor(state).to(self.device)
                if state_tensor.size(-1) != self.obs_space:
                    if state_tensor.size(-1) < self.obs_space:
                        padding = torch.zeros(self.obs_space - state_tensor.size(-1))
                        state_tensor = torch.cat([state_tensor, padding])
                    else:
                        state_tensor = state_tensor[:self.obs_space]
                return state_tensor
            
        except Exception as e:
            print(f"State processing error: {str(e)}")
            print(f"State shape: {np.array(state).shape}")
            print(f"Expected observation space: {self.obs_space}")
            return torch.zeros(self.obs_space).to(self.device)
            
    def normalize(self, data):
        """Normalize data to [0, 1] range"""
        if len(data) == 0:
            return [0.0]
        min_val, max_val = np.min(data), np.max(data)
        if min_val == max_val:
            return [0.0] * len(data)
        return ((data - min_val) / (max_val - min_val)).tolist()

    def select_action(self, state):
        """Select action using current policy"""
        with torch.no_grad():
            probs, value = self.network(state)
            dist = Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
        return action.item(), log_prob.item(), value.item()

    def store_transition(self, state, action, reward, next_state, done, log_prob, value):
        """Store transition in memory"""
        self.memory.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'log_prob': log_prob,
            'value': value
        })

    def update_policy(self):
        """Update policy using PPO"""
        if len(self.memory) < PPO_PARAMS['BATCH_SIZE']:
            return
            
        states = torch.stack([t['state'] for t in self.memory])
        actions = torch.tensor([t['action'] for t in self.memory], device=self.device, dtype=torch.float32)
        rewards = torch.tensor([t['reward'] for t in self.memory], device=self.device, dtype=torch.float32)
        dones = torch.tensor([t['done'] for t in self.memory], device=self.device, dtype=torch.float32)
        old_log_probs = torch.tensor([t['log_prob'] for t in self.memory], device=self.device, dtype=torch.float32)
        old_values = torch.tensor([t['value'] for t in self.memory], device=self.device, dtype=torch.float32)
        
        # Calculate advantages using GAE
        advantages = []
        gae = 0
        for i in reversed(range(len(rewards))):
            next_value = 0 if i == len(rewards) - 1 else old_values[i + 1]
            delta = rewards[i] + PPO_PARAMS['GAMMA'] * next_value * (1 - dones[i]) - old_values[i]
            gae = delta + PPO_PARAMS['GAMMA'] * PPO_PARAMS['GAE_LAMBDA'] * (1 - dones[i]) * gae
            advantages.insert(0, gae)
            
        advantages = torch.tensor(advantages, device=self.device, dtype=torch.float32)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        for _ in range(PPO_PARAMS['PPO_EPOCHS']):
            # Sample mini-batches
            indices = np.random.permutation(len(self.memory))
            
            for start in range(0, len(self.memory), PPO_PARAMS['BATCH_SIZE']):
                end = start + PPO_PARAMS['BATCH_SIZE']
                batch_indices = indices[start:end]
                
                if len(batch_indices) < 3:
                    continue
                    
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                
                probs, values = self.network(batch_states)
                dist = Categorical(probs)
                curr_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                
                # PPO losses
                ratios = torch.exp(curr_log_probs - batch_old_log_probs)
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1-PPO_PARAMS['CLIP_EPSILON'], 1+PPO_PARAMS['CLIP_EPSILON']) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values.squeeze(), rewards[batch_indices])
                
                # Combined loss
                loss = (policy_loss + 
                       PPO_PARAMS['VALUE_LOSS_COEF'] * value_loss - 
                       PPO_PARAMS['ENTROPY_COEF'] * entropy)
                
                # Update network
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), PPO_PARAMS['MAX_GRAD_NORM'])
                self.optimizer.step()
                
                self.training_step += 1
        
        self.memory.clear()

    def training(self, env, trainingParameters=[], verbose=True, rendering=True, plotTraining=True, showPerformance=True):
        """Train the agent according to the standard interface"""
        try:
            # Setup directories and logging
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_id = f"PPO_{env.marketSymbol}_{timestamp}"
            self.figures_dir = os.path.join('Figures', f'run_{self.run_id}')
            os.makedirs(self.figures_dir, exist_ok=True)
            self.writer = SummaryWriter(f'runs/run_{self.run_id}')
            
            num_episodes = trainingParameters[0] if trainingParameters else 1000
            episode_rewards = []
            
            # Add reward scaling
            reward_scaling = RunningMeanStd()
            
            for episode in tqdm(range(num_episodes), disable=not verbose):
                state = env.reset()
                state = self.processState(state, None)
                episode_reward = 0
                episode_value_estimates = []  # Track value estimates
                done = False
                
                while not done:
                    action, log_prob, value = self.select_action(state)
                    next_state, reward, done, _ = env.step(action)
                    next_state = self.processState(next_state, None)
                    
                    episode_value_estimates.append(value)  # Store value estimates
                    
                    # Scale the reward
                    reward_scaling.update(np.array([reward]))
                    scaled_reward = reward / (reward_scaling.std + 1e-8)
                    
                    self.store_transition(state, action, scaled_reward, next_state, done, log_prob, value)
                    
                    if len(self.memory) >= PPO_PARAMS['BATCH_SIZE']:
                        self.update_policy()
                    
                    state = next_state
                    episode_reward += reward  # Use original reward for tracking
                
                # Calculate average value estimate for the episode
                avg_value_estimate = np.mean(episode_value_estimates)
                
                if self.writer is not None:
                    self.writer.add_scalar('Training/episode_reward', episode_reward, episode)
                    self.writer.add_scalar('Training/average_reward', np.mean(self.trailing_rewards), episode)
                    self.writer.add_scalar('Training/value_estimate', avg_value_estimate, episode)  # New metric
                    self.writer.add_scalar('Training/value_estimate_std', np.std(episode_value_estimates), episode)  # Value function stability
                    self.writer.add_scalar('Training/learning_rate', self.optimizer.param_groups[0]['lr'], episode)
                
                if rendering and episode % 10 == 0:
                    self.render_to_dir(env)
            
            if plotTraining:
                self.plot_training_results(episode_rewards)
            
            return env
            
        except Exception as e:
            print(f"Training error: {str(e)}")
            raise
        finally:
            if self.writer is not None:
                self.writer.close()

    def testing(self, training_env, testing_env, rendering=False, showPerformance=False):
        """Test the agent according to the standard interface"""
        self.network.eval()
        state = testing_env.reset()
        done = False
        
        while not done:
            state_tensor = self.processState(state, None)
            with torch.no_grad():
                action_probs, _ = self.network(state_tensor)
                action = torch.argmax(action_probs).item()
            state, _, done, _ = testing_env.step(action)
            
        if rendering:
            self.render_to_dir(testing_env)
            
        self.network.train()
        return testing_env

    def plot_training_results(self, rewards):
        """Plot training results"""
        plt.figure(figsize=(10, 5))
        plt.plot(rewards)
        plt.title('Training Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.savefig(os.path.join(self.figures_dir, 'training_results.png'))
        plt.close()

    def render_to_dir(self, env):
        """Render environment to directory"""
        env.render()
        src_path = f"Figures/{str(env.marketSymbol)}_Rendering.png"
        dst_path = os.path.join(self.figures_dir, f"{str(env.marketSymbol)}_Rendering.png")
        
        if os.path.exists(src_path):
            shutil.move(src_path, dst_path)

    def __del__(self):
        """Cleanup method"""
        if hasattr(self, 'writer') and self.writer is not None:
            self.writer.close()

class RunningMeanStd:
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    @property
    def std(self):
        return np.sqrt(self.var)