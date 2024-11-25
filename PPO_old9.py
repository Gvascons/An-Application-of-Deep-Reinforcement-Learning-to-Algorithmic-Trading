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
from tradingPerformance import PerformanceEstimator
from pathlib import Path

# Create Figures directory if it doesn't exist (like TDQN does)
if not os.path.exists('Figs'):
    os.makedirs('Figs')

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
        # Convert input to tensor if it's not already
        if isinstance(x, list):
            x = torch.FloatTensor(x).to(self.actor[0].weight.device)
        elif isinstance(x, np.ndarray):
            x = torch.FloatTensor(x).to(self.actor[0].weight.device)
        
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
            
        features = self.shared(x)
        action_probs = F.softmax(self.actor(features), dim=-1)
        value = self.critic(features)
        
        return action_probs, value

class PPO:
    """Implementation of PPO algorithm for trading"""
    def __init__(self, state_dim, action_dim, device='cpu', marketSymbol=None):
        """Initialize PPO agent"""
        self.device = device
        
        # Calculate input size based on state structure
        # state = [prices(30) + lows(30) + highs(30) + volumes(30) + position(1)] = 121 total features
        self.input_size = 121  # Hardcode to match environment state size
        self.num_actions = action_dim  # Should be 2 (long/short)
        
        print(f"Initializing PPO with input size: {self.input_size}, action size: {self.num_actions}")
        
        # Initialize network with correct input size
        self.network = PPONetwork(self.input_size, self.num_actions).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=PPO_PARAMS['LEARNING_RATE'])
        
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

    def processState(self, state, info=None):
        """Process the state from the environment to match network input"""
        # State from trading env is a list of lists [prices, lows, highs, volumes, position]
        if isinstance(state, list):
            # Flatten the lists and concatenate
            flattened = []
            for sublist in state:
                if isinstance(sublist, list):
                    flattened.extend(sublist)  # For price histories (30 values each)
                else:
                    flattened.append(float(sublist))  # For position (single value)
            
            # Verify dimensions
            if len(flattened) != self.input_size:
                print(f"Warning: State dimension mismatch. Expected {self.input_size}, got {len(flattened)}")
                print(f"State structure: {[len(x) if isinstance(x, list) else 1 for x in state]}")
            
            return flattened
        
        # If state is already flat, ensure all elements are float
        if isinstance(state, (list, np.ndarray)):
            return [float(x) for x in state]
        
        return state

    def normalize(self, data):
        """Normalize data to [0, 1] range"""
        if len(data) == 0:
            return [0.0]
        min_val, max_val = np.min(data), np.max(data)
        if min_val == max_val:
            return [0.0] * len(data)
        return ((data - min_val) / (max_val - min_val)).tolist()

    def select_action(self, state):
        """Select an action from the current policy"""
        # Process state if needed
        state = self.processState(state)
        
        # Convert to tensor
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            probs, value = self.network(state)
            dist = Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        
        return action.item(), log_prob.item(), value.item()

    def store_transition(self, state, action, reward, next_state, done, log_prob, value):
        """Store a transition in memory"""
        # Process state if it's a list
        if isinstance(state, list):
            processed_state = []
            for sublist in state:
                if isinstance(sublist, list):
                    processed_state.extend(sublist)
                else:
                    processed_state.append(sublist)
        else:
            processed_state = state

        # Process next_state if it's a list
        if isinstance(next_state, list):
            processed_next_state = []
            for sublist in next_state:
                if isinstance(sublist, list):
                    processed_next_state.extend(sublist)
                else:
                    processed_next_state.append(sublist)
        else:
            processed_next_state = next_state

        # Store the processed transition
        self.memory.append({
            'state': processed_state,
            'action': action,
            'reward': float(reward),
            'next_state': processed_next_state,
            'done': float(done),
            'log_prob': float(log_prob),
            'value': float(value)
        })

    def update_policy(self):
        """Update policy using PPO"""
        if len(self.memory) < PPO_PARAMS['BATCH_SIZE']:
            return
        
        # Convert stored transitions to tensors
        states = torch.FloatTensor([t['state'] for t in self.memory]).to(self.device)
        actions = torch.LongTensor([t['action'] for t in self.memory]).to(self.device)
        rewards = torch.FloatTensor([t['reward'] for t in self.memory]).to(self.device)
        next_states = torch.FloatTensor([t['next_state'] for t in self.memory]).to(self.device)
        dones = torch.FloatTensor([t['done'] for t in self.memory]).to(self.device)
        old_log_probs = torch.FloatTensor([t['log_prob'] for t in self.memory]).to(self.device)
        old_values = torch.FloatTensor([t['value'] for t in self.memory]).to(self.device)
        
        # Compute advantages
        advantages = []
        gae = 0
        with torch.no_grad():
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
                
                # Get current policy outputs
                probs, values = self.network(batch_states)
                dist = Categorical(probs)
                curr_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                
                # Calculate losses separately
                ratios = torch.exp(curr_log_probs - batch_old_log_probs)
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1-PPO_PARAMS['CLIP_EPSILON'], 1+PPO_PARAMS['CLIP_EPSILON']) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values.squeeze(), rewards[batch_indices])
                
                # Combine losses
                loss = (policy_loss + 
                       PPO_PARAMS['VALUE_LOSS_COEF'] * value_loss - 
                       PPO_PARAMS['ENTROPY_COEF'] * entropy)
                
                # Update network
                self.optimizer.zero_grad()
                loss.backward()  # No need for retain_graph here since we're not doing multiple backwards
                nn.utils.clip_grad_norm_(self.network.parameters(), PPO_PARAMS['MAX_GRAD_NORM'])
                self.optimizer.step()
                
                self.training_step += 1
        
        self.memory.clear()

    def training(self, env, trainingParameters=[], verbose=True, rendering=True, plotTraining=True, showPerformance=True):
        try:
            # Create timestamp and run ID exactly like TDQN
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_id = f"PPO_{env.marketSymbol}_{timestamp}"
            #self.run_id = f"PPO_{timestamp}"

            # Create run-specific directory under Figures, exactly like TDQN
            #self.figures_dir = os.path.join('Figures', f'run_{self.run_id}')
            self.figures_dir = os.path.join('Figs', f'run_{self.run_id}')

            os.makedirs(self.figures_dir, exist_ok=True)
            
            # Initialize tensorboard writer
            self.writer = SummaryWriter(f'runs/run_{self.run_id}')
            
            # Pass directories to environment
            env.figures_dir = self.figures_dir
            
            num_episodes = trainingParameters[0] if trainingParameters else 1000
            iterations = trainingParameters[1] if len(trainingParameters) > 1 else 1
            
            # Initialize tracking arrays
            performanceTrain = []
            performanceTest = []
            episode_rewards = []
            
            # Store initial weights
            initialWeights = copy.deepcopy(self.network.state_dict())
            
            try:
                for episode in tqdm(range(num_episodes)):
                    state = env.reset()
                    episode_reward = 0
                    done = False
                    steps = 0
                    
                    while not done:
                        action, log_prob, value = self.select_action(state)
                        next_state, reward, done, _ = env.step(action)
                        
                        self.store_transition(state, action, reward, next_state, done, log_prob, value)
                        
                        if len(self.memory) >= PPO_PARAMS['BATCH_SIZE']:
                            self.update_policy()
                        
                        state = next_state
                        episode_reward += reward
                        steps += 1
                    
                    # Track episode metrics
                    episode_rewards.append(episode_reward)
                    
                    # Compute training performance (Sharpe Ratio)
                    train_analyser = PerformanceEstimator(env.data)
                    train_sharpe = train_analyser.computeSharpeRatio()
                    performanceTrain.append(train_sharpe)
                    self.writer.add_scalar('Training performance (Sharpe Ratio)', train_sharpe, episode)
                    
                    # Create and evaluate test environment
                    test_env = copy.deepcopy(env)
                    test_env = self.testing(env, test_env, rendering=False, showPerformance=False)
                    test_analyser = PerformanceEstimator(test_env.data)
                    test_sharpe = test_analyser.computeSharpeRatio()
                    performanceTest.append(test_sharpe)
                    self.writer.add_scalar('Testing performance (Sharpe Ratio)', test_sharpe, episode)
                    
                    if verbose:
                        print(f"\nEpisode {episode}")
                        print(f"Training Sharpe: {train_sharpe:.3f}")
                        print(f"Testing Sharpe: {test_sharpe:.3f}")
                    
                    # Plot training results periodically
                    if plotTraining and (episode + 1) % 10 == 0:  # Every 10 episodes
                        self.plotTraining(episode_rewards)
                    
                    if rendering:
                        self.render_to_dir(env)
                
                # Ensure directory exists before saving final plots
                os.makedirs(self.figures_dir, exist_ok=True)
                
                if plotTraining:
                    self.plot_performance_results(performanceTrain, performanceTest,
                                               np.std(performanceTrain), np.std(performanceTest))
                
                return env
                
            except KeyboardInterrupt:
                print("\nTraining interrupted by user")
                return env
                
        except Exception as e:
            print(f"Training error: {str(e)}")
            raise
        finally:
            if self.writer is not None:
                self.writer.close()

    def testing(self, trainingEnv, testingEnv, rendering=True, showPerformance=True):
        """Test the trained policy on new data"""
        try:
            self.network.eval()  # Set to evaluation mode
            state = testingEnv.reset()
            done = False
            episode_reward = 0
            actions_taken = []
            
            with torch.no_grad():
                while not done:
                    state = self.processState(state, None)
                    action_probs, _ = self.network(state)
                    
                    # Add some exploration during testing
                    if random.random() < 0.1:  # 10% exploration
                        action = random.choice([0, 1])  # Binary action space
                    else:
                        action = torch.argmax(action_probs).item()
                    
                    next_state, reward, done, info = testingEnv.step(action)
                    episode_reward += reward
                    actions_taken.append(action)
                    state = next_state
            
            # Log action distribution
            action_counts = {
                "Short (0)": actions_taken.count(0),
                "Long (1)": actions_taken.count(1)
            }
            print("\nAction Distribution during testing:")
            total_actions = len(actions_taken)
            for action_name, count in action_counts.items():
                print(f"{action_name}: {count} times ({count/total_actions*100:.1f}%)")
            
            # Show performance if requested
            if showPerformance:
                analyser = PerformanceEstimator(testingEnv.data)
                performance = analyser.computePerformance()
                print("\nTesting Performance:")
                for metric, value in performance:
                    print(f"{metric}: {value}")
            
            return testingEnv
            
        except Exception as e:
            print(f"Error in testing: {str(e)}")
            raise

    def plotTraining(self, rewards):
        """Plot the training phase results (rewards)"""
        try:
            # Ensure directory exists
            Path(self.figures_dir).mkdir(parents=True, exist_ok=True)
            
            fig = plt.figure()
            ax1 = fig.add_subplot(111, ylabel='Total reward collected', xlabel='Episode')
            ax1.plot(rewards)
            
            # Save figure using pathlib
            save_path = Path(self.figures_dir) / 'TrainingResults.png'
            plt.savefig(str(save_path))
            plt.close(fig)
        except Exception as e:
            print(f"Error in plotTraining: {str(e)}")

    def render_to_dir(self, env):
        """Render environment to directory exactly like TDQN"""
        env.render()
        src_path = ''.join(['Figs/', str(env.marketSymbol), '_Rendering', '.png'])
        dst_path = os.path.join(self.figures_dir, ''.join([str(env.marketSymbol), '_Rendering', '.png']))
        """Render environment to run-specific directory"""
        try:
            env.render()
            base_dir = os.path.dirname(os.path.abspath(__file__))
            src_path = os.path.join(base_dir, 'Figs', f"{str(env.marketSymbol)}_Rendering.png")
            dst_path = os.path.join(self.figures_dir, f"{str(env.marketSymbol)}_Rendering.png")
            
            if os.path.exists(src_path):
                shutil.move(src_path, dst_path)
        except Exception as e:
            print(f"Error in render_to_dir: {str(e)}")

    def plot_performance_results(self, train_perf, test_perf, train_std, test_std):
        """Plot training and testing performance with standard deviation"""
        try:
            # Create figure exactly like TDQN does
            fig = plt.figure()
            ax = fig.add_subplot(111, ylabel='Performance (Sharpe Ratio)', xlabel='Episode')
            episodes = range(len(train_perf))
            
            ax.plot(episodes, train_perf, label='Training', color='blue')
            ax.plot(episodes, test_perf, label='Testing', color='green')
            
            ax.fill_between(episodes,
                           np.array(train_perf) - train_std,
                           np.array(train_perf) + train_std,
                           alpha=0.2, color='blue')
            ax.fill_between(episodes,
                           np.array(test_perf) - test_std,
                           np.array(test_perf) + test_std,
                           alpha=0.2, color='green')
            
            plt.title('Training and Testing Performance')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Save figure exactly like TDQN does
            plt.savefig(os.path.join(self.figures_dir, 'TrainingTestingExpectedPerformance.png'))
            plt.close(fig)
            
        except Exception as e:
            print(f"Error in plot_performance_results: {str(e)}")
            raise

    def __del__(self):
        """Cleanup method"""
        if hasattr(self, 'writer') and self.writer is not None:
            self.writer.close()

    def log_performance_metrics(self, episode, train_sharpe, test_sharpe):
        """Log performance metrics to TensorBoard"""
        if self.writer is not None:
            self.writer.add_scalar('Performance/Train_Sharpe', train_sharpe, episode)
            self.writer.add_scalar('Performance/Test_Sharpe', test_sharpe, episode)
            
            # Log the difference between train and test Sharpe ratios to monitor overfitting
            self.writer.add_scalar('Performance/Train_Test_Gap', train_sharpe - test_sharpe, episode)

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