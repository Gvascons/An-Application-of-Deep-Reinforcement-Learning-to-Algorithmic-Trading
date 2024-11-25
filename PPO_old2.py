# coding=utf-8

"""
Goal: Implementing a PPO (Proximal Policy Optimization) algorithm specialized
      for algorithmic trading.
Authors: Assistant
Based on work by: Thibaut Théate and Damien Ernst
Institution: University of Liège
"""

###############################################################################
################################### Imports ###################################
###############################################################################

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

from tradingPerformance import PerformanceEstimator
from dataAugmentation import DataAugmentation
from tradingEnv import TradingEnv

# Create Figures directory if it doesn't exist
if not os.path.exists('Figures'):
    os.makedirs('Figures')

###############################################################################
################################ Global variables #############################
###############################################################################

# PPO specific parameters
CLIP_EPSILON = 0.1
VALUE_LOSS_COEF = 1.0
ENTROPY_COEF = 0.005
PPO_EPOCHS = 10
BATCH_SIZE = 128
GAMMA = 0.99
GAE_LAMBDA = 0.95

# Neural network parameters
HIDDEN_SIZE = 256
LEARNING_RATE = 0.0001
L2_FACTOR = 0.000001

# Training parameters
MAX_GRAD_NORM = 0.5

###############################################################################
############################# Neural Network Classes ###########################
###############################################################################

class ActorCritic(nn.Module):
    """
    Combined actor-critic network
    """
    def __init__(self, input_size, num_actions):
        super(ActorCritic, self).__init__()
        
        # Store the input size
        self.input_size = input_size
        
        # Ensure all layers use float32
        self.shared = nn.Sequential(
            nn.Linear(input_size, HIDDEN_SIZE).float(),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE).float(),
            nn.ReLU()
        )
        
        self.actor = nn.Linear(HIDDEN_SIZE, num_actions).float()
        self.critic = nn.Linear(HIDDEN_SIZE, 1).float()
        
        # Initialize weights
        for layer in self.shared:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)
                layer.bias.data.zero_()
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        self.actor.bias.data.zero_()
        nn.init.orthogonal_(self.critic.weight, gain=1)
        self.critic.bias.data.zero_()
        
    def forward(self, x):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        
        # Verify input shape
        assert x.shape[1] == self.input_size, \
            f"Input shape {x.shape[1]} doesn't match expected shape {self.input_size}"
            
        shared_out = self.shared(x)
        action_probs = F.softmax(self.actor(shared_out), dim=-1)
        value = self.critic(shared_out)
        return action_probs, value

###############################################################################
################################ Class PPO ###################################
###############################################################################

class PPO:
    """
    PPO agent implementation
    """
    def __init__(self, observationSpace, actionSpace, marketSymbol=""):
        """
        Initialize PPO agent
        """
        # Set random seed for reproducibility
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        
        # Set device (GPU if available, else CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Calculate the correct observation space size
        # 30 values each for: closes, RSI, momentum, volatility
        # Plus 1 for position
        self.actual_obs_space = 30 * 4 + 1  # Should be 121
        
        # Initialize the actor-critic model
        self.model = ActorCritic(self.actual_obs_space, actionSpace).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE, weight_decay=L2_FACTOR)
        
        # Store additional information
        self.marketSymbol = marketSymbol
        self.actionSpace = actionSpace
        self.transitions = []
        
        # Initialize tracking variables
        self.writer = None
        self.run_id = None
        self.figures_dir = None
        self.results_dir = None
        
        # Set model to eval mode initially
        self.model.eval()

        # Memory management parameters
        self.max_size = 10000  # Maximum number of transitions to store
        self.batch_size = BATCH_SIZE

    def calculate_rsi(self, prices, period=14):
        """
        Calculate Relative Strength Index with proper error handling
        """
        deltas = np.diff(prices)
        seed = deltas[:period+1]
        up = seed[seed >= 0].sum()/period
        down = -seed[seed < 0].sum()/period
        
        # Handle divide by zero
        if down == 0:
            rs = 100.0 if up > 0 else 0.0
        else:
            rs = up/down
        
        rsi = np.zeros_like(prices)
        rsi[:period] = 100. - 100./(1. + rs)

        for i in range(period, len(prices)):
            delta = deltas[i - 1]
            if delta > 0:
                upval = delta
                downval = 0.
            else:
                upval = 0.
                downval = -delta

            up = (up*(period-1) + upval)/period
            down = (down*(period-1) + downval)/period
            
            # Handle divide by zero
            if down == 0:
                rs = 100.0 if up > 0 else 0.0
            else:
                rs = up/down
            
            rsi[i] = 100. - 100./(1. + rs)

        return rsi

    def calculate_momentum(self, prices, period=10):
        """
        Calculate price momentum
        """
        momentum = np.zeros_like(prices)
        for i in range(period, len(prices)):
            momentum[i] = (prices[i] - prices[i-period]) / prices[i-period]
        momentum[:period] = momentum[period]
        return momentum

    def calculate_volatility(self, prices, period=10):
        """
        Calculate price volatility
        """
        volatility = np.zeros_like(prices)
        for i in range(period, len(prices)):
            volatility[i] = np.std(prices[i-period:i]) / np.mean(prices[i-period:i])
        volatility[:period] = volatility[period]
        return volatility

    def processState(self, state, coefficients):
        """
        Enhanced state processing with fixed size output
        """
        try:
            # If already a tensor, ensure it's float32 and correct size
            if torch.is_tensor(state):
                if state.shape[0] == self.actual_obs_space:
                    return state.float()
                else:
                    raise ValueError(f"Tensor state has incorrect shape: {state.shape[0]}")
            
            # Extract price data
            if isinstance(state[0], list):
                closes = np.array(state[0], dtype=np.float32)
                lows = np.array(state[1], dtype=np.float32)
                highs = np.array(state[2], dtype=np.float32)
                volumes = np.array(state[3], dtype=np.float32)
            else:
                # Handle case where state is already flattened
                return torch.FloatTensor(state[:self.actual_obs_space]).to(self.device)
            
            # Ensure all arrays are length 30
            def pad_or_truncate(arr, target_len=30):
                if len(arr) > target_len:
                    return arr[-target_len:]
                elif len(arr) < target_len:
                    return np.pad(arr, (target_len - len(arr), 0), 'edge')
                return arr
            
            # Process and pad/truncate all features
            closes = pad_or_truncate(closes)
            rsi = pad_or_truncate(self.calculate_rsi(closes))
            momentum = pad_or_truncate(self.calculate_momentum(closes))
            volatility = pad_or_truncate(self.calculate_volatility(closes))
            
            # Combine all features
            processed_state = []
            
            # Add normalized price data
            if coefficients[0][0] != coefficients[0][1]:
                normalized_closes = (closes - coefficients[0][0]) / (coefficients[0][1] - coefficients[0][0])
                processed_state.extend(normalized_closes.tolist())
            else:
                processed_state.extend([0.0] * 30)
            
            # Add technical indicators
            processed_state.extend(rsi.tolist())
            processed_state.extend(momentum.tolist())
            processed_state.extend(volatility.tolist())
            
            # Add position information (last element of state)
            if isinstance(state[-1], list):
                processed_state.append(float(state[-1][0]))
            else:
                processed_state.append(float(state[-1]))
            
            # Verify final length
            assert len(processed_state) == self.actual_obs_space, \
                f"Processed state length {len(processed_state)} doesn't match expected {self.actual_obs_space}"
            
            return torch.FloatTensor(processed_state).to(self.device)
            
        except Exception as e:
            print(f"State processing error: {str(e)}")
            print(f"State type: {type(state)}")
            print(f"State content: {state}")
            return torch.zeros(self.actual_obs_space).to(self.device)

    def processReward(self, reward):
        """
        Enhanced reward processing with trading-specific considerations
        """
        # Add position-based shaping
        if hasattr(self, 'previous_position'):
            # Penalize frequent position changes
            if self.current_position != self.previous_position:
                reward -= 0.1  # Transaction cost penalty
            
            # Reward trend following
            if (self.current_position == 1 and self.price_change > 0) or \
               (self.current_position == -1 and self.price_change < 0):
                reward *= 1.2  # Boost reward for correct trend following
        
        # Scale rewards to manageable range
        reward = np.clip(reward * 10, -1, 1)  # Amplify and clip rewards
        
        return reward

    def getNormalizationCoefficients(self, env):
        """
        Get normalization coefficients for the environment
        """
        # Get the data from the environment
        data = env.data
        
        # Calculate coefficients for returns
        returns = data['Close'].pct_change().dropna()
        return_min, return_max = returns.min(), returns.max()
        
        # Calculate coefficients for delta prices
        delta_prices = abs(data['High'] - data['Low'])
        delta_min, delta_max = delta_prices.min(), delta_prices.max()
        
        # Calculate coefficients for close price position
        close_pos = (data['Close'] - data['Low']) / (data['High'] - data['Low']).replace(0, float('inf'))
        pos_min, pos_max = close_pos.min(), close_pos.max()
        
        # Calculate coefficients for volumes
        volume_min, volume_max = data['Volume'].min(), data['Volume'].max()
        
        return [
            [return_min, return_max],
            [delta_min, delta_max],
            [pos_min, pos_max],
            [volume_min, volume_max]
        ]

    def select_action(self, state):
        """
        Modified action selection with trading-specific considerations
        """
        with torch.no_grad():
            action_probs, value = self.model(state)
            
            # Add position-based masking
            if hasattr(self, 'previous_position'):
                # Discourage frequent trading by adjusting probabilities
                if self.previous_position != 0:  # If already in a position
                    action_probs[0][1 - self.previous_position] *= 0.8  # Reduce probability of switching
            
            # Add momentum-based bias
            if hasattr(self, 'price_momentum'):
                if self.price_momentum > 0:
                    action_probs[0][1] *= 1.2  # Increase probability of going long
                elif self.price_momentum < 0:
                    action_probs[0][0] *= 1.2  # Increase probability of going short
            
            dist = Categorical(action_probs)
            action = dist.sample()
            
            return action.item(), dist.log_prob(action), value

    def training(self, trainingEnv, trainingParameters=[], verbose=True, rendering=True, plotTraining=True, showPerformance=True):
        """
        Train the PPO agent
        """
        try:
            # Create run-specific directories and ID at the start
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_id = f"PPO_{trainingEnv.marketSymbol}_{timestamp}"
            
            # Create base directories
            self.figures_dir = os.path.join('Figures', f'run_{self.run_id}')
            os.makedirs(self.figures_dir, exist_ok=True)
            os.makedirs('Results', exist_ok=True)
            
            # Create and share results directory
            self.results_dir = os.path.join('Results', f'run_{self.run_id}')
            os.makedirs(self.results_dir, exist_ok=True)
            
            # Share directories with environment
            trainingEnv.figures_dir = self.figures_dir
            trainingEnv.results_dir = self.results_dir
            
            # Initialize tensorboard writer
            self.writer = SummaryWriter(f'runs/run_{self.run_id}')
            
            # If required, print the training progression
            if verbose:
                print("Training progression (hardware selected => " + str(self.device) + "):")

            # Apply data augmentation
            dataAugmentation = DataAugmentation()
            trainingEnvList = dataAugmentation.generate(trainingEnv)

            # Initialize performance tracking arrays properly
            if plotTraining:
                performanceTrain = []  # List to store training Sharpe ratios
                performanceTest = []   # List to store testing Sharpe ratios
                num_episodes = trainingParameters[0] if trainingParameters else 1
                num_envs = len(trainingEnvList)
                score = np.zeros((num_envs, num_episodes))

                # Create proper testing environment
                marketSymbol = trainingEnv.marketSymbol
                startingDate = trainingEnv.endingDate
                endingDate = '2020-1-1'
                money = trainingEnv.data['Money'][0]
                stateLength = trainingEnv.stateLength
                transactionCosts = trainingEnv.transactionCosts
                testingEnv = TradingEnv(marketSymbol, startingDate, endingDate, money, stateLength, transactionCosts)
                testingEnv.figures_dir = self.figures_dir

            best_performance = float('-inf')
            patience = 10
            patience_counter = 0
            
            # Training loop
            for episode in tqdm(range(num_episodes), disable=not verbose):
                for i in range(len(trainingEnvList)):
                    # Track total reward per episode
                    totalReward = 0
                    
                    # Reset environment with random starting point
                    coefficients = self.getNormalizationCoefficients(trainingEnvList[i])
                    trainingEnvList[i].reset()
                    startingPoint = random.randrange(len(trainingEnvList[i].data.index))
                    trainingEnvList[i].setStartingPoint(startingPoint)
                    state = self.processState(trainingEnvList[i].state, coefficients)
                    done = False
                    
                    # Episode loop
                    while not done:
                        # Process state and select action
                        state_tensor = self.processState(state, self.getNormalizationCoefficients(trainingEnvList[i]))
                        action, log_prob, value = self.select_action(state_tensor)
                        
                        # Take action in environment
                        next_state, reward, done, info = trainingEnvList[i].step(action)
                        totalReward += reward
                        
                        # Store transition for PPO update
                        self.store_transition(state_tensor, action, reward, next_state, done)
                        
                        # Update policy if enough transitions are collected
                        if len(self.transitions) >= BATCH_SIZE:
                            self.update_policy()
                            self.transitions.clear()
                        
                        state = next_state

                    # Store episode reward
                    score[i, episode] = totalReward

                # Compute and store performance metrics after each episode
                if plotTraining:
                    # Training performance
                    trainingEnv = self.testing(trainingEnv, trainingEnv)
                    analyser = PerformanceEstimator(trainingEnv.data)
                    train_performance = analyser.computeSharpeRatio()
                    performanceTrain.append(train_performance)
                    self.writer.add_scalar('Training performance (Sharpe Ratio)', train_performance, episode)
                    trainingEnv.reset()

                    # Testing performance
                    testingEnv = self.testing(trainingEnv, testingEnv)
                    analyser = PerformanceEstimator(testingEnv.data)
                    test_performance = analyser.computeSharpeRatio()
                    performanceTest.append(test_performance)
                    self.writer.add_scalar('Testing performance (Sharpe Ratio)', test_performance, episode)
                    testingEnv.reset()

                # Early stopping check
                if test_performance > best_performance:
                    best_performance = test_performance
                    patience_counter = 0
                    self.saveModel(os.path.join(self.results_dir, 'best_model.pth'))
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"\nEarly stopping triggered at episode {episode}")
                        break

            # Assess the algorithm performance on the training trading environment
            trainingEnv = self.testing(trainingEnv, trainingEnv)

            # Only render once after all training is complete
            if rendering:
                self.render_to_dir(trainingEnv)

            # If required, plot the training results
            if plotTraining:
                # Plot Sharpe ratio progression
                fig = plt.figure()
                ax = fig.add_subplot(111, ylabel='Performance (Sharpe Ratio)', xlabel='Episode')
                ax.plot(performanceTrain, label="Training")
                ax.plot(performanceTest, label="Testing")
                ax.legend()
                plt.savefig(os.path.join(self.figures_dir, 'TrainingTestingPerformance.png'))
                plt.close(fig)

                # Plot rewards
                for i in range(len(trainingEnvList)):
                    fig = plt.figure()
                    ax1 = fig.add_subplot(111, ylabel='Total reward collected', xlabel='Episode')
                    ax1.plot(score[i, :])
                    plt.savefig(os.path.join(self.figures_dir, f'TrainingResults.png'))
                    plt.close(fig)

            # If required, print and save the strategy performance
            if showPerformance:
                analyser = PerformanceEstimator(trainingEnv.data)
                analyser.run_id = self.run_id
                analyser.displayPerformance('PPO', phase='training')

            return trainingEnv

        except Exception as e:
            print(f"\nTraining interrupted: {str(e)}")
            raise
        finally:
            if hasattr(self, 'writer'):
                self.writer.flush()

    def testing(self, trainingEnv, testingEnv, rendering=False, showPerformance=False):
        """
        Test the trained PPO agent
        """
        self.model.eval()
        state = testingEnv.reset()
        done = False
        
        while not done:
            state_tensor = self.processState(state, self.getNormalizationCoefficients(testingEnv))
            action, _, _ = self.select_action(state_tensor)
            state, _, done, _ = testingEnv.step(action)
            
        if rendering:
            self.render_to_dir(testingEnv)
            
        if showPerformance:
            analyser = PerformanceEstimator(testingEnv.data)
            analyser.run_id = self.run_id
            analyser.displayPerformance('PPO', phase='testing')
            
        return testingEnv

    def saveModel(self, fileName):
        """
        Save the PPO model
        """
        torch.save(self.model.state_dict(), fileName)

    def loadModel(self, fileName):
        """
        Load a saved PPO model
        """
        self.model.load_state_dict(torch.load(fileName, map_location=self.device))

    def render_to_dir(self, env):
        """
        Render the environment and save to the run directory
        """
        env.render()
        src_path = ''.join(['Figures/', str(env.marketSymbol), '_Rendering', '.png'])
        dst_path = os.path.join(self.figures_dir, ''.join([str(env.marketSymbol), '_Rendering', '.png']))
        
        if os.path.exists(src_path):
            shutil.move(src_path, dst_path)

    def __del__(self):
        """
        Cleanup
        """
        if hasattr(self, 'writer') and self.writer is not None:
            self.writer.close()

    def store_transition(self, state, action, reward, next_state, done):
        """
        Enhanced transition storage with proper state processing
        """
        try:
            # Ensure states are processed tensors
            if not torch.is_tensor(state):
                state = self.processState(state, self.get_normalization_coefficients(state))
            
            if not torch.is_tensor(next_state):
                next_state = self.processState(next_state, self.get_normalization_coefficients(next_state))
            
            # Calculate transition priority based on reward magnitude
            priority = abs(reward) + 0.01
            
            # Store transition with priority
            self.transitions.append({
                'state': state.detach(),  # Detach tensors from computation graph
                'action': action,
                'reward': reward,
                'next_state': next_state.detach(),
                'done': done,
                'priority': priority
            })
            
            # Keep most important transitions when buffer is full
            if len(self.transitions) > self.max_size:
                self.transitions.sort(key=lambda x: x['priority'], reverse=True)
                self.transitions = self.transitions[:self.max_size]
            
        except Exception as e:
            print(f"Error in store_transition: {str(e)}")
            print(f"State type: {type(state)}")
            print(f"Next state type: {type(next_state)}")
            if isinstance(state, list):
                print(f"State length: {len(state)}")
            if isinstance(next_state, list):
                print(f"Next state length: {len(next_state)}")
            raise

    def get_normalization_coefficients(self, state):
        """
        Calculate normalization coefficients for state processing
        """
        if isinstance(state[0], list):
            closes = np.array(state[0])
            return [(closes.min(), closes.max())]
        return [(0, 1)]  # Default normalization if state format is unexpected

    def update_policy(self):
        """
        Modified policy update with proper tensor handling
        """
        if len(self.transitions) < self.batch_size:
            return
        
        for _ in range(PPO_EPOCHS):
            # Sample batch using prioritized sampling
            batch = self.sample_batch()
            
            try:
                # Convert batch data to tensors, ensuring they're all on the same device
                states = torch.stack([t['state'] for t in batch]).to(self.device)
                actions = torch.tensor([t['action'] for t in batch], dtype=torch.long).to(self.device)
                rewards = torch.tensor([t['reward'] for t in batch], dtype=torch.float32).to(self.device)
                next_states = torch.stack([t['next_state'] for t in batch]).to(self.device)
                dones = torch.tensor([t['done'] for t in batch], dtype=torch.float32).to(self.device)
                
                # Get current policy outputs
                action_probs, values = self.model(states)
                dist = Categorical(action_probs)
                
                # Calculate advantages
                next_values = self.model(next_states)[1].detach()
                advantages = rewards + GAMMA * next_values * (1 - dones) - values.detach()
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
                # Calculate policy loss
                log_probs = dist.log_prob(actions)
                ratio = torch.exp(log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1-CLIP_EPSILON, 1+CLIP_EPSILON) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Calculate value loss
                value_loss = F.mse_loss(values.squeeze(), rewards + GAMMA * next_values.squeeze() * (1 - dones))
                
                # Calculate entropy bonus
                entropy = dist.entropy().mean()
                
                # Combined loss
                loss = policy_loss + VALUE_LOSS_COEF * value_loss - ENTROPY_COEF * entropy
                
                # Update networks
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), MAX_GRAD_NORM)
                self.optimizer.step()
                
            except Exception as e:
                print(f"Error in update_policy: {str(e)}")
                print(f"Batch size: {len(batch)}")
                print(f"First state shape: {batch[0]['state'].shape}")
                print(f"First next_state shape: {batch[0]['next_state'].shape}")
                raise

    def compute_returns(self, rewards, dones):
        """
        Enhanced return calculation with risk-adjusted rewards
        """
        returns = torch.zeros_like(rewards)
        running_return = 0
        
        # Calculate Sharpe ratio like adjustment
        reward_std = rewards.std() + 1e-8
        risk_adjusted_rewards = rewards / reward_std
        
        for t in reversed(range(len(rewards))):
            running_return = risk_adjusted_rewards[t] + GAMMA * running_return * (1 - dones[t])
            returns[t] = running_return
        
        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        return returns

    def plotExpectedPerformance(self, trainingEnv, trainingParameters=[], iterations=10):
        """
        Plot both individual and expected performance with confidence intervals
        """
        # Initialize performance arrays
        performanceTrain = np.zeros((trainingParameters[0], iterations))
        performanceTest = np.zeros((trainingParameters[0], iterations))
        
        try:
            for iteration in range(iterations):
                print(f'Expected performance evaluation progression: {iteration+1}/{iterations}')
                
                # Train and track performance
                for episode in range(trainingParameters[0]):
                    # ... training code ...
                    
                    # Store performance for this iteration
                    performanceTrain[episode][iteration] = train_performance
                    performanceTest[episode][iteration] = test_performance
                
                # Plot individual iteration performance
                fig = plt.figure()
                ax = fig.add_subplot(111, ylabel='Performance (Sharpe Ratio)', xlabel='Episode')
                ax.plot([performanceTrain[e][iteration] for e in range(trainingParameters[0])])
                ax.plot([performanceTest[e][iteration] for e in range(trainingParameters[0])])
                ax.legend(["Training", "Testing"])
                plt.savefig(os.path.join(self.figures_dir, 
                           f'TrainingTestingPerformance_{iteration+1}.png'))
                plt.close(fig)
            
            # Compute statistics for expected performance
            expectedPerformanceTrain = np.mean(performanceTrain, axis=1)
            expectedPerformanceTest = np.mean(performanceTest, axis=1)
            stdPerformanceTrain = np.std(performanceTrain, axis=1)
            stdPerformanceTest = np.std(performanceTest, axis=1)
            
            # Plot expected performance with confidence intervals
            fig = plt.figure()
            ax = fig.add_subplot(111, ylabel='Performance (Sharpe Ratio)', xlabel='Episode')
            ax.plot(expectedPerformanceTrain)
            ax.plot(expectedPerformanceTest)
            ax.fill_between(range(len(expectedPerformanceTrain)), 
                           expectedPerformanceTrain-stdPerformanceTrain, 
                           expectedPerformanceTrain+stdPerformanceTrain, alpha=0.25)
            ax.fill_between(range(len(expectedPerformanceTest)),
                           expectedPerformanceTest-stdPerformanceTest,
                           expectedPerformanceTest+stdPerformanceTest, alpha=0.25)
            ax.legend(["Training", "Testing"])
            plt.savefig(os.path.join(self.figures_dir, 
                       f'TrainingTestingExpectedPerformance.png'))
            plt.close(fig)
            
        except KeyboardInterrupt:
            print("\nWARNING: Expected performance evaluation prematurely interrupted...")

    def sample_batch(self):
        """
        Sample a batch of transitions using prioritized sampling
        """
        if len(self.transitions) < self.batch_size:
            return self.transitions  # Return all transitions if less than batch size
        
        # Calculate sampling probabilities based on priorities
        priorities = np.array([t['priority'] for t in self.transitions])
        probs = priorities / priorities.sum()
        
        # Sample indices based on priorities
        indices = np.random.choice(
            len(self.transitions), 
            size=self.batch_size, 
            p=probs, 
            replace=False
        )
        
        return [self.transitions[i] for i in indices]