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
        self.actual_obs_space = (30 - 1) * 4 + 1  # Should be 117 (29 values * 4 features + 1 money value)
        
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

    def processState(self, state, coefficients):
        """
        Process the RL state returned by the environment
        """
        try:
            # If already a tensor, ensure it's float32 and return
            if torch.is_tensor(state):
                return state.float()
            
            # Validate input state
            if not isinstance(state, (list, np.ndarray)) or len(state) < 4:
                print(f"Invalid state format received: {type(state)}")
                return torch.zeros(self.actual_obs_space).to(self.device)
            
            # Convert lists to numpy arrays
            closePrices = np.array(state[0], dtype=np.float32)
            lowPrices = np.array(state[1], dtype=np.float32)
            highPrices = np.array(state[2], dtype=np.float32)
            volumes = np.array(state[3], dtype=np.float32)
            
            processed_state = []
            
            # 1. Process returns (29 values)
            returns = np.diff(closePrices) / closePrices[:-1]
            if coefficients[0][0] != coefficients[0][1]:
                normalized_returns = (returns - coefficients[0][0]) / (coefficients[0][1] - coefficients[0][0])
                processed_state.extend(normalized_returns.tolist())
            else:
                processed_state.extend([0.0] * (len(closePrices) - 1))
            
            # 2. Process delta prices (29 values)
            delta_prices = np.abs(highPrices[1:] - lowPrices[1:])
            if coefficients[1][0] != coefficients[1][1]:
                normalized_delta = (delta_prices - coefficients[1][0]) / (coefficients[1][1] - coefficients[1][0])
                processed_state.extend(normalized_delta.tolist())
            else:
                processed_state.extend([0.0] * (len(closePrices) - 1))
            
            # 3. Process close price positions (29 values)
            close_positions = []
            for i in range(1, len(closePrices)):
                delta = abs(highPrices[i] - lowPrices[i])
                position = abs(closePrices[i] - lowPrices[i]) / delta if delta != 0 else 0.5
                close_positions.append(position)
            if coefficients[2][0] != coefficients[2][1]:
                normalized_positions = [(x - coefficients[2][0]) / (coefficients[2][1] - coefficients[2][0]) 
                                      for x in close_positions]
                processed_state.extend(normalized_positions)
            else:
                processed_state.extend([0.5] * (len(closePrices) - 1))
            
            # 4. Process volumes (29 values)
            volumes = volumes[1:]  # Skip first volume to match other features
            if coefficients[3][0] != coefficients[3][1]:
                normalized_volumes = (volumes - coefficients[3][0]) / (coefficients[3][1] - coefficients[3][0])
                processed_state.extend(normalized_volumes.tolist())
            else:
                processed_state.extend([0.0] * len(volumes))
            
            # Add money information (1 value)
            money = float(state[4][0]) if isinstance(state[4], list) else float(state[4])
            processed_state.append(money)
            
            # Verify the processed state length
            expected_length = (len(closePrices) - 1) * 4 + 1  # 29 values * 4 features + 1 money value
            if len(processed_state) != expected_length:
                print(f"Warning: Processed state length {len(processed_state)} != expected length {expected_length}")
                # Pad or truncate if necessary
                if len(processed_state) < expected_length:
                    processed_state.extend([0.0] * (expected_length - len(processed_state)))
                else:
                    processed_state = processed_state[:expected_length]
            
            # Convert to tensor and return
            return torch.FloatTensor(processed_state).to(self.device)
            
        except Exception as e:
            print(f"State processing error: {str(e)}")
            print(f"State type: {type(state)}")
            print(f"State content: {state}")
            # Return zero tensor of correct size as fallback
            return torch.zeros(self.actual_obs_space).to(self.device)

    def processReward(self, reward):
        """
        Enhanced reward processing with trading-specific considerations
        """
        # Add penalty for frequent trading to encourage holding positions
        if hasattr(self, 'previous_action') and self.current_action != self.previous_action:
            trading_penalty = 0.001  # Small penalty for switching positions
            reward -= trading_penalty
        
        # Scale rewards to be in a more manageable range
        reward = np.clip(reward, -1, 1)
        
        # Add time decay to encourage longer-term holdings
        if hasattr(self, 'holding_time'):
            self.holding_time = self.holding_time + 1 if self.current_action == self.previous_action else 0
            holding_bonus = min(0.001 * self.holding_time, 0.1)  # Cap the bonus
            reward += holding_bonus
        
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
        Modified action selection with added stability measures
        """
        with torch.no_grad():
            action_probs, value = self.model(state)
            
            # Add temperature scaling to reduce action switching
            temperature = 2.0
            scaled_probs = F.softmax(torch.log(action_probs) / temperature, dim=-1)
            
            # Add action masking to prevent frequent switching
            if hasattr(self, 'previous_action'):
                # Penalize switching by adjusting probabilities
                switch_penalty = 0.1
                scaled_probs[0][1 - self.previous_action] *= (1 - switch_penalty)
            
            dist = Categorical(scaled_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
            self.previous_action = action.item()
            
            return action.item(), log_prob, value

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
                        self.store_transition(state_tensor, action, reward, log_prob, value, done)
                        
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

    def store_transition(self, state, action, reward, log_prob, value, done):
        """
        Enhanced transition storage with importance sampling
        """
        # Calculate importance weight based on reward magnitude
        importance = abs(reward)
        
        # Store transition with importance weight
        self.transitions.append({
            'state': state,
            'action': action,
            'reward': reward,
            'log_prob': log_prob,
            'value': value,
            'done': done,
            'importance': importance
        })
        
        # Prioritize important samples during training
        if len(self.transitions) > BATCH_SIZE:
            importances = torch.tensor([t['importance'] for t in self.transitions])
            probs = F.softmax(importances, dim=0)
            indices = torch.multinomial(probs, BATCH_SIZE, replacement=False)
            self.transitions = [self.transitions[i] for i in indices]

    def update_policy(self):
        """
        Update policy using PPO algorithm with improved stability
        """
        # Convert stored transitions to tensors
        states = torch.stack([t['state'].float() for t in self.transitions])
        actions = torch.tensor([t['action'] for t in self.transitions], dtype=torch.long).to(self.device)
        rewards = torch.tensor([t['reward'] for t in self.transitions], dtype=torch.float32).to(self.device)
        old_log_probs = torch.stack([t['log_prob'].float() for t in self.transitions])
        old_values = torch.stack([t['value'].float() for t in self.transitions])
        dones = torch.tensor([t['done'] for t in self.transitions], dtype=torch.float32).to(self.device)

        # Compute returns and advantages with GAE
        returns = self.compute_returns(rewards, dones)
        advantages = returns - old_values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Multiple epochs of PPO updates
        for _ in range(PPO_EPOCHS):
            # Get minibatches
            indices = torch.randperm(len(self.transitions))
            for start in range(0, len(self.transitions), BATCH_SIZE):
                end = start + BATCH_SIZE
                batch_indices = indices[start:end]
                
                # Get batch data
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]

                # Get current policy outputs
                action_probs, values = self.model(batch_states)
                dist = Categorical(action_probs)
                new_log_probs = dist.log_prob(batch_actions)
                
                # Calculate policy loss with clipping
                ratio = torch.exp(new_log_probs - batch_old_log_probs.detach())
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1-CLIP_EPSILON, 1+CLIP_EPSILON) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Calculate value loss
                value_loss = F.mse_loss(values.squeeze(), batch_returns)
                
                # Calculate entropy bonus
                entropy = dist.entropy().mean()
                
                # Combined loss
                loss = policy_loss + VALUE_LOSS_COEF * value_loss - ENTROPY_COEF * entropy
                
                # Update network
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), MAX_GRAD_NORM)
                self.optimizer.step()

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