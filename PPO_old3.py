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

        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(input_size, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU()
        )

        # Actor head
        self.actor = nn.Linear(HIDDEN_SIZE, num_actions)

        # Critic head
        self.critic = nn.Linear(HIDDEN_SIZE, 1)

        # Initialize weights
        for layer in self.shared:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.constant_(layer.bias, 0)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.constant_(self.actor.bias, 0)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.constant_(self.critic.bias, 0)

    def forward(self, x):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        # Verify input shape
        assert x.shape[1] == self.input_size, \
            f"Input shape {x.shape[1]} doesn't match expected shape {self.input_size}"

        shared_out = self.shared(x)
        action_logits = self.actor(shared_out)
        action_probs = F.softmax(action_logits, dim=-1)
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
        # For example, if we have 4 features each with 30 values, plus 1 for position
        self.stateLength = 30
        self.num_features = 4  # Close, RSI, Momentum, Volatility
        self.actual_obs_space = self.stateLength * self.num_features + 1  # +1 for position

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
        Calculate Relative Strength Index
        """
        deltas = np.diff(prices)
        seed = deltas[:period+1]
        up = seed[seed >= 0].sum()/period
        down = -seed[seed < 0].sum()/period

        rs = up/down if down != 0 else 0
        rsi = np.zeros_like(prices)
        rsi[:period] = 100. - 100./(1. + rs)

        for i in range(period, len(prices)):
            delta = deltas[i - 1]
            upval = max(delta, 0)
            downval = -min(delta, 0)

            up = (up*(period-1) + upval)/period
            down = (down*(period-1) + downval)/period

            rs = up/down if down != 0 else 0
            rsi[i] = 100. - 100./(1. + rs)

        return rsi

    def calculate_momentum(self, prices, period=10):
        """
        Calculate price momentum
        """
        momentum = np.zeros_like(prices)
        for i in range(period, len(prices)):
            momentum[i] = prices[i] - prices[i-period]
        momentum[:period] = momentum[period]
        return momentum

    def calculate_volatility(self, prices, period=10):
        """
        Calculate price volatility
        """
        volatility = np.zeros_like(prices)
        for i in range(period, len(prices)):
            volatility[i] = np.std(prices[i-period+1:i+1])
        volatility[:period] = volatility[period]
        return volatility

    def processState(self, state, coefficients):
        """
        Process the RL state
        """
        try:
            # Extract price data
            closes = np.array(state[0], dtype=np.float32)
            lows = np.array(state[1], dtype=np.float32)
            highs = np.array(state[2], dtype=np.float32)
            volumes = np.array(state[3], dtype=np.float32)

            # Ensure all arrays are length stateLength
            def pad_or_truncate(arr, target_len=self.stateLength):
                if len(arr) > target_len:
                    return arr[-target_len:]
                elif len(arr) < target_len:
                    return np.pad(arr, (target_len - len(arr), 0), 'edge')
                return arr

            closes = pad_or_truncate(closes)
            lows = pad_or_truncate(lows)
            highs = pad_or_truncate(highs)
            volumes = pad_or_truncate(volumes)

            # Normalize prices
            if coefficients[0][0] != coefficients[0][1]:
                normalized_closes = (closes - coefficients[0][0]) / (coefficients[0][1] - coefficients[0][0])
            else:
                normalized_closes = np.zeros_like(closes)

            # Calculate technical indicators
            rsi = self.calculate_rsi(closes)
            rsi = pad_or_truncate(rsi)
            momentum = self.calculate_momentum(closes)
            momentum = pad_or_truncate(momentum)
            volatility = self.calculate_volatility(closes)
            volatility = pad_or_truncate(volatility)

            # Combine all features
            processed_state = []
            processed_state.extend(normalized_closes.tolist())
            processed_state.extend(rsi.tolist())
            processed_state.extend(momentum.tolist())
            processed_state.extend(volatility.tolist())

            # Add position information
            position = float(state[-1][0]) if isinstance(state[-1], list) else float(state[-1])
            processed_state.append(position)

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
        Process the RL reward
        """
        # Optionally, modify the reward
        return np.clip(reward, -1, 1)

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
        Select an action based on the current policy
        """
        action_probs, value = self.model(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
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
            num_episodes = trainingParameters[0] if trainingParameters else 100
            for episode in tqdm(range(num_episodes), disable=not verbose):
                for i in range(len(trainingEnvList)):
                    # Track total reward per episode
                    totalReward = 0

                    # Reset environment with random starting point
                    coefficients = self.getNormalizationCoefficients(trainingEnvList[i])
                    trainingEnvList[i].reset()
                    startingPoint = random.randrange(len(trainingEnvList[i].data.index))
                    trainingEnvList[i].setStartingPoint(startingPoint)
                    state = trainingEnvList[i].state
                    done = False

                    # Episode loop
                    while not done:
                        # Process state and select action
                        state_tensor = self.processState(state, coefficients)
                        action, log_prob, value = self.select_action(state_tensor)

                        # Take action in environment
                        next_state, reward, done, info = trainingEnvList[i].step(action)
                        totalReward += reward

                        # Process next state
                        next_state_tensor = self.processState(next_state, coefficients)

                        # Store transition
                        self.store_transition(state_tensor, action, log_prob, value, reward, next_state_tensor, done)

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
                    plt.savefig(os.path.join(self.figures_dir, f'TrainingResults_{i}.png'))
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
        coefficients = self.getNormalizationCoefficients(trainingEnv)
        state = testingEnv.reset()
        done = False

        while not done:
            state_tensor = self.processState(state, coefficients)
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

    def store_transition(self, state, action, log_prob, value, reward, next_state, done):
        """
        Store transitions with log probabilities and values
        """
        priority = abs(reward) + 0.01
        self.transitions.append({
            'state': state.detach(),
            'action': action,
            'log_prob': log_prob.detach(),
            'value': value.detach(),
            'reward': reward,
            'next_state': next_state.detach(),
            'done': done,
            'priority': priority
        })

        # Keep most important transitions when buffer is full
        if len(self.transitions) > self.max_size:
            self.transitions.sort(key=lambda x: x['priority'], reverse=True)
            self.transitions = self.transitions[:self.max_size]

    def update_policy(self):
        """
        Update policy using PPO loss
        """
        if len(self.transitions) < self.batch_size:
            return

        for _ in range(PPO_EPOCHS):
            # Sample batch
            batch = self.sample_batch()
            states = torch.stack([t['state'] for t in batch]).to(self.device)
            actions = torch.tensor([t['action'] for t in batch], dtype=torch.long).to(self.device)
            old_log_probs = torch.stack([t['log_prob'] for t in batch]).to(self.device)
            old_values = torch.stack([t['value'] for t in batch]).squeeze(-1).to(self.device)
            rewards = torch.tensor([t['reward'] for t in batch], dtype=torch.float32).to(self.device)
            dones = torch.tensor([t['done'] for t in batch], dtype=torch.float32).to(self.device)

            # Compute returns and advantages
            with torch.no_grad():
                # We need next_values for GAE
                next_states = torch.stack([t['next_state'] for t in batch]).to(self.device)
                _, next_values = self.model(next_states)
                next_values = next_values.squeeze(-1)
                returns, advantages = self.compute_returns(rewards, dones, old_values, next_values)
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Forward pass
            action_probs, values = self.model(states)
            dist = Categorical(action_probs)
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            values = values.squeeze(-1)

            # PPO ratio
            ratio = torch.exp(log_probs - old_log_probs)

            # Surrogate loss
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - CLIP_EPSILON, 1 + CLIP_EPSILON) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss = F.mse_loss(values, returns)

            # Total loss
            loss = policy_loss + VALUE_LOSS_COEF * value_loss - ENTROPY_COEF * entropy

            # Update
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), MAX_GRAD_NORM)
            self.optimizer.step()

    def compute_returns(self, rewards, dones, values, next_values):
        """
        Compute returns and advantages using GAE
        """
        returns = torch.zeros_like(rewards).to(self.device)
        advantages = torch.zeros_like(rewards).to(self.device)
        running_return = next_values[-1] * (1 - dones[-1])
        running_advantage = 0

        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + GAMMA * running_return * (1 - dones[t])
            returns[t] = running_return

            delta = rewards[t] + GAMMA * next_values[t] * (1 - dones[t]) - values[t]
            running_advantage = delta + GAMMA * GAE_LAMBDA * running_advantage * (1 - dones[t])
            advantages[t] = running_advantage

        return returns, advantages

    def sample_batch(self):
        """
        Sample a batch of transitions
        """
        if len(self.transitions) < self.batch_size:
            return self.transitions  # Return all transitions if less than batch size

        # Sample indices randomly
        indices = np.random.choice(len(self.transitions), size=self.batch_size, replace=False)

        return [self.transitions[i] for i in indices]
