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
CLIP_EPSILON = 0.2
VALUE_LOSS_COEF = 0.5
ENTROPY_COEF = 0.01
PPO_EPOCHS = 4
BATCH_SIZE = 32
GAMMA = 0.99
GAE_LAMBDA = 0.95

# Neural network parameters
HIDDEN_SIZE = 256
LEARNING_RATE = 0.0003
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
        # Set device (GPU if available, else CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Calculate the correct observation space size
        self.actual_obs_space = 30 * 4 + 1  # Should be 121
        
        # Ensure model uses float32
        self.model = ActorCritic(self.actual_obs_space, actionSpace).to(self.device).float()
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE, weight_decay=L2_FACTOR)
        
        self.marketSymbol = marketSymbol
        self.actionSpace = actionSpace
        self.transitions = []
        
    def processState(self, state, coefficients):
        """
        Process the RL state into a flat tensor with proper normalization
        """
        processed = []
        
        # Process the first 4 components (Close, Low, High, Volume)
        for i in range(len(state) - 1):
            processed.extend([float(x)/coefficients[i] for x in state[i]])
        
        # Add the position indicator
        processed.extend([float(x) for x in state[-1]])
        
        # Convert to tensor and ensure float32
        state_tensor = torch.FloatTensor(processed).to(self.device)
        
        # Verify the shape matches our expected size
        assert state_tensor.shape[0] == self.actual_obs_space, \
            f"Processed state size {state_tensor.shape[0]} doesn't match expected size {self.actual_obs_space}"
        
        return state_tensor

    def processReward(self, reward):
        """
        Process the RL reward
        """
        return torch.FloatTensor([reward]).to(self.device)

    def getNormalizationCoefficients(self, env):
        """
        Get normalization coefficients for the environment
        """
        return [env.data['Close'].max(),
                env.data['Low'].max(),
                env.data['High'].max(),
                env.data['Volume'].max()]

    def select_action(self, state):
        """
        Select action using the policy network
        """
        self.model.eval()
        with torch.no_grad():
            # Ensure state is float32
            state = state.float()
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
            # If required, print the training progression
            if verbose:
                print("Training progression (hardware selected => " + str(self.device) + "):")

            # Setup training environment and logging
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_id = f"{trainingEnv.marketSymbol}_{timestamp}"
            self.figures_dir = os.path.join('Figures', f'run_{self.run_id}')
            os.makedirs(self.figures_dir, exist_ok=True)
            self.writer = SummaryWriter(f'runs/run_{self.run_id}')

            # Share figures_dir with environments for consistent file saving
            trainingEnv.figures_dir = self.figures_dir
            
            # Initialize performance tracking
            if plotTraining:
                performanceTrain = []
                performanceTest = []
                episode_rewards = []  # Add this to track rewards for TrainingResults.png
                
                # Create a copy of training environment for testing
                testingEnv = copy.deepcopy(trainingEnv)
                testingEnv.figures_dir = self.figures_dir  # Share figures_dir with testing env

            # Training loop
            num_episodes = trainingParameters[0] if trainingParameters else 1
            for episode in tqdm(range(num_episodes), disable=not verbose):
                # Reset environment
                state = trainingEnv.reset()
                done = False
                episode_reward = 0
                
                # Episode loop
                while not done:
                    # Process state and select action
                    state_tensor = self.processState(state, self.getNormalizationCoefficients(trainingEnv))
                    action, log_prob, value = self.select_action(state_tensor)
                    
                    # Take action in environment
                    next_state, reward, done, info = trainingEnv.step(action)
                    episode_reward += reward
                    
                    # Store transition for PPO update
                    self.store_transition(state_tensor, action, reward, log_prob, value, done)
                    
                    # Update policy if enough transitions are collected
                    if len(self.transitions) >= BATCH_SIZE:
                        self.update_policy()
                        self.transitions.clear()
                    
                    state = next_state

                # Log episode results
                self.writer.add_scalar('Episode Reward', episode_reward, episode)
                
                # Compute and log performance metrics
                if plotTraining:
                    # Training performance
                    trainingPerformanceEnv = copy.deepcopy(trainingEnv)
                    trainingPerformanceEnv = self.testing(trainingEnv, trainingPerformanceEnv)
                    analyser = PerformanceEstimator(trainingPerformanceEnv.data)
                    performance = analyser.computeSharpeRatio()
                    performanceTrain.append(performance)
                    self.writer.add_scalar('Training Performance (Sharpe)', performance, episode)

                    # Testing performance
                    testingPerformanceEnv = copy.deepcopy(testingEnv)
                    testingPerformanceEnv = self.testing(trainingEnv, testingPerformanceEnv)
                    analyser = PerformanceEstimator(testingPerformanceEnv.data)
                    performance = analyser.computeSharpeRatio()
                    performanceTest.append(performance)
                    self.writer.add_scalar('Testing Performance (Sharpe)', performance, episode)

                # Optional rendering
                if rendering and episode % 10 == 0:
                    self.render_to_dir(trainingEnv)

                # Restore initial weights if using multiple iterations
                if hasattr(trainingParameters, '__len__') and len(trainingParameters) > 1:
                    if episode < (num_episodes - 1):
                        trainingEnv.reset()
                        testingEnv.reset()
                        self.model.load_state_dict(initialWeights)
                        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE, weight_decay=L2_FACTOR)
                        self.transitions.clear()

            # Assess the algorithm performance on the training trading environment
            trainingEnv = self.testing(trainingEnv, trainingEnv)

            # If required, show the rendering of the trading environment
            if rendering:
                self.render_to_dir(trainingEnv)

            # If required, plot the training results
            if plotTraining:
                fig = plt.figure()
                ax = fig.add_subplot(111, ylabel='Performance (Sharpe Ratio)', xlabel='Episode')
                ax.plot(performanceTrain)
                ax.plot(performanceTest)
                ax.legend(["Training", "Testing"])
                plt.savefig(os.path.join(self.figures_dir, f'TrainingTestingPerformance.png'))
                plt.close(fig)

            # If required, print and save the strategy performance
            if showPerformance:
                analyser = PerformanceEstimator(trainingEnv.data)
                analyser.run_id = self.run_id
                analyser.displayPerformance('PPO', phase='training')

            # Plot training results (similar to TDQN's plotTraining)
            if plotTraining:
                fig = plt.figure()
                ax1 = fig.add_subplot(111, ylabel='Total reward collected', xlabel='Episode')
                ax1.plot(episode_rewards)
                plt.savefig(os.path.join(self.figures_dir, 'TrainingResults.png'))
                plt.close(fig)

            return trainingEnv

        except Exception as e:
            print(f"Training interrupted: {str(e)}")
            import traceback
            traceback.print_exc()
            return trainingEnv
        finally:
            if hasattr(self, 'writer'):
                self.writer.flush()  # Ensure all pending events are written

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
        Store a transition for PPO update
        """
        self.transitions.append({
            'state': state,
            'action': action,
            'reward': reward,
            'log_prob': log_prob,
            'value': value,
            'done': done
        })

    def update_policy(self):
        """
        Update policy using PPO algorithm
        """
        # Convert stored transitions to tensors with explicit float32 dtype
        states = torch.stack([t['state'].float() for t in self.transitions])
        actions = torch.tensor([t['action'] for t in self.transitions], dtype=torch.long).to(self.device)
        rewards = torch.tensor([t['reward'] for t in self.transitions], dtype=torch.float32).to(self.device)
        old_log_probs = torch.stack([t['log_prob'].float() for t in self.transitions])
        old_values = torch.stack([t['value'].float() for t in self.transitions])
        dones = torch.tensor([t['done'] for t in self.transitions], dtype=torch.float32).to(self.device)

        # Compute returns and advantages
        returns = self.compute_returns(rewards, dones)
        advantages = returns - old_values.detach()
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update for multiple epochs
        for _ in range(PPO_EPOCHS):
            # Get new action probabilities and values
            action_probs, values = self.model(states)
            dist = Categorical(action_probs)
            new_log_probs = dist.log_prob(actions)
            
            # Compute PPO loss
            ratio = torch.exp(new_log_probs - old_log_probs.detach())
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1-CLIP_EPSILON, 1+CLIP_EPSILON) * advantages
            
            # Calculate actor and critic losses
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = F.mse_loss(values.squeeze(), returns)
            entropy = dist.entropy().mean()
            
            # Combined loss
            loss = actor_loss + VALUE_LOSS_COEF * critic_loss - ENTROPY_COEF * entropy
            
            # Update network
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), MAX_GRAD_NORM)
            self.optimizer.step()

    def compute_returns(self, rewards, dones):
        """
        Compute returns using GAE (Generalized Advantage Estimation)
        """
        returns = torch.zeros_like(rewards, dtype=torch.float32)
        running_return = 0.0
        
        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + GAMMA * running_return * (1 - dones[t])
            returns[t] = running_return
            
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        return returns