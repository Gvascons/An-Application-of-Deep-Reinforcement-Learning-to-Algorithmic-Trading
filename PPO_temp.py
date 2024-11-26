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
from dataAugmentation import DataAugmentation  # Ensure this is imported
from tradingEnv_temp import TradingEnv  # Ensure this is available
import pandas as pd
import traceback

# Create Figures directory if it doesn't exist
if not os.path.exists('Figs'):
    os.makedirs('Figs')

# Adjusted PPO hyperparameters
PPO_PARAMS = {
    'CLIP_EPSILON': 0.3,         # Increased clipping range
    'VALUE_LOSS_COEF': 0.5,
    'ENTROPY_COEF': 0.1,         # Increased entropy coefficient to encourage exploration
    'PPO_EPOCHS': 4,
    'BATCH_SIZE': 32,            # Reduced batch size
    'GAMMA': 0.99,
    'GAE_LAMBDA': 0.95,
    'LEARNING_RATE': 3e-4,       # Adjusted learning rate
    'MAX_GRAD_NORM': 0.5,
    'HIDDEN_SIZE': 256,
    'MEMORY_SIZE': 5000,
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
        
        # Actor head adjusted for new action space
        self.actor = nn.Sequential(
            nn.Linear(self.feature_dim // 2, self.feature_dim // 4),
            nn.ReLU(),
            nn.Linear(self.feature_dim // 4, num_actions)  # num_actions updated to reflect the action space
        )
        
        # Critic head remains the same
        self.critic = nn.Sequential(
            nn.Linear(self.feature_dim // 2, self.feature_dim // 4),
            nn.ReLU(),
            nn.Linear(self.feature_dim // 4, 1)
        )
        
        # Initialize weights with adjusted gain
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # Adjusted weight initialization
            nn.init.orthogonal_(module.weight, gain=nn.init.calculate_gain('relu'))
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
        action_logits = self.actor(features)
        action_probs = F.softmax(action_logits, dim=-1)
        value = self.critic(features)
        
        return action_probs, value

class PPO:
    """Implementation of PPO algorithm for trading"""
    def __init__(self, state_dim, action_dim, device='cpu', marketSymbol=None):
        """Initialize PPO agent"""
        self.device = device
        
        # Calculate input size based on state structure
        self.input_size = state_dim  # Use the actual state dimension
        self.num_actions = 4  # Actions: 0 = Hold, 1 = Buy, 2 = Sell, 3 = Close
        
        print(f"Initializing PPO with input size: {self.input_size}, action size: {self.num_actions}")
        
        # Initialize network with correct input size
        self.network = PPONetwork(self.input_size, self.num_actions).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=PPO_PARAMS['LEARNING_RATE'])
        
        # Initialize memory
        self.memory = deque(maxlen=PPO_PARAMS['MEMORY_SIZE'])
        
        # Initialize training tracking with TensorBoard
        current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_id = f'PPO_{marketSymbol}_{current_time}'
        self.figures_dir = os.path.join('Figs', f'run_{self.run_id}')
        os.makedirs(self.figures_dir, exist_ok=True)
        
        # Initialize TensorBoard writer
        self.writer = SummaryWriter(f'runs/run_{self.run_id}')
        
        # Initialize training step counter
        self.training_step = 0
        
        # Initialize performance tracking
        self.best_reward = float('-inf')
        self.trailing_rewards = deque(maxlen=100)

        # Additional tracking variables
        self.market_symbol = marketSymbol  # Store the market symbol
        self.prev_action = None
        self.prev_state = None

    def getNormalizationCoefficients(self, tradingEnv):
        """
        Same as in TDQN
        """
        # Retrieve the available trading data
        tradingData = tradingEnv.data
        closePrices = tradingData['Close'].tolist()
        lowPrices = tradingData['Low'].tolist()
        highPrices = tradingData['High'].tolist()
        volumes = tradingData['Volume'].tolist()

        # Retrieve the coefficients required for the normalization
        coefficients = []
        margin = 1
        # 1. Close price => returns (absolute) => maximum value (absolute)
        returns = [abs((closePrices[i]-closePrices[i-1])/closePrices[i-1]) for i in range(1, len(closePrices))]
        coeffs = (0, np.max(returns)*margin)
        coefficients.append(coeffs)
        # 2. Low/High prices => Delta prices => maximum value
        deltaPrice = [abs(highPrices[i]-lowPrices[i]) for i in range(len(lowPrices))]
        coeffs = (0, np.max(deltaPrice)*margin)
        coefficients.append(coeffs)
        # 3. Close/Low/High prices => Close price position => no normalization required
        coeffs = (0, 1)
        coefficients.append(coeffs)
        # 4. Volumes => minimum and maximum values
        coeffs = (np.min(volumes)/margin, np.max(volumes)*margin)
        coefficients.append(coeffs)

        return coefficients

    def processState(self, state, coefficients):
        """
        Process the RL state returned by the environment
        (appropriate format and normalization), similar to TDQN
        """
        # Normalization of the RL state
        closePrices = [state[0][i] for i in range(len(state[0]))]
        lowPrices = [state[1][i] for i in range(len(state[1]))]
        highPrices = [state[2][i] for i in range(len(state[2]))]
        volumes = [state[3][i] for i in range(len(state[3]))]

        # 1. Close price => returns => MinMax normalization
        returns = [(closePrices[i]-closePrices[i-1])/closePrices[i-1] for i in range(1, len(closePrices))]
        if coefficients[0][0] != coefficients[0][1]:
            state[0] = [((x - coefficients[0][0])/(coefficients[0][1] - coefficients[0][0])) for x in returns]
        else:
            state[0] = [0 for x in returns]
        # 2. Low/High prices => Delta prices => MinMax normalization
        deltaPrice = [abs(highPrices[i]-lowPrices[i]) for i in range(1, len(lowPrices))]
        if coefficients[1][0] != coefficients[1][1]:
            state[1] = [((x - coefficients[1][0])/(coefficients[1][1] - coefficients[1][0])) for x in deltaPrice]
        else:
            state[1] = [0 for x in deltaPrice]
        # 3. Close/Low/High prices => Close price position => No normalization required
        closePricePosition = []
        for i in range(1, len(closePrices)):
            deltaPrice = abs(highPrices[i]-lowPrices[i])
            if deltaPrice != 0:
                item = abs(closePrices[i]-lowPrices[i])/deltaPrice
            else:
                item = 0.5
            closePricePosition.append(item)
        if coefficients[2][0] != coefficients[2][1]:
            state[2] = [((x - coefficients[2][0])/(coefficients[2][1] - coefficients[2][0])) for x in closePricePosition]
        else:
            state[2] = [0.5 for x in closePricePosition]
        # 4. Volumes => MinMax normalization
        volumes = [volumes[i] for i in range(1, len(volumes))]
        if coefficients[3][0] != coefficients[3][1]:
            state[3] = [((x - coefficients[3][0])/(coefficients[3][1] - coefficients[3][0])) for x in volumes]
        else:
            state[3] = [0 for x in volumes]

        # Process the state structure to obtain the appropriate format
        state = [item for sublist in state for item in sublist]

        # Log the processed state
        # print(f"Processed State: {state}")

        return state

    def processReward(self, reward):
        """
        Same as in TDQN
        """
        rewardClipping = 1  # Assuming this is a global variable or define it here
        processed_reward = np.clip(reward, -rewardClipping, rewardClipping)
        # Log the original and processed reward
        # print(f"Original Reward: {reward}, Processed Reward: {processed_reward}")
        return processed_reward

    def select_action(self, state):
        """Select an action from the current policy"""
        # Convert to tensor
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            probs, value = self.network(state)
            dist = Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        
        # Log action probabilities and selected action
        # print(f"Action Probabilities: {probs.cpu().numpy()}, Selected Action: {action.item()}")
        
        return action.item(), log_prob.item(), value.item()

    def store_transition(self, state, action, reward, next_state, done, log_prob, value):
        """Store a transition in memory"""
        self.memory.append({
            'state': state,
            'action': action,
            'reward': float(reward),
            'next_state': next_state,
            'done': float(done),
            'log_prob': float(log_prob),
            'value': float(value)
        })

    def update_policy(self):
        """Update policy using PPO"""
        if len(self.memory) < PPO_PARAMS['BATCH_SIZE']:
            # print(f"Memory Size: {len(self.memory)} - Not enough samples to update policy.")
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
        # Log raw advantages
        # print(f"Raw Advantages: {advantages}")
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        # Log normalized advantages
        # print(f"Normalized Advantages: {advantages}")

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
                batch_returns = rewards[batch_indices]
                
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
                value_loss = F.mse_loss(values.squeeze(), batch_returns)
                
                # Combine losses
                loss = (policy_loss + 
                       PPO_PARAMS['VALUE_LOSS_COEF'] * value_loss - 
                       PPO_PARAMS['ENTROPY_COEF'] * entropy)
                
                # Update network
                self.optimizer.zero_grad()
                loss.backward()
                
                # Log gradient norms
                # total_norm = 0
                # for p in self.network.parameters():
                #     if p.grad is not None:
                #         param_norm = p.grad.data.norm(2)
                #         total_norm += param_norm.item() ** 2
                # total_norm = total_norm ** 0.5
                # print(f"Gradient Norm: {total_norm}")
                
                nn.utils.clip_grad_norm_(self.network.parameters(), PPO_PARAMS['MAX_GRAD_NORM'])
                self.optimizer.step()
                
                # Log metrics
                # print(f"Policy Loss: {policy_loss.item()}, Value Loss: {value_loss.item()}, Entropy: {entropy.item()}, Total Loss: {loss.item()}")
                
                if self.writer is not None:
                    self.writer.add_scalar('Loss/total', loss.item(), self.training_step)
                    self.writer.add_scalar('Loss/policy', policy_loss.item(), self.training_step)
                    self.writer.add_scalar('Loss/value', value_loss.item(), self.training_step)
                    self.writer.add_scalar('Loss/entropy', entropy.item(), self.training_step)
                
                self.training_step += 1
        
        self.memory.clear()

    def training(self, trainingEnv, trainingParameters=[], verbose=True, rendering=True, plotTraining=True, showPerformance=True):
        """Train the PPO agent"""
        try:
            num_episodes = trainingParameters[0] if trainingParameters else 1
            episode_rewards = []
            performanceTrain = []  # Track training performance
            performanceTest = []   # Track testing performance
            
            # Create run-specific directories and ID
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_id = f"PPO_{trainingEnv.marketSymbol}_{timestamp}"
            
            # Create base directories
            self.figures_dir = os.path.join('Figs', f'run_{self.run_id}')
            os.makedirs(self.figures_dir, exist_ok=True)
            os.makedirs('Results', exist_ok=True)
            
            # Initialize tensorboard writer
            self.writer = SummaryWriter(f'runs/run_{self.run_id}')
            
            # Pass the directories to the training environment
            trainingEnv.figures_dir = self.figures_dir
            trainingEnv.results_dir = self.figures_dir
            
            # Apply data augmentation techniques to improve the training set
            dataAugmentation = DataAugmentation()
            trainingEnvList = dataAugmentation.generate(trainingEnv)
            
            # Initialize testing environment
            if plotTraining or showPerformance:
                marketSymbol = trainingEnv.marketSymbol
                startingDate = trainingEnv.endingDate
                endingDate = '2020-1-1'  # Adjust the ending date as needed
                money = trainingEnv.data['Money'][0]
                stateLength = trainingEnv.stateLength
                transactionCosts = trainingEnv.transactionCosts
                testingEnv = TradingEnv(marketSymbol, startingDate, endingDate, money, stateLength, transactionCosts)
                performanceTest = []
            
            # If required, print the training progression
            if verbose:
                print("Training progression (hardware selected => " + str(self.device) + "):")
            
            for episode in tqdm(range(num_episodes), disable=not(verbose)):
                # Initialize action counters
                action_counts = {action: 0 for action in range(self.num_actions)}
                total_actions = 0
                
                # For each episode, train on the entire set of training environments
                for env_instance in trainingEnvList:
                    # Set the initial RL variables
                    coefficients = self.getNormalizationCoefficients(env_instance)
                    env_instance.reset()
                    startingPoint = random.randrange(len(env_instance.data.index))
                    env_instance.setStartingPoint(startingPoint)
                    state = self.processState(env_instance.state, coefficients)
                    done = False
                    steps = 0
                    
                    # Interact with the training environment until termination
                    while not done:
                        # Choose an action according to the RL policy and the current RL state
                        action, log_prob, value = self.select_action(state)
                        
                        # Update action counters
                        action_counts[action] += 1
                        total_actions += 1
                        
                        # Interact with the environment with the chosen action
                        nextState, reward, done, info = env_instance.step(action)
                        
                        # Process the RL variables retrieved and store the experience
                        reward = self.processReward(reward)
                        nextState_processed = self.processState(nextState, coefficients)
                        self.store_transition(state, action, reward, nextState_processed, done, log_prob, value)
                        
                        # Execute the PPO learning procedure
                        if len(self.memory) >= PPO_PARAMS['BATCH_SIZE']:
                            self.update_policy()
                        
                        # Update the RL state
                        state = nextState_processed
                        steps += 1
                        
                    # Update policy at the end of episode
                    if len(self.memory) >= PPO_PARAMS['BATCH_SIZE']:
                        self.update_policy()
                    
                    # Continuous tracking of the training performance
                    if plotTraining:
                        totalReward = sum([t['reward'] for t in self.memory])
                        episode_rewards.append(totalReward)
                
                # After the episode
                # Update action labels
                action_labels = {0: 'Hold', 1: 'Buy', 2: 'Sell', 3: 'Close'}
                # After the episode
                print(f"\nEpisode {episode}")
                print("Action Distribution during training:")
                for action in range(self.num_actions):
                    count = action_counts[action]
                    percentage = 100 * count / total_actions if total_actions > 0 else 0
                    action_label = action_labels.get(action, f"Action {action}")
                    print(f"{action_label} ({action}): {count} times ({percentage:.1f}%)")
                
                # Optional: Log to TensorBoard
                for action in range(self.num_actions):
                    self.writer.add_scalar(f'Actions/Training_Action_{action}', action_counts[action], episode)
                
                # Compute both training and testing current performances
                if plotTraining or showPerformance:
                    # Training set performance
                    trainingEnv = self.testing(trainingEnv, trainingEnv, rendering=False, showPerformance=False)
                    analyser = PerformanceEstimator(trainingEnv.data)
                    performance = analyser.computeSharpeRatio()
                    performanceTrain.append(performance)
                    self.writer.add_scalar('Training performance (Sharpe Ratio)', performance, episode)
                    trainingEnv.reset()
                    # Testing set performance
                    testingEnv = self.testing(trainingEnv, testingEnv, rendering=False, showPerformance=False)
                    analyser = PerformanceEstimator(testingEnv.data)
                    performance = analyser.computeSharpeRatio()
                    performanceTest.append(performance)
                    self.writer.add_scalar('Testing performance (Sharpe Ratio)', performance, episode)
                    testingEnv.reset()
                
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
                
                self.plotTraining(episode_rewards)
            
            # If required, print and save the strategy performance
            if showPerformance:
                analyser = PerformanceEstimator(trainingEnv.data)
                analyser.run_id = self.run_id  # Pass the full run_id
                analyser.displayPerformance('PPO', phase='training')
            
            return trainingEnv
            
        except Exception as e:
            print(f"Training error: {str(e)}")
            raise
        finally:
            if self.writer is not None:
                self.writer.flush()  # Ensure all pending events are written

    def testing(self, trainingEnv, testingEnv, rendering=True, showPerformance=True):
        """Test the trained policy on new data"""
        try:
            self.network.eval()  # Set to evaluation mode
            coefficients = self.getNormalizationCoefficients(trainingEnv)
            state = testingEnv.reset()
            state = self.processState(state, coefficients)
            done = False
            episode_reward = 0
            actions_taken = []
            
            # Initialize action counters
            action_counts = {action: 0 for action in range(self.num_actions)}
            total_actions = 0
            
            with torch.no_grad():
                while not done:
                    # Choose an action according to the RL policy and the current RL state
                    action_probs, _ = self.network(torch.FloatTensor(state).unsqueeze(0).to(self.device))
                    action = torch.argmax(action_probs, dim=1).item()
                    
                    # Update action counters
                    action_counts[action] += 1
                    total_actions += 1
                    
                    # Interact with the environment with the chosen action
                    nextState, reward, done, _ = testingEnv.step(action)
                    state = self.processState(nextState, coefficients)
                    episode_reward += reward
                    actions_taken.append(action)
            
            # After testing
            action_labels = {0: 'Short', 1: 'Long', 2: 'Hold'}
            print("\nAction Distribution during testing:")
            for action in range(self.num_actions):
                count = action_counts[action]
                percentage = 100 * count / total_actions if total_actions > 0 else 0
                action_label = action_labels.get(action, f"Action {action}")
                print(f"{action_label} ({action}): {count} times ({percentage:.1f}%)")
            
            # Optional: Log to TensorBoard
            for action in range(self.num_actions):
                self.writer.add_scalar(f'Actions/Testing_Action_{action}', action_counts[action])
            
            # If required, show the rendering of the testing environment
            if rendering:
                self.render_to_dir(testingEnv)
            
            # If required, compute and display the strategy performance
            if showPerformance:
                analyser = PerformanceEstimator(testingEnv.data)
                analyser.run_id = self.run_id
                analyser.displayPerformance('PPO', phase='testing')
            
            return testingEnv
            
        except Exception as e:
            print(f"Error in testing: {str(e)}")
            raise

    def plotTraining(self, rewards):
        """Plot the training phase results (rewards)"""
        try:
            fig = plt.figure()
            ax1 = fig.add_subplot(111, ylabel='Total reward collected', xlabel='Episode')
            ax1.plot(rewards)
            plt.savefig(os.path.join(self.figures_dir, 'TrainingResults.png'))
            plt.close(fig)
        except Exception as e:
            print(f"Error in plotTraining: {str(e)}")

    def render_to_dir(self, env):
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

    def __del__(self):
        """Cleanup method"""
        if hasattr(self, 'writer') and self.writer is not None:
            try:
                self.writer.close()
            except:
                pass

    def log_performance_metrics(self, episode, train_sharpe, test_sharpe):
        """Log performance metrics to TensorBoard"""
        if self.writer is not None:
            self.writer.add_scalar('Performance/Train_Sharpe', train_sharpe, episode)
            self.writer.add_scalar('Performance/Test_Sharpe', test_sharpe, episode)
            
            # Log the difference between train and test Sharpe ratios to monitor overfitting
            self.writer.add_scalar('Performance/Train_Test_Gap', train_sharpe - test_sharpe, episode)

    def move_rendering_to_dir(self, env):
        """Move rendering file to the run-specific directory"""
        src_path = os.path.join('Figs', f'{str(env.marketSymbol)}_TrainingTestingRendering.png')
        dst_path = os.path.join(self.figures_dir, f'{str(env.marketSymbol)}_TrainingTestingRendering.png')
        
        if os.path.exists(src_path):
            shutil.move(src_path, dst_path)
