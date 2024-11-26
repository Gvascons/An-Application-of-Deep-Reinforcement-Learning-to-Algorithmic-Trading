# coding=utf-8

"""
Goal: Implement a trading environment compatible with OpenAI Gym.
Authors: Thibaut Théate and Damien Ernst
Institution: University of Liège
"""

###############################################################################
################################### Imports ###################################
###############################################################################

import os
import gym
import math
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
from matplotlib import pyplot as plt

from dataDownloader import AlphaVantage
from dataDownloader import YahooFinance
from dataDownloader import CSVHandler
from fictiveStockGenerator import StockGenerator

###############################################################################
################################ Global variables #############################
###############################################################################

# Boolean handling the saving of the stock market data downloaded
saving = True

# Variable related to the fictive stocks supported
fictiveStocks = ('LINEARUP', 'LINEARDOWN', 'SINUSOIDAL', 'TRIANGLE')

###############################################################################
############################## Class TradingEnv ###############################
###############################################################################

class TradingEnv(gym.Env):
    """
    GOAL: Implement a custom trading environment compatible with OpenAI Gym.

    VARIABLES:
        - data: DataFrame monitoring the trading activity.
        - state: RL state to be returned to the RL agent.
        - reward: RL reward to be returned to the RL agent.
        - done: RL episode termination signal.
        - t: Current trading time step.
        - marketSymbol: Stock market symbol.
        - startingDate: Beginning of the trading horizon.
        - endingDate: Ending of the trading horizon.
        - stateLength: Number of trading time steps included in the state.
        - numberOfShares: Number of shares currently owned by the agent.
        - transactionCosts: Transaction costs associated with the trading
                            activity (e.g., 0.01 is 1% of loss).

    METHODS:
        - __init__: Object constructor initializing the trading environment.
        - reset: Perform a soft reset of the trading environment.
        - step: Transition to the next trading time step.
        - render: Illustrate graphically the trading environment.
    """

    def __init__(self, marketSymbol, startingDate, endingDate, money, stateLength=30,
                 transactionCosts=0, startingPoint=0):
        """
        GOAL: Object constructor initializing the trading environment by setting up
              the trading activity DataFrame as well as other important variables.

        INPUTS:
            - marketSymbol: Stock market symbol.
            - startingDate: Beginning of the trading horizon.
            - endingDate: Ending of the trading horizon.
            - money: Initial amount of money at the disposal of the agent.
            - stateLength: Number of trading time steps included in the RL state.
            - transactionCosts: Transaction costs associated with the trading
                                activity (e.g., 0.01 is 1% of loss).
            - startingPoint: Optional starting point (iteration) of the trading activity.

        OUTPUTS: /
        """

        # CASE 1: Fictive stock generation
        if(marketSymbol in fictiveStocks):
            stockGeneration = StockGenerator()
            if(marketSymbol == 'LINEARUP'):
                self.data = stockGeneration.linearUp(startingDate, endingDate)
            elif(marketSymbol == 'LINEARDOWN'):
                self.data = stockGeneration.linearDown(startingDate, endingDate)
            elif(marketSymbol == 'SINUSOIDAL'):
                self.data = stockGeneration.sinusoidal(startingDate, endingDate)
            else:
                self.data = stockGeneration.triangle(startingDate, endingDate)

        # CASE 2: Real stock loading
        else:
            # Check if the stock market data is already present in the database
            csvConverter = CSVHandler()
            csvName = "".join(['Data/', marketSymbol, '_', startingDate, '_', endingDate])
            exists = os.path.isfile(csvName + '.csv')

            # If affirmative, load the stock market data from the database
            if(exists):
                self.data = csvConverter.CSVToDataframe(csvName)
            # Otherwise, download the stock market data from Yahoo Finance and save it in the database
            else:
                downloader1 = YahooFinance()
                downloader2 = AlphaVantage()
                try:
                    self.data = downloader1.getDailyData(marketSymbol, startingDate, endingDate)
                except:
                    self.data = downloader2.getDailyData(marketSymbol, startingDate, endingDate)

                if saving == True:
                    csvConverter.dataframeToCSV(csvName, self.data)

        # Interpolate in case of missing data
        self.data.replace(0.0, np.nan, inplace=True)
        self.data.interpolate(method='linear', limit=5, limit_area='inside', inplace=True)
        self.data.fillna(method='ffill', inplace=True)
        self.data.fillna(method='bfill', inplace=True)
        self.data.fillna(0, inplace=True)

        # Set the trading activity DataFrame
        self.data['Position'] = 0
        self.data['Action'] = 0
        self.data['Holdings'] = 0.
        self.data['Cash'] = float(money)
        self.data['Money'] = self.data['Holdings'] + self.data['Cash']
        self.data['Returns'] = 0.

        # Set the RL variables common to every OpenAI Gym environment
        self.state = [self.data['Close'][0:stateLength].tolist(),
                      self.data['Low'][0:stateLength].tolist(),
                      self.data['High'][0:stateLength].tolist(),
                      self.data['Volume'][0:stateLength].tolist(),
                      [0]]  # Position
        self.reward = 0.
        self.done = False

        # Set additional variables related to the trading activity
        self.marketSymbol = marketSymbol
        self.startingDate = startingDate
        self.endingDate = endingDate
        self.stateLength = stateLength
        self.t = stateLength
        self.numberOfShares = 0
        self.transactionCosts = transactionCosts
        self.epsilon = 0.1

        # If required, set a custom starting point for the trading activity
        if startingPoint:
            self.setStartingPoint(startingPoint)

        # Update action space to include "Hold" action
        # Update action space to include "Close" action
        self.action_space = 4  # Actions: 0 = Hold, 1 = Buy, 2 = Sell, 3 = Close
        self.observation_space = 5  # Close, Low, High, Volume, Position

    def reset(self):
        """
        GOAL: Perform a soft reset of the trading environment.

        INPUTS: /

        OUTPUTS:
            - state: RL state returned to the trading strategy.
        """

        # Reset the trading activity DataFrame
        self.data['Position'] = 0
        self.data['Action'] = 0
        self.data['Holdings'] = 0.
        self.data['Cash'] = self.data['Cash'][0]
        self.data['Money'] = self.data['Holdings'] + self.data['Cash']
        self.data['Returns'] = 0.

        # Reset the RL variables common to every OpenAI Gym environment
        self.state = [self.data['Close'][0:self.stateLength].tolist(),
                      self.data['Low'][0:self.stateLength].tolist(),
                      self.data['High'][0:self.stateLength].tolist(),
                      self.data['Volume'][0:self.stateLength].tolist(),
                      [0]]  # Position
        self.reward = 0.
        self.done = False

        # Reset additional variables related to the trading activity
        self.t = self.stateLength
        self.numberOfShares = 0

        return self.state

    def computeLowerBound(self, cash, numberOfShares, price):
        """
        GOAL: Compute the lower bound of the complete RL action space,
              i.e., the minimum number of shares to trade.

        INPUTS:
            - cash: Value of the cash owned by the agent.
            - numberOfShares: Number of shares owned by the agent.
            - price: Last price observed.

        OUTPUTS:
            - lowerBound: Lower bound of the RL action space.
        """

        # Computation of the RL action lower bound
        deltaValues = - cash - numberOfShares * price * (1 + self.epsilon) * (1 + self.transactionCosts)
        if deltaValues < 0:
            lowerBound = deltaValues / (price * (2 * self.transactionCosts + (self.epsilon * (1 + self.transactionCosts))))
        else:
            lowerBound = deltaValues / (price * self.epsilon * (1 + self.transactionCosts))
        return lowerBound

    def step(self, action):
        """
        GOAL: Transition to the next trading time step based on the
              trading position decision made (long, short, or hold).

        INPUTS:
            - action: Trading decision (0 = Short, 1 = Long, 2 = Hold).

        OUTPUTS:
            - state: RL state to be returned to the RL agent.
            - reward: RL reward to be returned to the RL agent.
            - done: RL episode termination signal (boolean).
            - info: Additional information returned to the RL agent.
        """

        # Setting of some local variables
        t = self.t
        numberOfShares = self.numberOfShares
        customReward = False

        # CASE 1: HOLD POSITION (Do nothing)
        if action == 0:
            self.data['Position'][t] = self.data['Position'][t - 1]
            self.data['Cash'][t] = self.data['Cash'][t - 1]
            self.data['Holdings'][t] = self.numberOfShares * self.data['Close'][t]
            self.data['Action'][t] = 0  # No action

        # CASE 2: BUY (Long)
        elif action == 1:
            self.data['Position'][t] = 1
            # If already in Long position, maintain or add to it
            if self.data['Position'][t - 1] == 1:
                # Optionally add logic to increase position size
                self.data['Cash'][t] = self.data['Cash'][t - 1]
                self.data['Holdings'][t] = self.numberOfShares * self.data['Close'][t]
            else:
                # Close previous position if any
                if self.data['Position'][t - 1] == -1:
                    # Close Short position
                    self.data['Cash'][t] = self.data['Cash'][t - 1] - self.numberOfShares * self.data['Close'][t] * (1 + self.transactionCosts)
                    self.numberOfShares = 0
                else:
                    self.data['Cash'][t] = self.data['Cash'][t - 1]
                # Open Long position
                self.numberOfShares = math.floor(self.data['Cash'][t] / (self.data['Close'][t] * (1 + self.transactionCosts)))
                self.data['Cash'][t] -= self.numberOfShares * self.data['Close'][t] * (1 + self.transactionCosts)
                self.data['Holdings'][t] = self.numberOfShares * self.data['Close'][t]
                self.data['Action'][t] = 1

        # CASE 3: SELL (Short)
        elif action == 2:
            self.data['Position'][t] = -1
            # If already in Short position, maintain or add to it
            if self.data['Position'][t - 1] == -1:
                # Optionally add logic to increase position size
                self.data['Cash'][t] = self.data['Cash'][t - 1]
                self.data['Holdings'][t] = -self.numberOfShares * self.data['Close'][t]
            else:
                # Close previous position if any
                if self.data['Position'][t - 1] == 1:
                    # Close Long position
                    self.data['Cash'][t] = self.data['Cash'][t - 1] + self.numberOfShares * self.data['Close'][t] * (1 - self.transactionCosts)
                    self.numberOfShares = 0
                else:
                    self.data['Cash'][t] = self.data['Cash'][t - 1]
                # Open Short position
                self.numberOfShares = math.floor(self.data['Cash'][t] / (self.data['Close'][t] * (1 + self.transactionCosts)))
                self.data['Cash'][t] += self.numberOfShares * self.data['Close'][t] * (1 - self.transactionCosts)
                self.data['Holdings'][t] = -self.numberOfShares * self.data['Close'][t]
                self.data['Action'][t] = -1

        # CASE 4: CLOSE POSITION
        elif action == 3:
            self.data['Position'][t] = 0
            # Close any open positions
            if self.data['Position'][t - 1] == 1:
                # Close Long position
                self.data['Cash'][t] = self.data['Cash'][t - 1] + self.numberOfShares * self.data['Close'][t] * (1 - self.transactionCosts)
                self.numberOfShares = 0
            elif self.data['Position'][t - 1] == -1:
                # Close Short position
                self.data['Cash'][t] = self.data['Cash'][t - 1] - self.numberOfShares * self.data['Close'][t] * (1 + self.transactionCosts)
                self.numberOfShares = 0
            else:
                self.data['Cash'][t] = self.data['Cash'][t - 1]
            self.data['Holdings'][t] = 0
            self.data['Action'][t] = 0  # Representing closing position

        else:
            raise SystemExit("Invalid action!")

        # Update total wealth and compute reward
        self.data['Money'][t] = self.data['Holdings'][t] + self.data['Cash'][t]
        self.data['Returns'][t] = (self.data['Money'][t] - self.data['Money'][t - 1]) / self.data['Money'][t - 1]
        self.reward = self.data['Returns'][t]

        # Update state and time
        self.t += 1
        self.state = [self.data['Close'][self.t - self.stateLength: self.t].tolist(),
                    self.data['Low'][self.t - self.stateLength: self.t].tolist(),
                    self.data['High'][self.t - self.stateLength: self.t].tolist(),
                    self.data['Volume'][self.t - self.stateLength: self.t].tolist(),
                    [self.data['Position'][self.t - 1]]]
        self.done = self.t == self.data.shape[0]

        # Return step information
        return self.state, self.reward, self.done, {}
    
    def render(self):
        """
        GOAL: Illustrate graphically the trading activity, by plotting
              both the evolution of the stock market price and the
              evolution of the trading capital. All the trading decisions
              (long and short positions) are displayed as well.

        INPUTS: /

        OUTPUTS: /
        """
        # Set the Matplotlib figure and subplots
        fig = plt.figure(figsize=(10, 8))
        ax1 = fig.add_subplot(211, ylabel='Price', xlabel='Time')
        ax2 = fig.add_subplot(212, ylabel='Capital', xlabel='Time', sharex=ax1)

        # Plot the first graph -> Evolution of the stock market price
        self.data['Close'].plot(ax=ax1, color='blue', lw=2)
        ax1.plot(self.data.loc[self.data['Action'] == 1.0].index,
                 self.data['Close'][self.data['Action'] == 1.0],
                 '^', markersize=5, color='green')
        ax1.plot(self.data.loc[self.data['Action'] == -1.0].index,
                 self.data['Close'][self.data['Action'] == -1.0],
                 'v', markersize=5, color='red')

        # Plot the second graph -> Evolution of the trading capital
        self.data['Money'].plot(ax=ax2, color='blue', lw=2)
        ax2.plot(self.data.loc[self.data['Action'] == 1.0].index,
                 self.data['Money'][self.data['Action'] == 1.0],
                 '^', markersize=5, color='green')
        ax2.plot(self.data.loc[self.data['Action'] == -1.0].index,
                 self.data['Money'][self.data['Action'] == -1.0],
                 'v', markersize=5, color='red')

        # Generation of the two legends and plotting
        ax1.legend(["Price", "Long", "Short"])
        ax2.legend(["Capital", "Long", "Short"])

        plt.savefig(''.join(['Figs/', str(self.marketSymbol), '_Rendering', '.png']))
        plt.close(fig)

    def setStartingPoint(self, startingPoint):
        """
        GOAL: Setting an arbitrary starting point regarding the trading activity.
              This technique is used for better generalization of the RL agent.

        INPUTS:
            - startingPoint: Optional starting point (iteration) of the trading activity.

        OUTPUTS: /
        """

        # Setting a custom starting point
        self.t = np.clip(startingPoint, self.stateLength, len(self.data.index))

        # Set the RL variables common to every OpenAI Gym environment
        self.state = [self.data['Close'][self.t - self.stateLength: self.t].tolist(),
                      self.data['Low'][self.t - self.stateLength: self.t].tolist(),
                      self.data['High'][self.t - self.stateLength: self.t].tolist(),
                      self.data['Volume'][self.t - self.stateLength: self.t].tolist(),
                      [self.data['Position'][self.t - 1]]]  # Update position in state
        if self.t == self.data.shape[0]:
            self.done = True
