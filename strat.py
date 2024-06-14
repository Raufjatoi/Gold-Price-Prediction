
#Trading Strategy Development
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

data = pd.read_csv('goldstock.csv')

# Define basic trading strategy
def simple_strategy(data):
    signals = pd.DataFrame(index=data.index)
    signals['Signal'] = 0.0

    # Generate buy signals
    signals['Signal'][1:] = np.where(data['Close'][1:] > data['Close'][:-1], 1.0, 0.0)

    # Generate sell signals
    signals['Signal'][1:] = np.where(data['Close'][1:] < data['Close'][:-1], -1.0, 0.0)

    return signals

# Apply strategy to the dataset
signals = simple_strategy(data)

# Plot buy and sell signals
plt.figure(figsize=(12, 8))
plt.plot(data.index, data['Close'], label='Close Price', alpha=0.35)
plt.plot(signals.loc[signals['Signal'] == 1.0].index,
         data['Close'][signals['Signal'] == 1.0],
         '^', markersize=10, color='g', lw=0, label='Buy Signal')
plt.plot(signals.loc[signals['Signal'] == -1.0].index,
         data['Close'][signals['Signal'] == -1.0],
         'v', markersize=10, color='r', lw=0, label='Sell Signal')
plt.title('Basic Trading Strategy: Buy and Sell Signals')
plt.xlabel('Date')
plt.ylabel('Close Price ($)')
plt.legend()
plt.grid(True)
plt.show()

# Backtesting
def backtest_strategy(data, signals):
    # Initialize a DataFrame to store positions
    positions = pd.DataFrame(index=signals.index).fillna(0.0)

    # Buy signals (1.0): Buy 100 shares
    positions['Position'] = 100 * signals['Signal']

    # Initialize portfolio with value owned
    portfolio = positions.multiply(data['Close'], axis=0)

    # Store the difference in shares owned
    pos_diff = positions.diff()

    # Add 'holdings' to portfolio
    portfolio['Holdings'] = (positions.multiply(data['Close'], axis=0)).sum(axis=1)

    # Add 'cash' to portfolio
    # portfolio['Cash'] = initial_cash - (pos_diff.multiply(data['Close'], axis=0)).sum(axis=1).cumsum()

    return portfolio

# Evaluate the strategy
portfolio = backtest_strategy(data, signals)

# Plotting portfolio
plt.figure(figsize=(14, 7))
plt.plot(portfolio['Holdings'], "b")
plt.title('Portfolio vs Total Shares in Hand')
plt.xlabel('Time')
plt.ylabel('Portfolio')
plt.show()
