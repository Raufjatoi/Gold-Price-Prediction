# importing libs and loading data 
import pandas as pd

# Load dataset 
data = pd.read_csv('goldstock.csv')

# Convert 'Date' column to datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Set 'Date' column as index
data.set_index('Date', inplace=True)

# Display first few rows and columns of the dataset we can displau last ones too by just using tail
print(data.head())

#Visualize time series data
import matplotlib.pyplot as plt

# Plot closing prices
plt.figure(figsize=(12, 6))
plt.plot(data['Close'], label='Closing Price', color='blue')
plt.title('Gold Prices Over Time')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.legend()
plt.grid(True)
plt.show()

#Identify trends and seasonality
from statsmodels.tsa.seasonal import seasonal_decompose

# Decompose time series into trend, seasonal, and residual components
decomposition = seasonal_decompose(data['Close'], model='additive', period=365)

# Plot decomposition components
plt.figure(figsize=(12, 8))
plt.subplot(411)
plt.plot(data['Close'], label='Original', color='blue')
plt.legend(loc='upper left')
plt.subplot(412)
plt.plot(decomposition.trend, label='Trend', color='red')
plt.legend(loc='upper left')
plt.subplot(413)
plt.plot(decomposition.seasonal, label='Seasonal', color='green')
plt.legend(loc='upper left')
plt.subplot(414)
plt.plot(decomposition.resid, label='Residual', color='gray')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
