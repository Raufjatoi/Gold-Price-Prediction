# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# dataset has columns: Date, Close, Volume, Open, High, Low by just print describe you can find it 
data = pd.read_csv('goldstock.csv')

# Convert 'Date' column to datetime format if not already
data['Date'] = pd.to_datetime(data['Date'])

# Display the first few rows of the dataset to verify
print("First few rows of the dataset:")
print(data.head())

# Statistical Analysis

# Calculate basic statistics
basic_stats = data.describe()

# Compute correlations
correlations = data.corr()

# Print basic statistics and correlation matrix
print("\nBasic Statistics:")
print(basic_stats)

print("\nCorrelation Matrix:")
print(correlations)

test_result = stats.ttest_ind(data['Close'], data['Volume'])

print("\nStatistical Test Results:")
print("T-test p-value:", test_result.pvalue)
if test_result.pvalue < 0.05:
    print("Statistically significant difference detected.")
else:
    print("No statistically significant difference detected.")

# Plot additional statistical analyses (e.g., distribution plots)
plt.figure(figsize=(10, 6))
plt.hist(data['Close'], bins=30, alpha=0.75, color='b', label='Close Price Distribution')
plt.title('Distribution of Closing Prices')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
plt.show()
