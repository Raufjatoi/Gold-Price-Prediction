import pandas as pd 

df = pd.read_csv("goldstock.csv")

# Convert the 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Set the 'Date' column as the index
df.set_index('Date', inplace=True)

# Display the first few rows of the DataFrame
print(df.head())