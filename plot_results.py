import pandas as pd

import matplotlib.pyplot as plt

# Create a DataFrame from the data
df = pd.read_csv('data.csv', header=0)

max_mass = df['mass'].max()

# Normalize the data
# df['Normalized'] = df['Error'].abs() / df['Mass']
df['Normalized'] = df['error'].abs() / df['actual'].abs() * 100

mae = df['error'].abs().mean()
print("Mean Absolute Error (MAE):", mae)

# df = df[df['mass'] < 1000]

# Plot the normalized data
plt.scatter(df['mass'], df['Normalized'])
plt.xlabel('Data Point')
plt.ylabel('Normalized Value')
plt.title('Normalized Data Plot')
plt.savefig("plot.png")