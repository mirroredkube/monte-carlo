import json
import pandas as pd
import numpy as np
from scipy.stats import norm

# Read the portfolio data from the JSON file
with open('portfolio.json', 'r') as f:
    portfolio_data = json.load(f)

# Convert the data to a Pandas Series
portfolio = pd.Series(portfolio_data)

# Read the returns data from the JSON file
with open('returns.json', 'r') as f:
    returns_data = json.load(f)

# Convert the data to a Pandas DataFrame
returns = pd.DataFrame(returns_data)

# calculate total returns for portfolio
total_returns = np.dot(portfolio, returns.T)

# calculate mean and standard deviation of total returns
mean = np.mean(total_returns)
std_dev = np.std(total_returns)

# set confidence level and time horizon
confidence_level = 0.95
time_horizon = 6 # assuming returns are monthly

# calculate VaR using historical simulation
z_score = norm.ppf(1 - confidence_level)
VaR = mean - std_dev * z_score * np.sqrt(time_horizon)

print("Value at Risk (VaR) for portfolio: ", VaR)
