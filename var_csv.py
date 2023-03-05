import pandas as pd
import numpy as np
from scipy.stats import norm

# read portfolio data from CSV file
portfolio_df = pd.read_csv('portfolio.csv', header=None)
portfolio = np.array(portfolio_df[0])

# read returns data from CSV file
returns_df = pd.read_csv('returns.csv', header=None)
returns = np.array(returns_df)

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
