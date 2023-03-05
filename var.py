"""
In this example code, we first define the investment portfolio and the historical returns. 
We then define the confidence level and the time horizon. Finally, we use the Historical 
Simulation method to calculate the VaR, which involves sorting the historical returns, 
calculating the portfolio value based on the sorted returns for the given time horizon, 
and then multiplying it by the square root of the time horizon and the difference between 
1 and the confidence level.
"""

import numpy as np

# Define the investment portfolio and the historical returns
portfolio = np.array([100000, 50000, 25000, 75000, 125000])
returns = np.array([0.05, -0.02, 0.03, -0.01, -0.05, 0.02, -0.03, 0.04, 0.01, -0.02])

# Define the confidence level and the time horizon
confidence_level = 0.95
time_horizon = 252*2 # number of trading days in a year * 2

# Repeat each element in the portfolio array to match the size of the returns array
portfolio_repeated = np.repeat(portfolio, len(returns)//len(portfolio))

# Calculate the VaR using the Historical Simulation method
returns_sorted = np.sort(returns)
portfolio_value = np.dot(portfolio_repeated, returns_sorted[-time_horizon:])
var = portfolio_value * np.sqrt(time_horizon) * (1 - confidence_level)

print(f"Value at Risk (VaR) is (for {time_horizon} trading days):", var)

