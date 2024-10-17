import numpy as np
import matplotlib.pyplot as plt

# Define the parameters of the VAR model
phi = 0.95  # persistence of the price-dividend ratio
beta_r = 0.2  # sensitivity of returns to price-dividend ratio
beta_d = 0.1  # sensitivity of dividend growth to price-dividend ratio

# Time periods for the simulation
T = 20  # simulate for 20 periods

# Initialize arrays to store results
dp = np.zeros(T)  # price-dividend ratio
r = np.zeros(T)   # returns
delta_d = np.zeros(T)  # dividend growth

# Initial shock to the system
shock_type = "cash_flow"  # can be "cash_flow" or "discount_rate"
shock = np.zeros(3)  # [return_shock, dividend_growth_shock, price-dividend shock]
if shock_type == "cash_flow":
    shock[1] = 1  # Shock to dividend growth
else:
    shock[0] = 1  # Shock to returns

# VAR simulation
for t in range(1, T):
    # Update the system with the shock applied to the first period
    if t == 1:
        dp[t] = phi * dp[t-1] + shock[2]  # price-dividend ratio
        r[t] = beta_r * dp[t-1] + shock[0]  # returns
        delta_d[t] = beta_d * dp[t-1] + shock[1]  # dividend growth
    else:
        dp[t] = phi * dp[t-1]  # price-dividend ratio
        r[t] = beta_r * dp[t-1]  # returns
        delta_d[t] = beta_d * dp[t-1]  # dividend growth

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(dp, label='Price-Dividend Ratio (dp)', marker='o')
plt.plot(r, label='Returns (r)', marker='x')
plt.plot(delta_d, label='Dividend Growth (Δd)', marker='s')
plt.axvline(x=1, color='gray', linestyle='--', label='Shock Occurs')

plt.title(f'Impulse Response for {shock_type.capitalize()} Shock')
plt.xlabel('Time Periods')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()

# Reinitialize arrays to store results for discount-rate shock
dp_discount_rate = np.zeros(T)  # price-dividend ratio
r_discount_rate = np.zeros(T)   # returns
delta_d_discount_rate = np.zeros(T)  # dividend growth

# Initial shock for discount-rate
shock_type = "discount_rate"
shock = np.zeros(3)  # [return_shock, dividend_growth_shock, price-dividend shock]
shock[0] = 1  # Shock to returns

# VAR simulation for discount-rate shock
for t in range(1, T):
    # Update the system with the shock applied to the first period
    if t == 1:
        dp_discount_rate[t] = phi * dp_discount_rate[t-1] + shock[2]  # price-dividend ratio
        r_discount_rate[t] = beta_r * dp_discount_rate[t-1] + shock[0]  # returns
        delta_d_discount_rate[t] = beta_d * dp_discount_rate[t-1] + shock[1]  # dividend growth
    else:
        dp_discount_rate[t] = phi * dp_discount_rate[t-1]  # price-dividend ratio
        r_discount_rate[t] = beta_r * dp_discount_rate[t-1]  # returns
        delta_d_discount_rate[t] = beta_d * dp_discount_rate[t-1]  # dividend growth

# Plotting the comparison between cash-flow shock and discount-rate shock
plt.figure(figsize=(12, 8))

# Plot for cash-flow shock
plt.subplot(2, 1, 1)
plt.plot(dp, label='Price-Dividend Ratio (dp)', marker='o')
plt.plot(r, label='Returns (r)', marker='x')
plt.plot(delta_d, label='Dividend Growth (Δd)', marker='s')
plt.axvline(x=1, color='gray', linestyle='--', label='Shock Occurs')
plt.title(f'Impulse Response for Cash-Flow Shock')
plt.xlabel('Time Periods')
plt.ylabel('Value')
plt.legend()
plt.grid(True)

# Plot for discount-rate shock
plt.subplot(2, 1, 2)
plt.plot(dp_discount_rate, label='Price-Dividend Ratio (dp)', marker='o')
plt.plot(r_discount_rate, label='Returns (r)', marker='x')
plt.plot(delta_d_discount_rate, label='Dividend Growth (Δd)', marker='s')
plt.axvline(x=1, color='gray', linestyle='--', label='Shock Occurs')
plt.title(f'Impulse Response for Discount-Rate Shock')
plt.xlabel('Time Periods')
plt.ylabel('Value')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

