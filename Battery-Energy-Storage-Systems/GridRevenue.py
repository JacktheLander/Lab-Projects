import numpy as np
import matplotlib.pyplot as plt

# --- Data ---
megapacks = [25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
profit = [
    3042.74, 3651.29, 4259.84, 4868.38, 5476.93, 6085.48,
    6694.03, 7302.58, 7911.12, 8519.67, 9128.22, 9736.77, 10345.32, 10953.86, 11562.41, 12170.96
]


# --- Constants ---
cost_per_megapack = 1_500_000  # USD

# --- Calculations ---
CapEx = np.array(megapacks) * cost_per_megapack
days_to_pay_off = CapEx / np.array(profit)
years_to_pay_off = days_to_pay_off/ 365 # Convert days to years for the data
yearly_profit = (np.array(profit)*365) - 1_000_000 # Assumed $1M OpEx, yearly profit after paying off CapEx
OpEx_debt = years_to_pay_off * 1_000_000
years_remaining = np.array(OpEx_debt) / np.array(yearly_profit)
years = years_remaining + years_to_pay_off

# --- Plot 1: Profit ---
plt.figure(figsize=(10, 6))
plt.plot(megapacks, profit, marker='o', color='green', label='Daily Profit ($)')
plt.title('Total Arbitrage Revenue vs Number of Megapacks')
plt.xlabel('Number of Megapacks')
plt.ylabel('Daily Revenue ($)')
plt.grid(True)
plt.xticks(range(0, 101, 5))
plt.xlim(0, 100)
plt.legend()
plt.tight_layout()
plt.show()

# --- Plot 2: Expense ---
plt.figure(figsize=(10, 6))
plt.plot(megapacks, CapEx, marker='o', color='blue', linestyle='--', label='Total System Cost')
plt.title('Estimated Capex vs Number of Megapacks')
plt.xlabel('Number of Megapacks')
plt.ylabel('Total Cost ($)')
plt.grid(True)
plt.xticks(range(0, 101, 5))
plt.xlim(0, 100)
plt.ylim(0)
plt.legend()
plt.tight_layout()
plt.show()

# --- Plot 3: Years to Pay Off ---
plt.figure(figsize=(10, 6))
plt.plot(megapacks, years_to_pay_off, marker='o', color='darkgreen', label='Years to Pay CapEx')
plt.plot(megapacks, years, marker='o', color='blue', label='Years to Pay CapEx+OpEx')
plt.title('Years to Pay Off vs Number of Megapacks')
plt.xlabel('Number of Megapacks')
plt.ylabel('Years to Pay Off Debt')
plt.grid(True)
plt.xticks(range(0, 101, 5))
plt.xlim(min(megapacks), max(megapacks))
plt.legend()
plt.tight_layout()
plt.show()
