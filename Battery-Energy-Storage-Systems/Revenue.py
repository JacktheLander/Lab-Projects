import numpy as np
import matplotlib.pyplot as plt

# --- Data ---
megapacks = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
profit = [
    8287.89, 8882.55, 9477.20, 10064.25, 10616.56, 11161.00, 11704.40, 12259.88, 12831.85, 13461.49,
    14068.85, 14641.88, 15219.65, 15800.21, 16280.80, 16542.13, 16536.25, 16530.38, 16524.50, 16518.63
]

# --- Constants ---
cost_per_megapack = 1_500_000  # USD

# --- Calculations ---
expenses = np.array(megapacks) * cost_per_megapack
CapEx = expenses + 35_000_000 # Assumed $1/MW - 30% Investment Tax Credit (ITC), for solar farm
days_to_pay_off = CapEx / np.array(profit)
years_to_pay_off = days_to_pay_off/ 365 # Convert days to years for the data
yearly_profit = (np.array(profit)*365) - 1_250_000 # Assumed $1.25M/yr OpEx, yearly profit after paying off CapEx
OpEx_debt = years_to_pay_off * 1_250_000
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
plt.xlim(5, 100)
plt.legend()
plt.tight_layout()
plt.show()

# --- Plot 2: Expense ---
plt.figure(figsize=(10, 6))
plt.plot(megapacks, expenses, marker='o', color='red', label='Megapack Cost')
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
