import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- Step 1: Load the CSV file ---
file_path = "CAISO-netdemand-20250606.csv"  # Adjust if using a different path
df = pd.read_csv(file_path)

# --- Step 2: Extract the Net Demand row ---
time_labels = df.columns[1:]  # Exclude the first column with row labels
net_demand_values = df[df[df.columns[0]] == 'Net demand'].iloc[0, 1:].astype(float)

# --- Step 3: Determine hourly tick positions ---
tick_indices = np.arange(0, len(time_labels), 12)  # 5-minute steps → 12 per hour
tick_labels = [time_labels[i] for i in tick_indices]

# --- Step 4: Plot the graph ---
plt.figure(figsize=(15, 5))
plt.plot(time_labels, net_demand_values, label='Net Demand (MW)')
plt.xticks(ticks=tick_indices, labels=tick_labels, rotation=45, fontsize=8)
plt.title("CAISO Net Demand - June 6, 2025")
plt.xlabel("Time of Day (Hourly)")
plt.ylabel("Net Demand (MW)")
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.show()