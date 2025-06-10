import pandas as pd
import matplotlib.pyplot as plt

# Load LMP CSV file
df = pd.read_csv("LMP_prev_week.csv")

# Parse timestamp
df['INTERVALSTARTTIME_GMT'] = pd.to_datetime(df['INTERVALSTARTTIME_GMT'])

# Extract time-of-day component (ignoring date)
df['TIME'] = df['INTERVALSTARTTIME_GMT'].dt.time

# Group by time and compute average LMP
time_avg = df.groupby('TIME')['MW'].mean().reset_index()
time_avg.columns = ['Time', 'Average_LMP']

# (Optional) Export to CSV
time_avg.to_csv("Avg_LMP.csv", index=False)

# Convert time to string for plotting
time_avg['Time_str'] = time_avg['Time'].astype(str)

# Plot the average LMP per 15-minute interval
plt.figure(figsize=(14, 6))
plt.plot(time_avg['Time_str'], time_avg['Average_LMP'], marker='o')
plt.title("Average LMP Over the Last Week")
plt.xlabel("Time of Day (UTC)")
plt.ylabel("Average LMP ($/MWh)")
plt.xticks(rotation=90)
plt.grid(True)
plt.tight_layout()
plt.show()
