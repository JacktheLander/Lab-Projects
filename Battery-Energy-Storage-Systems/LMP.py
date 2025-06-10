import pandas as pd
import matplotlib.pyplot as plt

# --- Step 1: Load the LMP CSV file ---
lmp_file_path = "LMP_6.6.25.csv"  # Update path if needed
lmp_df = pd.read_csv(lmp_file_path)
# Clean and sort the data before plotting
filtered_df = lmp_df[
    (lmp_df['LMP_TYPE'] == 'LMP') &
    (lmp_df['NODE'] == '0096WD_7_N001')
].copy()

# Convert timestamps
filtered_df['INTERVALSTARTTIME_GMT'] = pd.to_datetime(filtered_df['INTERVALSTARTTIME_GMT'])

# Sort by time
filtered_df = filtered_df.sort_values(by='INTERVALSTARTTIME_GMT')

# Plot the cleaned LMP data
plt.figure(figsize=(15, 6))
plt.plot(filtered_df['INTERVALSTARTTIME_GMT'], filtered_df['MW'], marker='o', linestyle='-', color='blue')
plt.title('Corrected CAISO Day-Ahead LMP - Node 0096WD_7_N001')
plt.xlabel('Time (UTC)')
plt.ylabel('LMP ($/MWh)')
plt.grid(True)
plt.tight_layout()
plt.xticks(rotation=45)
plt.show()

