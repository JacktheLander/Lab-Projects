import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV and parse timestamps
df = pd.read_csv('SolarOutput_6.6.06.csv')
df['LocalTime'] = pd.to_datetime(df['LocalTime'], format='%m/%d/%y %H:%M')
df.set_index('LocalTime', inplace=True)

# Filter to just the week 2006-06-03 through 2006-06-10
df_week = df.loc['2006-06-06']

# Plot: Commanded Power Profile for that week
plt.figure()
plt.plot(df_week.index, df_week['Power(MW)'])
plt.xlabel('Time (PST)')
plt.ylabel('Power (MW)')
plt.title('50 MW Solar Farm Output: 06-06')
plt.tight_layout()
plt.show()
