import pandas as pd
import matplotlib.pyplot as plt
import os

from matplotlib import rcParams
rcParams.update({"figure.autolayout": True})
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
FontSize = 14
font = {"family": "Times New Roman", "size": FontSize}
plt.rc("font", **font)  # pass in the font dict as kwargs
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "cm"

csv_file = 'Assets/Movie_Theater_Earnings_2023.csv'
data = pd.read_csv(csv_file)

# Ensure the columns are correctly named; replace with actual column names if different
date_column = data.columns[0]  # First column
earnings_column = data.columns[1]  # Second column

# Convert dates to datetime and ensure earnings are numeric
data[date_column] = pd.to_datetime(data[date_column], errors='coerce')
data[earnings_column] = pd.to_numeric(data[earnings_column], errors='coerce')

# Group data by month and sum earnings
data['Month'] = data[date_column].dt.to_period('M')
monthly_earnings = data.groupby('Month')[earnings_column].sum().reset_index()

# Extract data for plotting
months = monthly_earnings['Month'].astype(str)
earnings = monthly_earnings[earnings_column]

# Plot the bar chart
plt.figure(figsize=(10, 6))
plt.bar(months, earnings, color='blue')
plt.xlabel('Month')
plt.ylabel('Earnings ($)')
plt.title('Monthly Earnings')
plt.xticks(rotation=45)
plt.tight_layout()

# Show and save the plot
FileNameJpg = "Monthly_Earnings_Plot.jpg"
plt.savefig("Outputs/"+FileNameJpg, format='jpg')
plt.show()
