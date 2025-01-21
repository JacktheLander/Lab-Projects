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

csv_file = 'Assets/sine_wave.csv'
data = pd.read_csv(csv_file)

x_column = data.columns[0]
y_column = data.columns[1]

# Extract data for plotting
x = data[x_column]
y = data[y_column]

# Plot the sine wave
plt.figure(figsize=(10, 6))
plt.plot(x, y, label='Sine Wave', color='blue')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Sine Wave')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Show and save the plot
FileNameJpg = "Sine_Wave_Plot.jpg"
plt.savefig("Outputs/"+FileNameJpg, format='jpg')
plt.show()  # -* Must do show after save for some reason