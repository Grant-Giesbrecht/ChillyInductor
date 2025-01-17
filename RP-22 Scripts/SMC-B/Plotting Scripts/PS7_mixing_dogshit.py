import pandas as pd
import matplotlib.pyplot as plt

# Path to your Keysight ADS CSV file
file_path = "C:\\Users\\grant\\Documents\\ADS_Sims\\Kinetic_Inductance_Sim1_wrk\\manual_export_data\\mixing_sim_whatever.csv"

# Reading the CSV file
data = pd.read_csv(file_path)

START_IDX = 0
END_IDX = 18

# Displaying the first few rows of the dataset


plt.figure(1, figsize=(10, 5))
plt.plot(data['L'][START_IDX:END_IDX]*1e3, data['h1'][START_IDX:END_IDX], linestyle=':', marker='o', label="Fundamental")
plt.plot(data['L'][START_IDX:END_IDX]*1e3, data['h2'][START_IDX:END_IDX], linestyle=':', marker='o', label="2nd Harmonic")
plt.plot(data['L'][START_IDX:END_IDX]*1e3, data['h3'][START_IDX:END_IDX], linestyle=':', marker='o', label="3rd Harmonic")
plt.grid()
plt.xlabel("Device Length (mm)")
plt.legend()
plt.ylabel("Tone Power (dBm)")
plt.title("Simulated Length Dependence, $f_{RF}$ = 4 GHz, $I_{dc}$ = 5 mA")
plt.tight_layout()
plt.savefig("PS7_fig1.png", dpi=500)
plt.show()

