import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Aptos'

widths = [4.4, 3.7, 3.3, 3, 2.5]
Z_narrow = [29.37, 34.7, 38.76, 42.42, 50.14]
Z_nominal = [38.11, 45.05, 50.21, 54.87, 64.91]
Z_wide = [50.13, 59.20, 65.88, 72.31, 84.95]

plt.figure(1, figsize=(6, 4))
COLOR1 = (0, 0.2, 0.5)
COLOR2 = (0, 0.5, 0.2)

lineNom, = plt.plot(widths, Z_nominal, linestyle=':', marker='s', color=COLOR1)
lineNar, = plt.plot(widths, Z_narrow, linestyle=':', marker='v', color=COLOR1)
lineWid, = plt.plot(widths, Z_wide, linestyle=':', marker='^', color=COLOR1)
plt.fill_between(widths, Z_narrow, Z_wide, color=COLOR1, alpha=0.2, linewidth=0)

plt.xlabel("Trace Width  ($\mu$m)")
plt.ylabel("Characteristic Impedance")
plt.legend([lineWid, lineNom, lineNar], ["Upper Bound", "Nominal", "Lower Bound"])
plt.grid(True)
plt.title("Chip Impedance Uncertainty")


plt.savefig("PS2_fig1.png", dpi=500)

plt.show()