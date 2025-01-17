from graf.base import *

plt.rcParams['font.family'] = 'Aptos'

# Read GrAF files
graf_R4C1_cc = Graf()
graf_R4C1_cc.load_hdf("C:\\Users\\grant\\Desktop\\Comparable conditions\\R4C1_CE2_ccond.graf")

graf_R3C1_cc = Graf()
graf_R3C1_cc.load_hdf("C:\\Users\\grant\\Desktop\\Comparable conditions\\R3C1_CE2_ccond.graf")

graf_R3C1T3_cc = Graf()
graf_R3C1T3_cc.load_hdf(os.path.join(".", "comparable_conditions", "R3C1T3_CE2.graf"))

# Access X and Y data
R4C1_x = graf_R4C1_cc.axes['Ax0'].traces['Tr0'].x_data
R4C1_y = graf_R4C1_cc.axes['Ax0'].traces['Tr0'].y_data

R3C1_x = graf_R3C1_cc.axes['Ax0'].traces['Tr0'].x_data
R3C1_y = graf_R3C1_cc.axes['Ax0'].traces['Tr0'].y_data

R3C1T3_x = graf_R3C1T3_cc.axes['Ax0'].traces['Tr0'].x_data
R3C1T3_y = graf_R3C1T3_cc.axes['Ax0'].traces['Tr0'].y_data

# Make new plot
plt.figure(1)
plt.plot(R4C1_x, R4C1_y, linestyle=':', marker='.', label='model E')
plt.plot(R3C1_x, R3C1_y, linestyle=':', marker='.', label='model D')
plt.grid(True)
plt.xlabel("2nd Harmonic Frequency (GHz)")
plt.ylabel("Conversion Efficiency (%)")
plt.legend()

# Round to nearest GHz
R4C1_x_rd = [round(xx_) for xx_ in graf_R4C1_cc.axes['Ax0'].traces['Tr0'].x_data]
R4C1_x_rdu = np.unique(R4C1_x_rd)
R4C1_y_mean = []
R4C1_y_stdev = []
R4C1_x_stdev = []
for x in R4C1_x_rdu:
	
	ys = []
	xs = []
	for idx, xrd in enumerate(R4C1_x_rd):
		if xrd == x:
			ys.append(R4C1_y[idx])
			xs.append(R4C1_x[idx])
	
	R4C1_y_mean.append(np.mean(ys))
	R4C1_y_stdev.append(np.std(ys))
	R4C1_x_stdev.append(np.std(xs))

R4C1_y_mean = np.array(R4C1_y_mean)
R4C1_y_stdev = np.array(R4C1_y_stdev)
R4C1_x_stdev = np.array(R4C1_x_stdev)

R3C1_x_rd = [round(xx_) for xx_ in graf_R3C1_cc.axes['Ax0'].traces['Tr0'].x_data]
R3C1_x_rdu = np.unique(R3C1_x_rd)
R3C1_y_mean = []
R3C1_y_stdev = []
R3C1_x_stdev = []
for x in R3C1_x_rdu:
	
	ys = []
	xs = []
	for idx, xrd in enumerate(R3C1_x_rd):
		if xrd == x:
			ys.append(R3C1_y[idx])
			xs.append(R3C1_x[idx])
	
	R3C1_y_mean.append(np.mean(ys))
	R3C1_y_stdev.append(np.std(ys))
	R3C1_x_stdev.append(np.std(xs))

R3C1_y_mean = np.array(R3C1_y_mean)
R3C1_y_stdev = np.array(R3C1_y_stdev)
R3C1_x_stdev = np.array(R3C1_x_stdev)

R3C1T3_x_rd = [round(xx_) for xx_ in graf_R3C1T3_cc.axes['Ax0'].traces['Tr0'].x_data]
R3C1T3_x_rdu = np.unique(R3C1T3_x_rd)
R3C1T3_y_mean = []
R3C1T3_y_stdev = []
R3C1T3_x_stdev = []
for x in R3C1T3_x_rdu:
	
	ys = []
	xs = []
	for idx, xrd in enumerate(R3C1T3_x_rd):
		if xrd == x:
			ys.append(R3C1T3_y[idx])
			xs.append(R3C1T3_x[idx])
	
	R3C1T3_y_mean.append(np.mean(ys))
	R3C1T3_y_stdev.append(np.std(ys))
	R3C1T3_x_stdev.append(np.std(xs))

R3C1T3_y_mean = np.array(R3C1T3_y_mean)
R3C1T3_y_stdev = np.array(R3C1T3_y_stdev)
R3C1T3_x_stdev = np.array(R3C1T3_x_stdev)


# Make new plot
COLOR1 = (0, 0.2, 0.5)
COLOR2 = (0, 0.5, 0.2)
COLOR3 = (0.3, 0, 0.55)

plt.figure(2, figsize=(4.5, 3))
line1, = plt.plot(R4C1_x_rdu, R4C1_y_mean, linestyle=':', marker='.', color=COLOR1)
plt.errorbar(R4C1_x_rdu, R4C1_y_mean, xerr=R4C1_x_stdev, yerr=R4C1_y_stdev, elinewidth=0.5, color=COLOR1)
plt.fill_between(R4C1_x_rdu, R4C1_y_mean - R4C1_y_stdev, R4C1_y_mean + R4C1_y_stdev, color=COLOR1, alpha=0.2, linewidth=0)

line2, = plt.plot(R3C1_x_rdu, R3C1_y_mean, linestyle=':', marker='.', color=COLOR2)
plt.errorbar(R3C1_x_rdu, R3C1_y_mean, xerr=R3C1_x_stdev, yerr=R3C1_y_stdev, elinewidth=0.5, color=COLOR2)
plt.fill_between(R3C1_x_rdu, R3C1_y_mean - R3C1_y_stdev, R3C1_y_mean + R3C1_y_stdev, color=COLOR2, alpha=0.2, linewidth=0)

plt.grid(True)
plt.xlabel("2nd Harmonic Frequency (GHz)")
plt.ylabel("Conversion Efficiency (%)")
plt.legend([line1, line2], ["4.4 $\mu m$ device", "3.7 $\mu m$ device"])
plt.title("Second Harmonic Conversion Efficiency\n$P_{RF}$ = -25 dBm, $I_{dc}$ = 700 $\mu$A, L = 43 mm")
plt.tight_layout()

## Print conditiosn

plt.savefig("PS1_fig2.svg", format='svg')
plt.savefig("PS1_fig2.png", format='png', dpi=500)

print("R4C1 Conditions:")
dict_summary(graf_R4C1_cc.info.conditions)

print("R3C1 Conditions:")
print(graf_R3C1_cc.info.conditions)

plt.figure(3, figsize=(8.5, 5))
# plt.figure(3, figsize=(6, 4))
line1, = plt.plot(R4C1_x_rdu, R4C1_y_mean, linestyle='--', marker='.', color=COLOR1)
plt.errorbar(R4C1_x_rdu, R4C1_y_mean, xerr=R4C1_x_stdev, yerr=R4C1_y_stdev, elinewidth=0.5, color=COLOR1)
plt.fill_between(R4C1_x_rdu, R4C1_y_mean - R4C1_y_stdev, R4C1_y_mean + R4C1_y_stdev, color=COLOR1, alpha=0.2, linewidth=0)

line2, = plt.plot(R3C1_x_rdu, R3C1_y_mean, linestyle='--', marker='.', color=COLOR2)
plt.errorbar(R3C1_x_rdu, R3C1_y_mean, xerr=R3C1_x_stdev, yerr=R3C1_y_stdev, elinewidth=0.5, color=COLOR2, linewidth=0)
plt.fill_between(R3C1_x_rdu, R3C1_y_mean - R3C1_y_stdev, R3C1_y_mean + R3C1_y_stdev, color=COLOR2, alpha=0.2, linewidth=0)

line3, = plt.plot(R3C1T3_x_rdu, R3C1T3_y_mean, linestyle='--', marker='.', color=COLOR3)
plt.errorbar(R3C1T3_x_rdu, R3C1T3_y_mean, xerr=R3C1T3_x_stdev, yerr=R3C1T3_y_stdev, elinewidth=0.5, color=COLOR3, linewidth=0)
plt.fill_between(R3C1T3_x_rdu, R3C1T3_y_mean - R3C1T3_y_stdev, R3C1T3_y_mean + R3C1T3_y_stdev, color=COLOR3, alpha=0.2, linewidth=0)

plt.grid(True)
plt.xlabel("2nd Harmonic Frequency (GHz)")
plt.ylabel("Conversion Efficiency (%)")
plt.legend([line1, line2, line3], ["4.4 $\mu m$ device, L = 43 mm", "3.7 $\mu m$ device, L = 43 mm", "3.7 $\mu m$ device, L = 454 mm"])
plt.title("Second Harmonic Conversion Efficiency\n$P_{RF}$ = -25 dBm, $I_{dc}$ = 700 $\mu$A")
plt.tight_layout()

plt.savefig("PS1_fig3.png", format='png', dpi=500)

plt.show()

