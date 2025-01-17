from graf.base import *
import os

def get_colormap_colors(colormap_name, n):
	"""
	Returns 'n' colors as tuples that are evenly spaced in the specified colormap.
	
	Parameters:
	colormap_name (str): The name of the colormap.
	n (int): The number of colors to return.
	
	Returns:
	list: A list of 'n' colors as tuples.
	"""
	cmap = plt.get_cmap(colormap_name)
	colors = [cmap(i / (n - 1)) for i in range(n)]
	return colors

TWILIGHT10 = get_colormap_colors('plasma', 10)

plt.rcParams['font.family'] = 'Aptos'

# Read GrAF files
graf_13 = Graf()

powers = [-13, -15, -17, -19, -21, -23, -25, -30]
x_data_lists = []
y_data_lists = []
x_data_lists_b = []
y_data_lists_b = []

folder_name = "bias_vs_power_RP22B_MP3a_t8_24Sept2024_R3C1T2_r1"

graf_13.load_hdf(os.path.join(".", folder_name, "P-13dBm.graf"))
x_13 = graf_13.axes['Ax0'].traces['Tr0'].x_data[1:]
y_13 = graf_13.axes['Ax0'].traces['Tr0'].y_data[1:]
y_data_lists.append(y_13)
x_data_lists.append(x_13)

graf_13.load_hdf(os.path.join(".", folder_name, "P-15dBm.graf"))
x_15 = graf_13.axes['Ax0'].traces['Tr0'].x_data[1:]
y_15 = graf_13.axes['Ax0'].traces['Tr0'].y_data[1:]
y_data_lists.append(y_15)
x_data_lists.append(x_15)

graf_13.load_hdf(os.path.join(".", folder_name, "P-17dBm.graf"))
x_17 = graf_13.axes['Ax0'].traces['Tr0'].x_data[1:]
y_17 = graf_13.axes['Ax0'].traces['Tr0'].y_data[1:]
y_data_lists.append(y_17)
x_data_lists.append(x_17)

graf_13.load_hdf(os.path.join(".", folder_name, "P-19dBm.graf"))
x_19 = graf_13.axes['Ax0'].traces['Tr0'].x_data[1:]
y_19 = graf_13.axes['Ax0'].traces['Tr0'].y_data[1:]
y_data_lists.append(y_19)
x_data_lists.append(x_19)

graf_13.load_hdf(os.path.join(".", folder_name, "P-21dBm.graf"))
x_21 = graf_13.axes['Ax0'].traces['Tr0'].x_data[1:]
y_21 = graf_13.axes['Ax0'].traces['Tr0'].y_data[1:]
y_data_lists.append(y_21)
x_data_lists.append(x_21)

graf_13.load_hdf(os.path.join(".", folder_name, "P-23dBm.graf"))
x_23 = graf_13.axes['Ax0'].traces['Tr0'].x_data[1:]
y_23 = graf_13.axes['Ax0'].traces['Tr0'].y_data[1:]
y_data_lists.append(y_23)
x_data_lists.append(x_23)

graf_13.load_hdf(os.path.join(".", folder_name, "P-25dBm.graf"))
x_25 = graf_13.axes['Ax0'].traces['Tr0'].x_data[1:]
y_25 = graf_13.axes['Ax0'].traces['Tr0'].y_data[1:]
y_data_lists.append(y_25)
x_data_lists.append(x_25)

graf_13.load_hdf(os.path.join(".", folder_name, "P-30dBm.graf"))
x_30 = graf_13.axes['Ax0'].traces['Tr0'].x_data[1:]
y_30 = graf_13.axes['Ax0'].traces['Tr0'].y_data[1:]
y_data_lists.append(y_30)
x_data_lists.append(x_30)

folder_name_b = "bias_vs_power_RP22B_MP3a_t8_13Sept2024_R4C1T2_r1"

graf_13.load_hdf(os.path.join(".", folder_name_b, "P-13dBm.graf"))
x_13b = graf_13.axes['Ax0'].traces['Tr0'].x_data[1:]
y_13b = graf_13.axes['Ax0'].traces['Tr0'].y_data[1:]
y_data_lists_b.append(y_13b)
x_data_lists_b.append(x_13b)

graf_13.load_hdf(os.path.join(".", folder_name_b, "P-15dBm.graf"))
x_15b = graf_13.axes['Ax0'].traces['Tr0'].x_data[1:]
y_15b = graf_13.axes['Ax0'].traces['Tr0'].y_data[1:]
y_data_lists_b.append(y_15b)
x_data_lists_b.append(x_15b)

graf_13.load_hdf(os.path.join(".", folder_name_b, "P-17dBm.graf"))
x_17b = graf_13.axes['Ax0'].traces['Tr0'].x_data[1:]
y_17b = graf_13.axes['Ax0'].traces['Tr0'].y_data[1:]
y_data_lists_b.append(y_17b)
x_data_lists_b.append(x_17b)

graf_13.load_hdf(os.path.join(".", folder_name_b, "P-19dBm.graf"))
x_19b = graf_13.axes['Ax0'].traces['Tr0'].x_data[1:]
y_19b = graf_13.axes['Ax0'].traces['Tr0'].y_data[1:]
y_data_lists_b.append(y_19b)
x_data_lists_b.append(x_19b)

graf_13.load_hdf(os.path.join(".", folder_name_b, "P-21dBm.graf"))
x_21b = graf_13.axes['Ax0'].traces['Tr0'].x_data[1:]
y_21b = graf_13.axes['Ax0'].traces['Tr0'].y_data[1:]
y_data_lists_b.append(y_21b)
x_data_lists_b.append(x_21b)

graf_13.load_hdf(os.path.join(".", folder_name_b, "P-23dBm.graf"))
x_23b = graf_13.axes['Ax0'].traces['Tr0'].x_data[1:]
y_23b = graf_13.axes['Ax0'].traces['Tr0'].y_data[1:]
y_data_lists_b.append(y_23b)
x_data_lists_b.append(x_23b)

graf_13.load_hdf(os.path.join(".", folder_name_b, "P-25dBm.graf"))
x_25b = graf_13.axes['Ax0'].traces['Tr0'].x_data[1:]
y_25b = graf_13.axes['Ax0'].traces['Tr0'].y_data[1:]
y_data_lists_b.append(y_25b)
x_data_lists_b.append(x_25b)

graf_13.load_hdf(os.path.join(".", folder_name_b, "P-30dBm.graf"))
x_30b = graf_13.axes['Ax0'].traces['Tr0'].x_data[1:]
y_30b = graf_13.axes['Ax0'].traces['Tr0'].y_data[1:]
y_data_lists_b.append(y_30b)
x_data_lists_b.append(x_30b)

sc_bias_list = []
norm_bias_list = []
THRESHOLD = 10
for xidx, ydl in enumerate(y_data_lists):
	
	ydl = np.array(ydl)
	
	idx_norm = np.where(ydl > THRESHOLD)
	sc_bias_list.append(x_data_lists[xidx][idx_norm[0][0]-1])
	norm_bias_list.append(x_data_lists[xidx][idx_norm[0][0]])
sc_bias_list = np.array(sc_bias_list)
norm_bias_list = np.array(norm_bias_list)

sc_bias_list_b = []
norm_bias_list_b = []
THRESHOLD = 20
for xidx, ydl in enumerate(y_data_lists_b):
	
	ydl = np.array(ydl)
	
	idx_norm = np.where(ydl > THRESHOLD)
	sc_bias_list_b.append(x_data_lists_b[xidx][idx_norm[0][0]-1])
	norm_bias_list_b.append(x_data_lists_b[xidx][idx_norm[0][0]])
	
	print(idx_norm)
	
sc_bias_list_b = np.array(sc_bias_list_b)
norm_bias_list_b = np.array(norm_bias_list_b)

# Make new plot
plt.figure(1)
plt.plot(x_13, y_13, linestyle=':', marker='.', color=TWILIGHT10[1], label="P=-13 dBm")
plt.plot(x_15, y_15, linestyle=':', marker='.', color=TWILIGHT10[2], label="P=-15 dBm")
plt.plot(x_17, y_17, linestyle=':', marker='.', color=TWILIGHT10[3], label="P=-17 dBm")
plt.plot(x_19, y_19, linestyle=':', marker='.', color=TWILIGHT10[4], label="P=-19 dBm")
plt.plot(x_21, y_21, linestyle=':', marker='.', color=TWILIGHT10[5], label="P=-21 dBm")
plt.plot(x_23, y_23, linestyle=':', marker='.', color=TWILIGHT10[6], label="P=-23 dBm")
plt.plot(x_25, y_25, linestyle=':', marker='.', color=TWILIGHT10[7], label="P=-25 dBm")
# plt.plot(x_30, y_30, linestyle=':', marker='.', color=TWILIGHT10[8], label="P=-30 dBm")
plt.grid(True)
plt.xlabel("Requested DC Bias (mA)")
plt.ylabel("Chip Resistance ($\Omega$)")
plt.legend()

# Make new plot
plt.figure(4)
plt.plot(x_13b, y_13b, linestyle=':', marker='.', color=TWILIGHT10[1], label="P=-13 dBm")
plt.plot(x_15b, y_15b, linestyle=':', marker='.', color=TWILIGHT10[2], label="P=-15 dBm")
plt.plot(x_17b, y_17b, linestyle=':', marker='.', color=TWILIGHT10[3], label="P=-17 dBm")
plt.plot(x_19b, y_19b, linestyle=':', marker='.', color=TWILIGHT10[4], label="P=-19 dBm")
plt.plot(x_21b, y_21b, linestyle=':', marker='.', color=TWILIGHT10[5], label="P=-21 dBm")
plt.plot(x_23b, y_23b, linestyle=':', marker='.', color=TWILIGHT10[6], label="P=-23 dBm")
plt.plot(x_25b, y_25b, linestyle=':', marker='.', color=TWILIGHT10[7], label="P=-25 dBm")
# plt.plot(x_30, y_30, linestyle=':', marker='.', color=TWILIGHT10[8], label="P=-30 dBm")
plt.grid(True)
plt.xlabel("Requested DC Bias (mA)")
plt.ylabel("Chip Resistance ($\Omega$)")
plt.legend()

COLOR1 = (0, 0.2, 0.5)
COLOR2 = (0, 0.5, 0.2)
# (0.2, 0.2, 0.6)

plt.figure(2, figsize=(8, 4))
line1, = plt.plot(powers, (norm_bias_list+sc_bias_list)/2, linestyle=':', marker='s', color=COLOR2)
plt.fill_between(powers, sc_bias_list, norm_bias_list, color=COLOR2, alpha=0.2, linewidth=0)
plt.xlabel("$P_{RF}$ (dBm)")
plt.ylabel("Critical Current (mA)")
plt.grid(True)

plt.title("$I_{C}$ and $P_{RF}$ Dependence, 3.7 $\mu m$ Device at 7 GHz")

plt.savefig("PS5_fig2.png", dpi=500)

plt.figure(3, figsize=(8, 4))
line1, = plt.plot(powers, (norm_bias_list+sc_bias_list)/2, linestyle=':', marker='s', color=COLOR2)
plt.fill_between(powers, sc_bias_list, norm_bias_list, color=COLOR2, alpha=0.2, linewidth=0)
line2, = plt.plot(powers, (norm_bias_list_b+sc_bias_list_b)/2, linestyle=':', marker='s', color=COLOR1)
plt.fill_between(powers, sc_bias_list_b, norm_bias_list_b, color=COLOR1, alpha=0.2, linewidth=0)
plt.xlabel("$P_{RF}$ (dBm)")
plt.ylabel("Critical Current (mA)")
plt.grid(True)
plt.ylim([0, 1.7])
plt.title("$I_{C}$ and $P_{RF}$ Dependence at 7 GHz")
plt.legend([line1, line2], ["3.7 $\mu m$ Device", "4.4 $\mu m$ Device"])

plt.savefig("PS5_fig3.png", dpi=500)

plt.show()