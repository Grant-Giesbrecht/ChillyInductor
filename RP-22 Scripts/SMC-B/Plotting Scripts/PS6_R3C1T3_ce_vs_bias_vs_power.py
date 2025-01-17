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

folder_name = os.path.join("ce_vs_bias_vs_power_R3C1T3", "05Oct2024_autosave")

graf_13.load_hdf(os.path.join(".", folder_name, "P-13dBm_ce2.graf"))
x_13 = graf_13.axes['Ax0'].traces['Tr0'].x_data[1:]
y_13 = graf_13.axes['Ax0'].traces['Tr0'].y_data[1:]
y_data_lists.append(y_13)
x_data_lists.append(x_13)

graf_13.load_hdf(os.path.join(".", folder_name, "P-15dBm_ce2.graf"))
x_15 = graf_13.axes['Ax0'].traces['Tr0'].x_data[1:]
y_15 = graf_13.axes['Ax0'].traces['Tr0'].y_data[1:]
y_data_lists.append(y_15)
x_data_lists.append(x_15)

graf_13.load_hdf(os.path.join(".", folder_name, "P-17dBm_ce2.graf"))
x_17 = graf_13.axes['Ax0'].traces['Tr0'].x_data[1:]
y_17 = graf_13.axes['Ax0'].traces['Tr0'].y_data[1:]
y_data_lists.append(y_17)
x_data_lists.append(x_17)

graf_13.load_hdf(os.path.join(".", folder_name, "P-19dBm_ce2.graf"))
x_19 = graf_13.axes['Ax0'].traces['Tr0'].x_data[1:]
y_19 = graf_13.axes['Ax0'].traces['Tr0'].y_data[1:]
y_data_lists.append(y_19)
x_data_lists.append(x_19)

graf_13.load_hdf(os.path.join(".", folder_name, "P-21dBm_ce2.graf"))
x_21 = graf_13.axes['Ax0'].traces['Tr0'].x_data[1:]
y_21 = graf_13.axes['Ax0'].traces['Tr0'].y_data[1:]
y_data_lists.append(y_21)
x_data_lists.append(x_21)

graf_13.load_hdf(os.path.join(".", folder_name, "P-23dBm_ce2.graf"))
x_23 = graf_13.axes['Ax0'].traces['Tr0'].x_data[1:]
y_23 = graf_13.axes['Ax0'].traces['Tr0'].y_data[1:]
y_data_lists.append(y_23)
x_data_lists.append(x_23)

graf_13.load_hdf(os.path.join(".", folder_name, "P-25dBm_ce2.graf"))
x_25 = graf_13.axes['Ax0'].traces['Tr0'].x_data[1:]
y_25 = graf_13.axes['Ax0'].traces['Tr0'].y_data[1:]
y_data_lists.append(y_25)
x_data_lists.append(x_25)

graf_13.load_hdf(os.path.join(".", folder_name, "P-30dBm_ce2.graf"))
x_30 = graf_13.axes['Ax0'].traces['Tr0'].x_data[1:]
y_30 = graf_13.axes['Ax0'].traces['Tr0'].y_data[1:]
y_data_lists.append(y_30)
x_data_lists.append(x_30)

folder_name_b = os.path.join("ce_vs_bias_vs_power_R3C1T3", "05Oct2024_autosave")

graf_13.load_hdf(os.path.join(".", folder_name_b, "P-13dBm_ce3.graf"))
x_13b = graf_13.axes['Ax0'].traces['Tr0'].x_data[1:]
y_13b = graf_13.axes['Ax0'].traces['Tr0'].y_data[1:]
y_data_lists_b.append(y_13b)
x_data_lists_b.append(x_13b)

graf_13.load_hdf(os.path.join(".", folder_name_b, "P-15dBm_ce3.graf"))
x_15b = graf_13.axes['Ax0'].traces['Tr0'].x_data[1:]
y_15b = graf_13.axes['Ax0'].traces['Tr0'].y_data[1:]
y_data_lists_b.append(y_15b)
x_data_lists_b.append(x_15b)

graf_13.load_hdf(os.path.join(".", folder_name_b, "P-17dBm_ce3.graf"))
x_17b = graf_13.axes['Ax0'].traces['Tr0'].x_data[1:]
y_17b = graf_13.axes['Ax0'].traces['Tr0'].y_data[1:]
y_data_lists_b.append(y_17b)
x_data_lists_b.append(x_17b)

graf_13.load_hdf(os.path.join(".", folder_name_b, "P-19dBm_ce3.graf"))
x_19b = graf_13.axes['Ax0'].traces['Tr0'].x_data[1:]
y_19b = graf_13.axes['Ax0'].traces['Tr0'].y_data[1:]
y_data_lists_b.append(y_19b)
x_data_lists_b.append(x_19b)

graf_13.load_hdf(os.path.join(".", folder_name_b, "P-21dBm_ce3.graf"))
x_21b = graf_13.axes['Ax0'].traces['Tr0'].x_data[1:]
y_21b = graf_13.axes['Ax0'].traces['Tr0'].y_data[1:]
y_data_lists_b.append(y_21b)
x_data_lists_b.append(x_21b)

graf_13.load_hdf(os.path.join(".", folder_name_b, "P-23dBm_ce3.graf"))
x_23b = graf_13.axes['Ax0'].traces['Tr0'].x_data[1:]
y_23b = graf_13.axes['Ax0'].traces['Tr0'].y_data[1:]
y_data_lists_b.append(y_23b)
x_data_lists_b.append(x_23b)

graf_13.load_hdf(os.path.join(".", folder_name_b, "P-25dBm_ce3.graf"))
x_25b = graf_13.axes['Ax0'].traces['Tr0'].x_data[1:]
y_25b = graf_13.axes['Ax0'].traces['Tr0'].y_data[1:]
y_data_lists_b.append(y_25b)
x_data_lists_b.append(x_25b)

graf_13.load_hdf(os.path.join(".", folder_name_b, "P-30dBm_ce3.graf"))
x_30b = graf_13.axes['Ax0'].traces['Tr0'].x_data[1:]
y_30b = graf_13.axes['Ax0'].traces['Tr0'].y_data[1:]
y_data_lists_b.append(y_30b)
x_data_lists_b.append(x_30b)

# sc_bias_list = []
# norm_bias_list = []
# THRESHOLD = 10
# for xidx, ydl in enumerate(y_data_lists):
	
# 	ydl = np.array(ydl)
	
# 	idx_norm = np.where(ydl > THRESHOLD)
# 	sc_bias_list.append(x_data_lists[xidx][idx_norm[0][0]-1])
# 	norm_bias_list.append(x_data_lists[xidx][idx_norm[0][0]])
# sc_bias_list = np.array(sc_bias_list)
# norm_bias_list = np.array(norm_bias_list)

# sc_bias_list_b = []
# norm_bias_list_b = []
# THRESHOLD = 20
# for xidx, ydl in enumerate(y_data_lists_b):
	
# 	ydl = np.array(ydl)
	
# 	idx_norm = np.where(ydl > THRESHOLD)
# 	sc_bias_list_b.append(x_data_lists_b[xidx][idx_norm[0][0]-1])
# 	norm_bias_list_b.append(x_data_lists_b[xidx][idx_norm[0][0]])
	
# 	print(idx_norm)
	
# sc_bias_list_b = np.array(sc_bias_list_b)
# norm_bias_list_b = np.array(norm_bias_list_b)

# Make new plot
plt.figure(1, figsize=(5.5, 5))
plt.plot(x_13[:-6], y_13[:-6], linestyle=':', marker='.', color=TWILIGHT10[1], label="P=-13 dBm")
plt.plot(x_15[:-2], y_15[:-2], linestyle=':', marker='.', color=TWILIGHT10[2], label="P=-15 dBm")
plt.plot(x_17[:-2], y_17[:-2], linestyle=':', marker='.', color=TWILIGHT10[3], label="P=-17 dBm")
plt.plot(x_19[:-2], y_19[:-2], linestyle=':', marker='.', color=TWILIGHT10[4], label="P=-19 dBm")
plt.plot(x_21[:-2], y_21[:-2], linestyle=':', marker='.', color=TWILIGHT10[5], label="P=-21 dBm")
plt.plot(x_23[:-2], y_23[:-2], linestyle=':', marker='.', color=TWILIGHT10[6], label="P=-23 dBm")
plt.plot(x_25[:-2], y_25[:-2], linestyle=':', marker='.', color=TWILIGHT10[7], label="P=-25 dBm")
# plt.plot(x_30, y_30, linestyle=':', marker='.', color=TWILIGHT10[8], label="P=-30 dBm")
plt.grid(True)
plt.xlabel("DC Bias (mA)")
plt.ylabel("2nd Harmonic Conversion Efficiency (%)")
plt.legend()
plt.title("3.7 $\mu m$ Device, 2 $\cdot f_{RF}$ = 5 GHz, L=454 mm")
plt.tight_layout()

plt.savefig("PS6_fig1_tall.png", dpi=500)

# plt.figure(2, figsize=(10, 4))
plt.figure(2, figsize=(5.5, 5))
plt.plot(x_13b[:-6], y_13b[:-6], linestyle=':', marker='.', color=TWILIGHT10[1], label="P=-13 dBm")
plt.plot(x_15b[:-2], y_15b[:-2], linestyle=':', marker='.', color=TWILIGHT10[2], label="P=-15 dBm")
plt.plot(x_17b[:-2], y_17b[:-2], linestyle=':', marker='.', color=TWILIGHT10[3], label="P=-17 dBm")
plt.plot(x_19b[:-2], y_19b[:-2], linestyle=':', marker='.', color=TWILIGHT10[4], label="P=-19 dBm")
plt.plot(x_21b[:-2], y_21b[:-2], linestyle=':', marker='.', color=TWILIGHT10[5], label="P=-21 dBm")
plt.plot(x_23b[:-2], y_23b[:-2], linestyle=':', marker='.', color=TWILIGHT10[6], label="P=-23 dBm")
plt.plot(x_25b[:-2], y_25b[:-2], linestyle=':', marker='.', color=TWILIGHT10[7], label="P=-25 dBm")
# plt.plot(x_30, y_30, linestyle=':', marker='.', color=TWILIGHT10[8], label="P=-30 dBm")
plt.grid(True)
plt.xlabel("DC Bias (mA)")
plt.ylabel("3rd Harmonic Conversion Efficiency (%)")
plt.legend()
plt.title("3.7 $\mu m$ Device, 3 $\cdot f_{RF}$ = 7.5 GHz, L=454 mm")
plt.tight_layout()

plt.savefig("PS6_fig2_tall.png", dpi=500)

plt.show()

