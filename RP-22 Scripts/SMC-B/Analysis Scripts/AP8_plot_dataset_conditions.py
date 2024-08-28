from ganymede import * 
from pylogfile.base import *
import sys
from chillyinductor.rp22_helper import *
import matplotlib.pyplot as plt
import json
from heimdallr.base import interpret_range

#------------------------------------------------------------
# Create list of files to analyze

import matplotlib.pyplot as plt
import numpy as np

log = LogPile()

# file_list = []
# colors = []
# markers = []
# labels = []
# marker_sizes = []

# datapath = get_datadir_path(rp=22, smc='B', sub_dirs=['*R4C4*C', 'Track 1 4mm'])
# if datapath is None:
# 	print(f"{Fore.RED}Failed to find data location{Style.RESET_ALL}")
# 	sys.exit()
# else:
# 	print(f"{Fore.GREEN}Located data directory at: {Fore.LIGHTBLACK_EX}{datapath}{Style.RESET_ALL}")
# filename = "RP22B_MP3_t2_8Aug2024_R4C4T1_r1.hdf"
# file_list.append(os.path.join(datapath, filename))
# markers.append('s')
# colors.append((0, 0, 0.7))
# labels.append(filename)
# marker_sizes.append(25)

# filename = "RP22B_MP3_t1_1Aug2024_R4C4T1_r1.hdf"
# file_list.append(os.path.join(datapath, filename))
# markers.append('s')
# colors.append((0, 0.5, 0))
# labels.append(filename)
# marker_sizes.append(25)

# # filename = "RP22B_MP3_t1_31July2024_R4C4T1_r1_autosave.hdf"
# # file_list.append(os.path.join(datapath, filename))

# datapath = get_datadir_path(rp=22, smc='B', sub_dirs=['*R4C4*C', 'Track 2 43mm'])
# if datapath is None:
# 	print(f"{Fore.RED}Failed to find data location{Style.RESET_ALL}")
# 	sys.exit()
# else:
# 	print(f"{Fore.GREEN}Located data directory at: {Fore.LIGHTBLACK_EX}{datapath}{Style.RESET_ALL}")
# filename = "RP22B_MP3a_t3_19Aug2024_R4C4T2_r1.hdf"
# file_list.append(os.path.join(datapath, filename))
# markers.append('s')
# colors.append((0.7, 0, 0))
# labels.append(filename)
# marker_sizes.append(25)


#------ Set 2

file_list = []
colors = []
markers = []
labels = []
marker_sizes = []

datapath = get_datadir_path(rp=22, smc='B', sub_dirs=['*R4C4*C', 'Track 2 43mm'])
if datapath is None:
	print(f"{Fore.RED}Failed to find data location{Style.RESET_ALL}")
	sys.exit()
else:
	print(f"{Fore.GREEN}Located data directory at: {Fore.LIGHTBLACK_EX}{datapath}{Style.RESET_ALL}")

filename = "RP22B_MP3a_t3_19Aug2024_R4C4T2_r1.hdf"
file_list.append(os.path.join(datapath, filename))
markers.append('s')
colors.append((0, 0, 0.7))
labels.append(filename)
marker_sizes.append(25)

filename = "RP22B_MP3a_t2_20Aug2024_R4C4T2_r1.hdf"
file_list.append(os.path.join(datapath, filename))
markers.append('s')
colors.append((0, 0.5, 0))
labels.append(filename)
marker_sizes.append(25)

filename = "RP22B_MP3a_t4_26Aug2024_R4C4T2_r1.hdf"
file_list.append(os.path.join(datapath, filename))
markers.append('s')
colors.append((0.7, 0, 0))
labels.append(filename)
marker_sizes.append(25)

##--------------------------------------------
# Read HDF5 Files

conf_list = []

num_axes = 0
for analysis_file in file_list:
	
	log.info(f"Loading file '>{analysis_file}<' contents into memory")
	
	data = hdf_to_dict(analysis_file)
	
	try:
		conf = json.loads(data['info']['configuration'].decode())
	except:
		sys.exit()
	
	VERT_SPACING = 1
	
	# Count number of axes
	local_num_axes = 0
	for k in conf.keys():
		
		# Get values
		try:
			vals = interpret_range(conf[k])
		except:
			continue
		local_num_axes += 1
	
	num_axes = np.max([num_axes, local_num_axes])
	
	conf_list.append(conf)

## ---------------------------------
# Make plot

# Plot each thing
marker_size = 10

fig, axs = plt.subplots(num_axes, 1, figsize=(13, 7))

# Scan over files
for src_idx, conf in enumerate(conf_list):
	
	# Scan over parameters
	data_idx = 0
	for k in conf.keys():
		
		# Get values
		try:
			vals = interpret_range(conf[k])
		except:
			continue
		
		unit_str = conf[k]['unit']
		
		# Plot data
		axs[data_idx].grid(True)
		axs[data_idx].scatter(vals, [len(conf_list)-src_idx]*len(vals), [marker_sizes[src_idx]]*len(vals), marker=markers[src_idx], color=colors[src_idx], label=labels[src_idx])
		axs[data_idx].set_xlabel(f"{k} [{unit_str}]")
		
		# Set parameters on last loops
		if src_idx == len(conf_list) -1:
			# axs[data_idx].legend()
			axs[data_idx].set_yticks(list(range(1, len(conf_list)+1)))
			axs[data_idx].set_yticklabels(reversed(labels))
		
		data_idx += 1

fig.tight_layout()
plt.show()