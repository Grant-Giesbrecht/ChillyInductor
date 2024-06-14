''' This script is designed to plot S-parameter data from a VNA.
'''

import pandas as pd
import matplotlib.pyplot as plt
from chillyinductor.rp22_helper import *
import sys
from colorama import Fore, Style
import os
import numpy as np
import argparse

#------------------------------------------------------------
# Arg Parse

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--save', help="Save the CSV data as an HDF", action='store_true')
args = parser.parse_args()

#------------------------------------------------------------
# Functions

def lin2dB(x):
	return 20*np.log10(x)

def translate_df(df_in):
	
	comp_S11 = df_in["re:Trc1_S11"] + 1.0j*df_in["im:Trc1_S11"]
	comp_S22 = df_in["re:Trc2_S22"] + 1.0j*df_in["im:Trc2_S22"]
	comp_S21 = df_in["re:Trc3_S21"] + 1.0j*df_in["im:Trc3_S21"]
	comp_S12 = df_in["re:Trc4_S12"] + 1.0j*df_in["im:Trc4_S12"]
	
	# Get magnitude (dB)
	mag_S11 = lin2dB(np.abs(comp_S11))
	mag_S22 = lin2dB(np.abs(comp_S22))
	mag_S21 = lin2dB(np.abs(comp_S21))
	mag_S12 = lin2dB(np.abs(comp_S12))
	
	# Get angle (radians)
	arg_S11 = np.angle(comp_S11)
	arg_S22 = np.angle(comp_S22)
	arg_S21 = np.angle(comp_S21)
	arg_S12 = np.angle(comp_S12)
	
	nd = {"freq_Hz": df_in["freq[Hz]"], "S11_dB":mag_S11, "S22_dB":mag_S22, "S21_dB":mag_S21, "S12_dB":mag_S12, "S11_rad":arg_S11, "S22_rad":arg_S22, "S21_rad":arg_S21, "S12_rad":arg_S12 }
	
	new_df = pd.DataFrame(data=nd)
	
	return new_df

#------------------------------------------------------------
# Import Data

# Find data dir
datapath = get_datadir_path(rp=22, smc='A')
if datapath is None:
	print(f"{Fore.RED}Failed to find data location{Style.RESET_ALL}")
	sys.exit()
else:
	print(f"{Fore.GREEN}Located data directory at: {Fore.LIGHTBLACK_EX}{datapath}{Style.RESET_ALL}")

raw_filename_through = "OutputCal_Through_12June2024.csv"
raw_filename_cbt = "OutputCal_CableBiasTee_12June2024.csv"

save_name = "OutputCal_12June2024.hdf"

# Get filenames
analysis_file_thru = os.path.join(datapath, raw_filename_through)
analysis_file_cbt = os.path.join(datapath, raw_filename_cbt)
save_file_path = os.path.join(datapath, save_name)

# Make dataframes
# NOTE: This function only works if you delete the '# VERSION 1.0' header made by the VNA
df_thru = pd.read_csv(analysis_file_thru, engine='python', sep=';')
df_cbt = pd.read_csv(analysis_file_cbt, engine='python', sep=';')

#------------------------------------------------------------
# Format data nicely

# Remove unnamed column
df_thru = df_thru.drop("Unnamed: 9", axis=1)
df_cbt = df_cbt.drop("Unnamed: 9", axis=1)

# # Rename columns
# col_map = {"freq[Hz]":"freq_Hz"}
# df_thru.rename(columns=col_map)
# df_cbt.rename(columns=col_map)

# Convert format
df_thru = translate_df(df_thru)
df_cbt = translate_df(df_cbt)

mksz = 6

# for stl in plt.style.available:
# 	plt.style.use(stl)
# 	print(f"Using style: {Fore.YELLOW}{stl}{Style.RESET_ALL} ")
	
# plt.style.use("C:\\Users\\grant\\Documents\\GitHub\\ChillyInductor\\src\\custom.mplstyle.py")

plt.figure(1)
plt.subplot(1, 2, 1)
plt.plot(df_thru.freq_Hz/1e9, df_thru.S11_dB, linestyle=':', color="#fcbe03", marker='.', markersize=mksz, label="S11")
plt.plot(df_thru.freq_Hz/1e9, df_thru.S22_dB, linestyle=':', color="#eb0f0c", marker='.', markersize=mksz, label="S22")
plt.plot(df_thru.freq_Hz/1e9, df_thru.S21_dB, linestyle=':', color="#03bafc", marker='.', markersize=mksz, label="S21")
plt.plot(df_thru.freq_Hz/1e9, df_thru.S12_dB, linestyle=':', color="#03a32b", marker='.', markersize=mksz, label="S12")

plt.xlabel("Frequency (GHz)")
plt.ylabel("(dB)")
plt.legend()
plt.title("Through Measurement")

plt.grid(True)

plt.figure(1)
plt.subplot(1, 2, 2)
plt.plot(df_cbt.freq_Hz/1e9, df_cbt.S11_dB, linestyle=':', color="#fcbe03", marker='.', markersize=mksz, label="S11")
plt.plot(df_cbt.freq_Hz/1e9, df_cbt.S22_dB, linestyle=':', color="#eb0f0c", marker='.', markersize=mksz, label="S22")
plt.plot(df_cbt.freq_Hz/1e9, df_cbt.S21_dB, linestyle=':', color="#03bafc", marker='.', markersize=mksz, label="S21")
plt.plot(df_cbt.freq_Hz/1e9, df_cbt.S12_dB, linestyle=':', color="#03a32b", marker='.', markersize=mksz, label="S12")

plt.xlabel("Frequency (GHz)")
plt.ylabel("(dB)")
plt.legend()
plt.title("Output Network Measurement")

plt.grid(True)
plt.show()

#------------------------------------------------------------
# Save data if requested

if args.save:
	
	desc_str = f"This file was created from {raw_filename_cbt} and {raw_filename_through}, which were processed\n"
	desc_str += f"and re-saved. through_cal contains the S-parameter data for when the calibrated VNA was connected\n"
	desc_str += f"to the through-calibration standard. This was used a baseline to show the quality of the calibration.\n"
	desc_str += f"output_cal contains the s-parameter data for when the calibrated VNA was connected to the cable\n"
	desc_str += f"(which usually connects the cryostat to the bias tee) and the bias tee on the spectrum analyzer."
	
	# orient=list tells Pandas to save the data as lists, not dictionaries mapping index to value
	root_dict = {'through_cal': df_thru.to_dict(orient='list'), 'output_cal':df_thru.to_dict(orient='list'), 'info':{'description':desc_str}}
	
	save_hdf(root_dict, save_file_path)

