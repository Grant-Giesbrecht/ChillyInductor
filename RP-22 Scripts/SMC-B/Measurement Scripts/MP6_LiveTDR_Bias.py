''' The purpose of this script is to measure the harmonic generation capacity of the C2024Q1 chips.
'''

from heimdallr.all import *
from chillyinductor.rp22_helper import *
import numpy as np
import os
from inputimeout import inputimeout, TimeoutOccurred
from ganymede import *
import datetime
from pathlib import Path
import argparse
import matplotlib.pyplot as plt

# Set directories for data and sweep configuration
DATA_DIRECTORY = "data"
LOG_DIRECTORY = "logs"

##======================================================
# Read user arguments

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--detail', help="Show detailed log messages.", action='store_true')
parser.add_argument('--loglevel', help="Set the logging display level.", choices=['LOWDEBUG', 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], type=str.upper)
args = parser.parse_args()

# Initialize log
log = LogPile()
if args.loglevel is not None:
	log.set_terminal_level(args.loglevel)
else:
	log.set_terminal_level("DEBUG")
log.str_format.show_detail = args.detail

##======================================================
# Initialize instruments

csa = TektronixCSA8000("GPIB0::3::INSTR", log)
if csa.online:
	log.info("TDR >ONLINE<")
else:
	log.critical("Failed to connect to TDR!")
	exit()
	
dmm = Keysight34400("GPIB0::28::INSTR", log)
if dmm.online:
	log.info("Multimeter >ONLINE<")
else:
	log.critical("Failed to connect to multimeter!")
	exit()
	
tempctrl = LakeShoreModel335("GPIB::12::INSTR", log)
if tempctrl.online:
	log.info("Temperature controller >ONLINE<")
else:
	log.critical("Failed to connect to temperature controller!")
	exit()

##======================================================
# Calibrate current-sense resistor

temp_meas = tempctrl.get_temp()
cs_time = str(datetime.datetime.now())
log.debug(f"Measured temperature at start of current-sense resistor calibration: {temp_meas} K", detail=f"Timestamp = {cs_time}")

# Wait for user to connect through
print(markdown(f"Measuring impedance of current sense resistor. Leave the multimeter connected to the current sense resistor.\n\n>Please disconnect lines from CS resistor to bias tees.<"))
a = input("Press enter when ready:")

# Prepare DMM to measure resistance
dmm.set_measurement(DigitalMultimeterCtg1.MEAS_RESISTANCE_2WIRE, DigitalMultimeterCtg1.RANGE_AUTO)

# Measure resistance
cs_res = dmm.trigger_and_read()
cs_time = datetime.datetime.now()

log.info(f">Current-sense resistor impedance:< Measured R = {cs_res} ohms.")

temp_meas = tempctrl.get_temp()
log.debug(f"Measured temperature at end of current-sense resistor calibration: {temp_meas} K", detail=f"Timestamp = {str(cs_time)}")

# Have the script read itself (to put in output file)
try:
	self_contents = Path(__file__).read_text()
except Exception as e:
	self_contents = f"ERROR: Failed to read script source. ({e})"

# Define dataset dictionary
dataset = {"calibration":{"current_sense_res_ohms": cs_res, "time_of_meas": str(cs_time)}, "dataset":{"tdr_name":[], "TDR_x": [], "TDR_y":[], "TDR_x_unit":[], "TDR_y_unit":[], "V_dc_read":[], "I_dc_mA":[], "temp_K":[]}}



##=============================
# Main sweep

dmm.set_measurement(DigitalMultimeterCtg1.MEAS_VOLT_DC, DigitalMultimeterCtg1.RANGE_AUTO)

while True:
	
	# Get user input
	usr_input = input(f"{Fore.YELLOW}Name to Save{Style.RESET_ALL}> ")
	
	# Exit command
	if usr_input.upper() == "EXIT":
		log.info("Exiting.")
		break
	else:
		
		# Get user input for applied voltage
		read_voltage = True
		while read_voltage:
			applied_DC_voltage_str = input(f"{Fore.YELLOW}\tApplied DC voltage: {Style.RESET_ALL}> ")
			try:
				applied_DC_voltage = float(applied_DC_voltage_str)
				read_voltage = False
			except:
				read_voltage = True
		
		# Measure current temperature
		temp_meas = tempctrl.get_temp()
		
		# Measure voltage/current
		v_cs_read = np.abs(dmm.trigger_and_read())
		
		# Record trace
		wav = csa.get_waveform(4)
		log.debug(f"Measured waveform.")
		
		print(wav['x'])
		print(wav['y'])
		plt.plot(wav['x'], wav['y'])
		plt.show()
		
		words = usr_input.split('/')
		if 'rho' in words or 'gamma' in words:
			wav['y_units'] = 'reflection coef'
		
		dataset['dataset']['tdr_name'].append(usr_input)
		dataset['dataset']['TDR_x'].append(wav['x'])
		dataset['dataset']['TDR_y'].append(wav['y'])
		dataset['dataset']['TDR_x_unit'].append(wav['x_units'])
		dataset['dataset']['TDR_y_unit'].append(wav['y_units'])
		dataset['dataset']['V_dc_read'].append(v_cs_read)
		dataset['dataset']['I_dc_mA'].append(v_cs_read/cs_res*1e3)
		dataset['dataset']['temp_K'].append(temp_meas)
		
		
		log.info(f"Saving TDR waveform with name: {usr_input}")
		
		# # Walk through levels and create/access dictionaries
		# curr_dict = dataset
		# for lvl in words:
			
		# 	# Check if final level dictionary - save waveform dict
		# 	if lvl == words[-1]:
		# 		curr_dict[lvl] = wav
		# 	else:
				
		# 		# If level hasn't been made yet, create a new dictionary
		# 		if lvl not in curr_dict:
		# 			curr_dict[lvl] = {}
				
		# 		# Update pointer
		# 		curr_dict = curr_dict[lvl]


# Get operator notes
sweep_name = input("Sweep name:")
op_notes = input("Operator notes:")
avg_points = input("Averaging points number: ")
hpoints = input("Number of horizontal points: ")

dataset['info'] = {"operator_notes": op_notes, "source_script": "MP4_Live_TDR.py", 'averaging_points':avg_points, 'horiz_points':hpoints}

print(f"Saving data...")

# Save data and log
log.save_hdf(os.path.join(LOG_DIRECTORY, f"{sweep_name}.log.hdf"))

dict_to_hdf(dataset, os.path.join(DATA_DIRECTORY, f"{sweep_name}.hdf"))

print(f" -> Data saved. Exiting.")