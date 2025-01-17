'''
Let system equilibrate at numerous temps to get better temp vs R measurement.

28-June-2024
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

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--detail', help="Show detailed log messages.", action='store_true')
parser.add_argument('--loglevel', help="Set the logging display level.", choices=['LOWDEBUG', 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], type=str.upper)
args = parser.parse_args()

# Set directories for data and sweep configuration
CONF_DIRECTORY = "sweep_configs"
DATA_DIRECTORY = "data"
LOG_DIRECTORY = "logs"

LOGGING_PERIOD_S = 10

TIME_AUTOSAVE_S = 600

# DELTA_THRESHOLD_T = 0.01
# DELTA_THRESHOLD_R = 10

sweep_name = input("Sweep name: ")


##======================================================
# Get configuration

conf_file_prefix = input(f"{Fore.YELLOW}Configuration file name: (No extension or folder){Style.RESET_ALL}")
conf_file_name = os.path.join(".", CONF_DIRECTORY, f"{conf_file_prefix}.json")

# Load configuration data
try:
	with open(conf_file_name, "r") as outfile:
		conf_data = json.load(outfile)
except Exception as e:
	log.critical(f"Failed to read configuration file >{conf_file_name}<. ({e})")
	exit()

# Interpret data
try:
	temp_sp_list = interpret_range(conf_data['temp_sp_list_K'], print_err=True)
	DELTA_THRESHOLD_R = float(conf_data['delta_threshold_R_ohms'])
	DELTA_THRESHOLD_T = float(conf_data['delta_threshold_T_K'])
	DELTA_THRESHOLD_TARG = float(conf_data['delta_threshold_target_K'])
except:
	log.critical(f"Failed to interpret sweep configuration. Exiting.")
	exit()

log = LogPile()
if args.loglevel is not None:
	log.set_terminal_level(args.loglevel)
else:
	log.set_terminal_level("DEBUG")
log.str_format.show_detail = args.detail
log.info(f"Detected {len(temp_sp_list)} measurement points.", detail=f"Measurement points = {temp_sp_list} [K]")

##======================================================
# Initialize instruments

operator_notes = input("Operator notes: ")

dmm = Keysight34400("GPIB0::28::INSTR", log)
tempctrl = LakeShoreModel335("GPIB::12::INSTR", log)

if dmm.online:
	log.info("Multimeter >ONLINE<")
else:
	log.critical("Failed to connect to multimeter!")
	exit()
if tempctrl.online:
	log.info("Temperature controller >ONLINE<")
else:
	log.critical("Failed to connect to temperature controller!")
	exit()

##======================================================
# Calibrate impedance of system

# Wait for user to connect through
print(f"Measuring impedance of short circuit. Please connect a through.")
a = input("Press enter when ready:")

# Prepare DMM to measure resistance
dmm.set_measurement(DigitalMultimeterCtg1.MEAS_RESISTANCE_2WIRE, DigitalMultimeterCtg1.RANGE_AUTO)
dmm.set_low_power_mode(True)

# Measure resistance
short_res = dmm.trigger_and_read()
short_temp = tempctrl.get_temp()
sc_time = datetime.datetime.now()

log.info(f">Short-Circuit Impedance:< Measured R = {short_res} ohms and T = {short_temp} K.")

# HAve the script read itself (to put in output file)
try:
	self_contents = Path(__file__).read_text()
except Exception as e:
	self_contents = f"ERROR: Failed to read script source. ({e})"

# Define dataset dictionary
dataset = {"calibration":{"short_circ_temp_K": short_temp, "short_circ_res_ohm":short_res, "short_circ_time": str(sc_time)}, "dataset":{"temperatures":[], "resistances":[], "times":[], "temp_set_points_K":[]}, 'info': {'source_script': __file__, 'operator_notes':operator_notes, 'configuration': json.dumps(conf_data), 'source_script_full':self_contents}}

##======================================================
# Begin logging data

# Wait for user to connect through
print(f"Ready to perform primary logging.")
a = input("Press enter when ready:")

# Enable temperature controller output
tempctrl.set_setpoint(3) # Set a low target temperature
tempctrl.set_range(LakeShoreModel335.RANGE_MID)

time_last_save = time.time()

# List of temperature set points
# temp_sp_list = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 12.5, 11.5, 10.5, 9.5, 8.5, 7.5, 6.5, 5.5, 4.5, 3.5]

temp_idx = 0

# Select new temperature setpoint
temp_sp = temp_sp_list[temp_idx]
tempctrl.set_setpoint(temp_sp)

holdoff_count = 0

running = True
while running:
	
	log.debug(f"Beginning measurement")
	
	# Measure resistance
	res = dmm.trigger_and_read()
	temp = tempctrl.get_temp()
	log.info(f"Measured values: >R = {res} Ohms<, >T = {temp} K<.")
	
	# Record value
	dataset['dataset']['temp_set_points_K'].append(temp_sp)
	dataset['dataset']['temperatures'].append(temp)
	dataset['dataset']['resistances'].append(res)
	dataset['dataset']['times'].append(str(datetime.datetime.now()))
	
	# Wait for user input - check to quit
	try:
		usr_input = inputimeout(prompt="Enter 'q' to quit: ", timeout=LOGGING_PERIOD_S)
	except TimeoutOccurred:
		usr_input = ''
	
	# Check if the system has equilibrated
	if len(dataset['dataset']['temperatures']) > 21:
		
		# Check if holdoff is set
		if holdoff_count > 0:
			holdoff_count -= 1
		else:
		
			# Get temperature averages
			old_temp = np.mean(dataset['dataset']['temperatures'][-19:-15])
			new_temp = np.mean(dataset['dataset']['temperatures'][-4:-1])
			temp_delta = np.abs(new_temp - old_temp)
			temp_targ_delta = np.abs(new_temp - temp_sp)
			
			# Get resistance averages
			old_res = np.mean(dataset['dataset']['resistances'][-19:-15])
			new_res = np.mean(dataset['dataset']['resistances'][-4:-1])
			res_delta = np.abs(new_res - old_res)
			
			# Check for sufficient change
			if (temp_delta < DELTA_THRESHOLD_T) and (res_delta < DELTA_THRESHOLD_R) and (temp_targ_delta < DELTA_THRESHOLD_TARG):
				temp_idx += 1
				log.info(f"Achieved equilibrium condition. delta T ({temp_delta} K) < threshold ({DELTA_THRESHOLD_T} K), delta R ({res_delta} Ohms) < threshold ({DELTA_THRESHOLD_R} Ohms), target deviation ({temp_targ_delta} K) < threshold ({DELTA_THRESHOLD_TARG} K)", detail=f"Old average = {old_temp} K and {old_res} Ohms, new average = {new_temp} K and {new_res} Ohms.")
				

				# Break if exiting list
				if temp_idx >= len(temp_sp_list):
					running = False
					log.info("Reached end of sweep, ending.")
				else:
					temp_sp = temp_sp_list[temp_idx]
					
					log.info(f"Changed set point to {temp_sp} K, index {temp_idx} of {len(temp_sp_list)}.")
					tempctrl.set_setpoint(temp_sp)
				
				holdoff_count = 20
			
			else:
				log.debug(f"Equilibrium condition has not been met. delta T = {temp_delta} K, delta R = {res_delta} Ohms, target deviation = {temp_targ_delta} K.", detail=f"Old average = {old_temp} K, new average = {new_temp} K.")
	
	# Quit if told
	if usr_input.lower() == "q":
		log.info(f"Received exit signal quitting.")
		break
	
	# Check for autosave
	if time.time() - time_last_save > TIME_AUTOSAVE_S:
		
		# Save data and log
		log.save_hdf(os.path.join(LOG_DIRECTORY, f"{sweep_name}_autosave.log.hdf"))
		
		dict_to_hdf(dataset, os.path.join(DATA_DIRECTORY, f"{sweep_name}_autosave.hdf"))
		log.debug(f"Autosaved logs and data.")
		
		time_last_save = time.time()

log.info("Turning off temperature controller heater.")

# Turn off heater so chip cools back down
tempctrl.set_setpoint(0)
tempctrl.set_range(LakeShoreModel335.RANGE_OFF)

print(f"Saving data...")

# Save data and log
log.save_hdf(os.path.join(LOG_DIRECTORY, f"{sweep_name}.log.hdf"))

dict_to_hdf(dataset, os.path.join(DATA_DIRECTORY, f"{sweep_name}.hdf"))

print(f" -> Data saved. Exiting.")