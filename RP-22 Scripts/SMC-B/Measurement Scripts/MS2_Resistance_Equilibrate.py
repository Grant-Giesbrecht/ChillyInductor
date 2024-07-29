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

DATA_DIRECTORY = "data"
LOG_DIRECTORY = "logs"

LOGGING_PERIOD_S = 10

TIME_AUTOSAVE_S = 600

DELTA_THRESHOLD_T = 0.01
DELTA_THRESHOLD_R = 10

sweep_name = input("Sweep name: ")

##======================================================
# Initialize instruments

log = LogPile()

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

# Measure resistance
short_res = dmm.trigger_and_read()
short_temp = tempctrl.get_temp()
sc_time = datetime.datetime.now()

log.info(f">Short-Circuit Impedance:< Measured R = {short_res} ohms and T = {short_temp} K.")

# Define dataset dictionary
dataset = {"calibration":{"short_circ_temp_K": short_temp, "short_circ_res_ohm":short_res, "short_circ_time": str(sc_time)}, "dataset":{"temperatures":[], "resistances":[], "times":[], "temp_set_points_K":[]}}

##======================================================
# Begin logging data

# Wait for user to connect through
print(f"Ready to perform primary logging.")
a = input("Press enter when ready:")

# Enable temperature controller output
tempctrl.set_setpoint(3) # Set a low target temperature
tempctrl.set_range(LakeShoreModel335.RANGE_MID)

# Prepare DMM to measure resistance
dmm.set_measurement(DigitalMultimeterCtg1.MEAS_RESISTANCE_2WIRE, DigitalMultimeterCtg1.RANGE_AUTO)

time_last_save = time.time()

# List of temperature set points
# temp_sp_list = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 12.5, 11.5, 10.5, 9.5, 8.5, 7.5, 6.5, 5.5, 4.5, 3.5]
temp_sp_list = [3, 4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5]

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
			
			# Get resistance averages
			old_res = np.mean(dataset['dataset']['resistances'][-19:-15])
			new_res = np.mean(dataset['dataset']['resistances'][-4:-1])
			res_delta = np.abs(new_res - old_res)
			
			# Check for sufficient change
			if (temp_delta < DELTA_THRESHOLD_T) and (res_delta < DELTA_THRESHOLD_R):
				temp_idx += 1
				log.info(f"Achieved equilibrium condition. delta T ({temp_delta} K) < threshold ({DELTA_THRESHOLD_T} K) and delta R ({res_delta} Ohms) < threshold ({DELTA_THRESHOLD_R} Ohms).", detail=f"Old average = {old_temp} K and {old_res} Ohms, new average = {new_temp} K and {new_res} Ohms.")
				

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
				log.debug(f"Equilibrium condition has not been met. delta T = {temp_delta} K, delta R = {res_delta} Ohms.", detail=f"Old average = {old_temp} K, new average = {new_temp} K.")
	
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