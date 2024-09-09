'''
Measures temperature and resistance (using a DMM) as a function of time. Used to monitor the impedance of the chip
while cooling down.

24-June-2024
'''

from heimdallr.all import *
from chillyinductor.rp22_helper import *
import numpy as np
import os
from inputimeout import inputimeout, TimeoutOccurred
from ganymede import *
import datetime

DATA_DIRECTORY = "C:\\Users\\gmg3\\Mega\\remote_data\\data"
LOG_DIRECTORY = "C:\\Users\\gmg3\\Mega\\remote_data\\logs"

LOGGING_PERIOD_S = 5

TIME_AUTOSAVE_S = 600

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
dataset = {"calibration":{"short_circ_temp_K": short_temp, "short_circ_res_ohm":short_res, "short_circ_time": str(sc_time)}, "dataset":{"temperatures":[], "resistances":[], "times":[]}}

##======================================================
# Begin logging data

# Wait for user to connect through
print(f"Ready to perform primary logging.")
a = input("Press enter when ready:")

# Prepare DMM to measure resistance
dmm.set_measurement(DigitalMultimeterCtg1.MEAS_RESISTANCE_2WIRE, DigitalMultimeterCtg1.RANGE_AUTO)

time_last_save = time.time()

while True:
	
	log.debug(f"Beginning measurement")
	
	# Measure resistance
	res = dmm.trigger_and_read()
	temp = tempctrl.get_temp()
	log.info(f"Measured values: >R = {res} Ohms<, >T = {temp} K<.")
	
	dataset['dataset']['temperatures'].append(temp)
	dataset['dataset']['resistances'].append(res)
	dataset['dataset']['times'].append(str(datetime.datetime.now()))
	
	# Wait for user input - check to quit
	try:
		usr_input = inputimeout(prompt="Enter 'q' to quit: ", timeout=LOGGING_PERIOD_S)
	except TimeoutOccurred:
		usr_input = ''
	
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
	

print(f"Saving data...")

# Save data and log
log.save_hdf(os.path.join(LOG_DIRECTORY, f"{sweep_name}.log.hdf"))

dict_to_hdf(dataset, os.path.join(DATA_DIRECTORY, f"{sweep_name}.hdf"))

print(f" -> Data saved. Exiting.")