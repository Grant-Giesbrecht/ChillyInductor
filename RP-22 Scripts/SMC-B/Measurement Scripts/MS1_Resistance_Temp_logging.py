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

DATA_DIRECTORY = "data"
LOG_DIRECTORY = "logs"

LOGGING_PERIOD_S = 10

##======================================================
# Initialize instruments

log = LogPile()

dmm = Keysight34400("GPIB0::28::INSTR")
tempctrl = LakeShoreModel335("GPIB::12::INSTR")

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

dataset = []
calset = []

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

log.info(f">Short-Circuit Impedance:< Measured R = {short_res} ohms and T = {short_temp} K.")

##======================================================
# Begin logging data

# Wait for user to connect through
print(f"Ready to perform primary logging.")
a = input("Press enter when ready:")

# Prepare DMM to measure resistance
dmm.set_measurement(DigitalMultimeterCtg1.MEAS_RESISTANCE_2WIRE, DigitalMultimeterCtg1.RANGE_AUTO)

while True:
	
	log.debug(f"Beginning measurement")
	
	# Measure resistance
	res = dmm.trigger_and_read()
	temp = tempctrl.get_temp()
	log.info(f"Measured values: >R = {res} Ohms<, >T = {temp} K<.")
	
	# Wait for user input - check to quit
	try:
		usr_input = inputimeout(prompt="Enter 'q' to quit: ", timeout=10)
	except TimeoutOccurred:
		pass
	
	# Quit if told
	if usr_input.lower() == "q":
		log.info(f"Received exit signal quitting.")
		break

