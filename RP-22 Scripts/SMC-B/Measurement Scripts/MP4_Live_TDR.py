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
	log.set_terminal_level(DEBUG)
log.str_format.show_detail = args.detail

##======================================================
# Initialize instruments

csa = TektronixCSA8000("GPIB0::3::INSTR", log)
if csa.online:
	log.info("TDR >ONLINE<")
else:
	log.critical("Failed to connect to TDR!")
	exit()

dataset = {}
while True:
	
	# Get user input
	usr_input = input(f"{Fore.YELLOW}Name to Save{Style.RESET_ALL}> ")
	
	# Exit command
	if usr_input.upper() == "EXIT":
		log.info("Exiting.")
		break
	else:
		
		# Record trace
		wav = csa.get_waveform(4)
		log.debug(f"Measured waveform.")
		
		# Get path
		words = usr_input.split('/')
		words.insert(0, "dataset")
		
		log.info(f"Saving TDR waveform to location: {usr_input}")
		
		# Walk through levels and create/access dictionaries
		curr_dict = dataset
		for lvl in words:
			
			# Check if final level dictionary - save waveform dict
			if lvl == words[-1]:
				curr_dict[lvl] = wav
			else:
				
				# If level hasn't been made yet, create a new dictionary
				if lvl not in curr_dict:
					curr_dict[lvl] = {}
				
				# Update pointer
				curr_dict = curr_dict[lvl]


# Get operator notes
sweep_name = input("Sweep name:")
op_notes = input("Operator notes:")


dataset['info'] = {"operator_notes": op_notes, "source_script": "MP4_Live_TDR.py"}

print(f"Saving data...")

# Save data and log
log.save_hdf(os.path.join(LOG_DIRECTORY, f"{sweep_name}.log.hdf"))

dict_to_hdf(dataset, os.path.join(DATA_DIRECTORY, f"{sweep_name}.hdf"))

print(f" -> Data saved. Exiting.")