from heimdallr.all import *
import numpy as np
import os

CONF_DIRECTORY ="sweep_configs"

# Create logger
log = LogPile()

# Get configuration
conf_file_prefix = input(f"Configuration file name: (No extension or folder)")
conf_file_name = os.path.join(".", CONF_DIRECTORY, f"{conf_file_prefix}.json")

# Load configuration data
try:
	with open(conf_file_name, "r") as outfile:
		conf_data = json.load(outfile)
except:
	log.critical(f"Failed to read configuration file >{conf_file_name}<")
	exit()
	
# Verify conf data exists
if conf_data is None:
	log.critical(f"Failed to read configuration file >{conf_file_name}<.")

# Interpret data
try:
	freq_a = interpret_range(conf_data['frequency_RF'])
	freq_b = interpret_range(conf_data['frequency_LO'])
	power_RF_dBm = interpret_range(conf_data['power_RF'])
	power_LO_dBm = interpret_range(conf_data['power_LO'])
	
	rbw = conf_data['spectrum_analyzer']['RBW_Hz']
	f_start = conf_data['spectrum_analyzer']['freq_start_Hz']
	f_end = conf_data['spectrum_analyzer']['freq_end_Hz']
except:
	log.critical(f"Corrupt configuration file >{conf_file_name}<.")
	exit()
	
# Get dataset name
sweep_name = input(f"Dataset name: ")

# Connect to instruments
sg1 = Keysight8360L("GPIB::18::INSTR", log)
sg2 = AgilentE4400("GPIB::19::INSTR", log)
# sa = SiglentSSA3000X("TCPIP0::192.168.0.10::INSTR", log)
nrx = RohdeSchwarzNRX("", log)
sa1 = RohdeSchwarzFSQ("", log)

# Check connection status
if sa1.online:
	log.info("Spectrum Analyzer >ONLINE<")
else:
	log.critical("Failed to connect to spectrum analyzer!")
	exit()
if nrx.online:
	log.info("Power meters >ONLINE<")
else:
	log.critical("Failed to connect to power meters!")
	exit()
if sg1.online:
	log.info("Keysight SG >ONLINE<")
else:
	log.critical("Failed to connect to Keysight SG!")
	exit()
if sg2.online:
	log.info("Agilent SG >ONLINE<")
else:
	log.critical("Failed to connect to Agilent SG!")
	exit()
print()



dataset = []

# Configure spectrum analyzer
sa1.set_res_bandwidth(rbw)
sa1.set_freq_start(f_start)
sa1.set_freq_start(f_end)

npts = len(freq_a)*len(freq_b)*len(power_dBm)
count = 0

# Sweep points
for fa in freq_a:
	
	log.info(f"Changing freq_a to >{fa/1e9}< GHz")
	
	for fb in freq_b:
		
		log.info(f"Changing freq_b to >{fb/1e9}< GHz")
		
		for p_dBm in power_dBm:
			
			count += 1
			print(f"Beginning sweep {count} of {npts}")
			
			log.info(f"Changing power to >{p_dBm}< dBm")
			
			sg1.set_enable_rf(True)
			sg2.set_enable_rf(True)
			
			# Adjust conditions
			sg1.set_freq(fa)
			sg2.set_freq(fb)
			sg1.set_power(p_dBm)
			sg2.set_power(p_dBm)
			
			# Get waveform
			wvfrm = sa.get_trace_data(1)
			
			# Save data
			dp = {'freq_a_GHz':fa/1e9, 'freq_b_GHz':fb/1e9, 'power_dBm':p_dBm, 'waveform_f_s':wvfrm['x'], 'waveform_s_dBm':wvfrm['y']}
			dataset.append(dp)

sg1.set_enable_rf(False)
sg2.set_enable_rf(False)

# Make hybrid log/dataset file
hybrid = {'dataset':dataset, 'log':log.to_dict(), 'configuration': conf_data}

# Writing to sample.json
with open(f"{sweep_name}.json", "w") as outfile:
	outfile.write(json.dumps(hybrid, indent=4))