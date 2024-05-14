from heimdallr.all import *
import numpy as np

sweep_name = input(f"Dataset name: ")

# Create logger
log = LogPile()

# Connect to instruments
sg1 = Keysight8360L("GPIB::18::INSTR", log)
sg2 = AgilentE4400("GPIB::19::INSTR", log)
sa = SiglentSSA3000X("TCPIP0::192.168.0.10::INSTR", log)

if sa.online:
	log.info("Spectrum Analyzer >ONLINE<")
else:
	log.critical("Failed to connect to spectrum analyzer!")
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

# Sweep state
freq_a = list(np.linspace(0.8e9, 1.4e9, 7))
freq_b = list(np.linspace(0.1e9, 0.4e9, 4))
rbw = 100e3
f_start = 100e6
f_end = 2.1e9
# power_dBm = [0, 3, 6]
power_dBm = [-10, -6]

dataset = []

# Configure spectrum analyzer
sa.set_res_bandwidth(rbw)
sa.set_freq_start(f_start)
sa.set_freq_start(f_end)

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
hybrid = {'dataset':dataset, 'log':log.to_dict()}

# Writing to sample.json
with open(f"{sweep_name}.json", "w") as outfile:
    outfile.write(json.dumps(hybrid, indent=4))