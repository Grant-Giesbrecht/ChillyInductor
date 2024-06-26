'''
Used to perform Calibration Step 1A for Measurement Procedure 1 (MP1-C1A).

Measures the power at the output of the -10 dB hybrid on a spectrum analyzer (through)
and at the power meter port (coupled).

Goal is to justify model of coupler s.t. loss between coupled and through is only a function
of frequency.
'''

from heimdallr.all import *
import numpy as np
import os
from chillyinductor.rp22_helper import *
from collections import Counter

# Set directories for data and sweep configuration
CONF_DIRECTORY = "sweep_configs"
DATA_DIRECTORY = "data"
LOG_DIRECTORY = "logs"

# Create logger
log = LogPile()

def save_data(calset, conf_data, src_script, operator_notes, sweep_name, autosave:bool=False, log:LogPile=None):
	
	append_str = ""
	if autosave:
		append_str = "_autosave"
	
	conf_dict = {'source_script': src_script, 'operator_notes':operator_notes, 'configuration':json.dumps(conf_data)}
	
	# Make hybrid meta-data/dataset file
	root_dict = {'calibration_data':reform_dictlist(calset), 'info': conf_dict}
	
	# # Save data - JSON
	# t0 = time.time()
	# with open(os.path.join(DATA_DIRECTORY, f"{sweep_name}{append_str}.json"), "w") as outfile:
	# 	outfile.write(json.dumps(hybrid, indent=4))
	# t_save_json = time.time() - t0
	
	# if log is not None:
	# 	log.debug(f"Autosaved data to JSON in {t_save_json} seconds.")
	
	# Save log
	if log is not None:
		t0 = time.time()
		log.save_hdf(os.path.join(LOG_DIRECTORY, f"{sweep_name}{append_str}.log.hdf"))
		t_save_log = time.time() - t0
	
	if log is not None:
		log.debug(f"Autosaved log to HDF in {t_save_log} seconds.")
	
	# Save data - HDF5
	t0 = time.time()
	save_hdf(root_dict, os.path.join(DATA_DIRECTORY, f"{sweep_name}{append_str}.hdf"))
	t_save_hdf = time.time() - t0
	
	if log is not None:
		log.debug(f"Autosaved data to HDF in {t_save_hdf} seconds.")

# Get configuration
conf_file_prefix = input(f"Configuration file name: (No extension or folder)")
conf_file_name = os.path.join(".", CONF_DIRECTORY, f"{conf_file_prefix}.json")

# Load configuration data
try:
	with open(conf_file_name, "r") as outfile:
		conf_data = json.load(outfile)
except Exception as e:
	log.critical(f"Failed to read configuration file >{conf_file_name}<. ({e})")
	exit()
	
# Verify conf data exists
if conf_data is None:
	log.critical(f"Failed to read configuration file >{conf_file_name}<.")

# Interpret data
try:
	freq_rf = interpret_range(conf_data['frequency_RF'], print_err=True)
	freq_lo = interpret_range(conf_data['frequency_LO'], print_err=True)
	power_RF_dBm = interpret_range(conf_data['power_RF'], print_err=True)
	power_LO_dBm = interpret_range(conf_data['power_LO'], print_err=True)
	
	sa_conf = conf_data['spectrum_analyzer']
except:
	log.critical(f"Corrupt configuration file >{conf_file_name}<.")
	exit()
	
# Get dataset name
sweep_name = input(f"Dataset name: ")

# Get operator notes
operator_notes = input(f"Operator Notes: ")

# Connect to instruments
sg1 = Keysight8360L("GPIB::18::INSTR", log)
sg2 = AgilentE4400("GPIB::19::INSTR", log)
sa1 = RohdeSchwarzFSQ("TCPIP0::192.168.1.14::INSTR", log)
nrp = RohdeSchwarzNRP("USB0::0x0AAD::0x0139::101706::INSTR", log)

sa1.set_continuous_trigger(False)

#TODO: Add power calibration
#TODO: Add frequency calibration
#TODO: Add system setup with node-map and photo
#TODO: add DC bias other than manual (rip)
#TODO: Add in sweep-start option to write manual notes

# Check connection status
if sa1.online:
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
if nrp.online:
	log.info("Rohde&Schwarz NRP >ONLINE<")
else:
	log.critical("Failed to connect to Rohde&Schwarz NRP!")
	exit()
print()

dataset = []

# Get total number of points

dummy_sa_conditions = calc_sa_conditions(sa_conf, 1e9, 1e9, remove_duplicates=False, print_error=True)

try:
	npts = len(freq_rf)*len(freq_lo)*len(power_RF_dBm)*len(power_LO_dBm)*len(dummy_sa_conditions)
except Exception as e:
	log.critical(f"Error in configuration settings.")
	print(f"freq_rf = {freq_rf}")
	print(f"freq_lo = {freq_lo}")
	print(f"power_RF_dBm = {power_RF_dBm}")
	print(f"power_LO_dBm = {power_LO_dBm}")
	print(f"dummy_sa_conditions = {dummy_sa_conditions}")
count = 0

fb = 1e9
p_lo_dBm = -40
sg2.set_freq(fb)
sg2.set_power(p_lo_dBm)

#RF Points
fb = 1e9
for fa in freq_rf:
	
	log.info(f"Setting freq_rf to >{fa/1e9}< GHz")
	
	for p_rf_dBm in power_RF_dBm:
		
		log.info(f"Setting RF power to >{p_rf_dBm}< dBm")
			
		# Adjust conditions on signal generators
		sg1.set_freq(fa)
		sg1.set_power(p_rf_dBm)
		
		# Turn on signal generators
		sg1.set_enable_rf(True)
		sg2.set_enable_rf(False)
		
		sa_conditions = calc_sa_conditions(sa_conf, fa, None)
		
		# Set conditions for spectrum analyzer
		dp = None
		for idx_sac, sac in enumerate(sa_conditions):
			
			fstart = sac['f_start']
			fend = sac['f_end']
			frbw = sac['rbw']
			
			log.info(f"Measuring frequency range >{fstart/1e6}< MHz to >{fend/1e6}< MHz, RBW = >:q{frbw/1e3}< kHz.")
			
			count += 1
			print(f"Beginning measurement {count} of {npts}.")
			
			# Configure spectrum analyzer
			sa1.set_res_bandwidth(frbw)
			sa1.set_freq_start(fstart)
			sa1.set_freq_end(fend)
		
			# Start trigger on spectrum analyzer
			sa1.send_manual_trigger()
			
			# Perform NRP measurement if first sweep
			if dp is None:
				# Set frequency on power sensors
				nrp.set_meas_frequency(fa)
				
				# Trigger NRP
				nrp.send_trigger(wait=True)
				nrp_pwr = nrp.get_measurement()
			
			# Wait for FSQ to finish sweep
			sa1.wait_ready()
			
			# Get waveform
			wvfrm = sa1.get_trace_data(1)
			
			# Save data
			rbw_list = [sac['rbw']]*len(wvfrm['x'])
			if dp is None:
				dp = {'freq_rf_GHz':fa/1e9,
					'freq_lo_GHz':fb/1e9,
					'power_LO_dBm':p_lo_dBm,
					'power_RF_dBm': p_rf_dBm,
					'waveform_f_Hz':wvfrm['x'],
					'waveform_s_dBm':wvfrm['y'],
					'waveform_rbw_Hz':rbw_list,
					'rf_enabled': sg1.get_enable_rf(),
					'lo_enabled': sg2.get_enable_rf(),
					'coupled_power_meas_dBm': nrp_pwr
				}
			else:
				wav_x = dp['waveform_f_Hz'] + wvfrm['x']
				wav_y = dp['waveform_s_dBm'] + wvfrm['y']
				wav_rbw = dp['waveform_rbw_Hz'] + rbw_list
				
				# Find duplicate frequencies
				dupl_freqs = [k for k,v in Counter(wav_x).items() if v>1]
				
				# Found duplicates - resolve duplicates
				if len(dupl_freqs) > 0:
					pass
				
				# Sort result
				wav_x_new = []
				wav_y_new = []
				wav_rbw_new = []
				for wx,wy,wr in sorted(zip(wav_x, wav_y, wav_rbw)):
					wav_x_new.append(wx)
					wav_y_new.append(wy)
					wav_rbw_new.append(wr)
				
				# Update dictionary
				dp['waveform_f_Hz'] = wav_x_new
				dp['waveform_s_dBm'] = wav_y_new
				dp['waveform_rbw_Hz'] = wav_rbw_new
		
		# Append to dataset
		dataset.append(dp)

fa = 1e9
p_rf_dBm = -40
sg1.set_freq(fa)
sg1.set_power(p_rf_dBm)

# LO Points
for fb in freq_lo:
	
	log.info(f"Setting freq_lo to >{fb/1e9}< GHz")
		
	for p_lo_dBm in power_LO_dBm:
		
		log.info(f"Changing LO power to >{p_lo_dBm}< dBm")
		
		# Adjust conditions on signal generators
		sg2.set_freq(fb)
		sg2.set_power(p_lo_dBm)
		
		# Turn on signal generators
		sg1.set_enable_rf(False)
		sg2.set_enable_rf(True)
		
		sa_conditions = calc_sa_conditions(sa_conf, None, fb)
		
		# Set conditions for spectrum analyzer
		dp = None
		for idx_sac, sac in enumerate(sa_conditions):
			
			fstart = sac['f_start']
			fend = sac['f_end']
			frbw = sac['rbw']
			
			log.info(f"Measuring frequency range >{fstart/1e6}< MHz to >{fend/1e6}< MHz, RBW = >:q{frbw/1e3}< kHz.")
			
			count += 1
			print(f"Beginning measurement {count} of {npts}.")
			
			# Configure spectrum analyzer
			sa1.set_res_bandwidth(frbw)
			sa1.set_freq_start(fstart)
			sa1.set_freq_end(fend)
		
			# Start trigger on spectrum analyzer
			sa1.send_manual_trigger()
			
			# Perform NRP measurement if first sweep
			if dp is None:
				# Set frequency on power sensors
				nrp.set_meas_frequency(fb)
				
				# Trigger NRP
				nrp.send_trigger(wait=True)
				nrp_pwr = nrp.get_measurement()
			
			# Wait for FSQ to finish sweep
			sa1.wait_ready()
			
			# Get waveform
			wvfrm = sa1.get_trace_data(1)
			
			# Save data
			rbw_list = [sac['rbw']]*len(wvfrm['x'])
			if dp is None:
				dp = {'freq_rf_GHz':fa/1e9,
					'freq_lo_GHz':fb/1e9,
					'power_LO_dBm':p_lo_dBm,
					'power_RF_dBm': p_rf_dBm,
					'waveform_f_Hz':wvfrm['x'],
					'waveform_s_dBm':wvfrm['y'],
					'waveform_rbw_Hz':rbw_list,
					'rf_enabled': sg1.get_enable_rf(),
					'lo_enabled': sg2.get_enable_rf(),
					'coupled_power_meas_dBm': nrp_pwr
				}
			else:
				wav_x = dp['waveform_f_Hz'] + wvfrm['x']
				wav_y = dp['waveform_s_dBm'] + wvfrm['y']
				wav_rbw = dp['waveform_rbw_Hz'] + rbw_list
				
				# Find duplicate frequencies
				dupl_freqs = [k for k,v in Counter(wav_x).items() if v>1]
				
				# Found duplicates - resolve duplicates
				if len(dupl_freqs) > 0:
					pass
				
				# Sort result
				wav_x_new = []
				wav_y_new = []
				wav_rbw_new = []
				for wx,wy,wr in sorted(zip(wav_x, wav_y, wav_rbw)):
					wav_x_new.append(wx)
					wav_y_new.append(wy)
					wav_rbw_new.append(wr)
				
				# Update dictionary
				dp['waveform_f_Hz'] = wav_x_new
				dp['waveform_s_dBm'] = wav_y_new
				dp['waveform_rbw_Hz'] = wav_rbw_new
		
		# Append to dataset
		dataset.append(dp)

# Turn off signal generators
sg1.set_enable_rf(False)
sg2.set_enable_rf(False)

# Save data and logs
save_data(dataset, conf_data, __file__, operator_notes, sweep_name, autosave=False, log=log)

