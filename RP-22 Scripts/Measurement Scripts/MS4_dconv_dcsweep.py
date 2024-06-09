'''
Builds off of MS1 by integrating a calibration routine into the mix.

Performs the primary measurements (post-calibration) for MP1 (Measurement Procedure 1).
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

t_last_autosave = time.time()
AUTOSAVE_PERIOD_S = 60*10

def save_data(dataset, calset, conf_data, src_script, operator_notes, sweep_name, autosave:bool=False, log:LogPile=None):
	
	append_str = ""
	if autosave:
		append_str = "_autosave"
	
	conf_dict = {'source_script': src_script, 'operator_notes':operator_notes, 'configuration':json.dumps(conf_data)}
	
	# Make hybrid meta-data/dataset file
	root_dict = {'dataset': reform_dictlist(dataset), 'calibration_data':reform_dictlist(calset), 'info': conf_dict}
	
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
	
# Create logger
log = LogPile()

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
	
# Verify conf data exists
if conf_data is None:
	log.critical(f"Failed to read configuration file >{conf_file_name}<.")
	exit()
	
# Interpret data
try:
	freq_rf = interpret_range(conf_data['frequency_RF'], print_err=True)
	freq_lo = interpret_range(conf_data['frequency_LO'], print_err=True)
	power_RF_dBm = interpret_range(conf_data['power_RF'], print_err=True)
	power_LO_dBm = interpret_range(conf_data['power_LO'], print_err=True)
	Idc_list_mA = interpret_range(conf_data['dc_current'], print_err=True)
	CURR_SENSE_R = float(conf_data['curr_sense_ohms'])
	
	sa_conf = conf_data['spectrum_analyzer']
except:
	log.critical(f"Corrupt configuration file >{conf_file_name}<.")
	exit()
	
# Get dataset name
sweep_name = input(f"{Fore.YELLOW}Dataset name: {Style.RESET_ALL}")

# Get operator notes
operator_notes = input(f"{Fore.YELLOW}Operator Notes: {Style.RESET_ALL}")

# Connect to instruments
sg1 = Keysight8360L("GPIB::18::INSTR", log)
sg2 = AgilentE4400("GPIB::19::INSTR", log)
nrp = RohdeSchwarzNRP("USB0::0x0AAD::0x0139::101706::INSTR", log)
sa1 = RohdeSchwarzFSQ("TCPIP0::192.168.1.14::INSTR", log)
osc = RigolDS1000Z("TCPIP0::192.168.1.20::INSTR", log)
mfli = ZurichInstrumentsMFLI("dev5652", "192.168.88.82", log)

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
if osc.online:
	log.info("Rigol DS1000Z >ONLINE<")
else:
	log.critical("Failed to connect to Rigol DS1000Z!")
	exit()
if mfli.online:
	log.info("Zurich Instruments MFLI >ONLINE<")
else:
	log.critical("Failed to connect to Zurich Instruments MFLI!")
	exit()
print()

calset = []
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

##======================================================
# Configure MFLI

mfli.set_50ohm(False)
mfli.set_range(10)
mfli.set_output_ac_ampl(0)
mfli.set_output_ac_enable(False)
mfli.set_differential_enable(True)
mfli.set_offset(0)
mfli.set_output_enable(True)

##======================================================
# Configure oscilloscope

osc.clear_measurements()
osc.add_measurement(Oscilloscope2Ctg.MEAS_VAVG, 1) # Chan. 1 average 
osc.add_measurement(Oscilloscope2Ctg.MEAS_VAVG, 2) # Chan. 2 average

# Run in-situ calibration

sa1.set_continuous_trigger(False) # Otherwise wait-ready will not work!

# Turn off LO signal generators
sg2.set_enable_rf(False)

# RF Calibration
fb = 1e9
p_lo_dBm = -40
for fa in freq_rf:
	log.info(f"Setting freq_rf to >{fa/1e9}< GHz")
	for p_rf_dBm in power_RF_dBm:
		log.info(f"Setting RF power to >{p_rf_dBm}< dBm")
		
		# Configure SG
		sg1.set_freq(fa)
		sg1.set_power(p_rf_dBm)
		sg1.set_enable_rf(True)
		
		# Measure on SA
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
		calset.append(dp)
sg1.set_enable_rf(False)
sg2.set_enable_rf(False)

# LO Calibration
fa = 1e9
p_rf_dBm = -40
for fb in freq_lo:
	log.info(f"Setting freq_lo to >{fb/1e9}< GHz")
	for p_lo_dBm in power_LO_dBm:
		log.info(f"Setting LO power to >{p_lo_dBm}< dBm")
		
		# Configure SG
		sg2.set_freq(fb)
		sg2.set_power(p_lo_dBm)
		sg2.set_enable_rf(True)
		
		# Measure on SA
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
		calset.append(dp)

log.info("Beginning measurements.")

# Sweep points
for Idc in Idc_list_mA:
	
	# Set offset voltage
	Voffset = CURR_SENSE_R * Idc/1e3
	mfli.set_offset(Voffset)
	mfli.set_output_enable(True)
	
	for fa in freq_rf:
		
		log.info(f"Setting freq_rf to >{fa/1e9}< GHz")
		
		for fb in freq_lo:
			
			log.info(f"Setting freq_lo to >{fb/1e9}< GHz")
			
			for p_rf_dBm in power_RF_dBm:
				
				log.info(f"Setting RF power to >{p_rf_dBm}< dBm")
				
				for p_lo_dBm in power_LO_dBm:
					
					log.info(f"Changing LO power to >{p_lo_dBm}< dBm")
					
					# Adjust conditions on signal generators
					sg1.set_freq(fa)
					sg2.set_freq(fb)
					sg1.set_power(p_rf_dBm)
					sg2.set_power(p_lo_dBm)
					
					# Turn on signal generators
					sg1.set_enable_rf(True)
					sg2.set_enable_rf(True)
					
					sa_conditions = calc_sa_conditions(sa_conf, fa, fb)
					
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
						
						# Read DC current
						Vch1 = osc.get_measurement(Oscilloscope2Ctg.MEAS_VAVG, 1)
						Vch2 = osc.get_measurement(Oscilloscope2Ctg.MEAS_VAVG, 2)
						
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
								'coupled_power_meas_dBm': nrp_pwr,
								'DC_current_est_V': np.abs(Vch1 - Vch2),
								'target_DC_current_mA': Idc,
								'output_set_voltage_V': Voffset
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
					
					# Check for autosave
					if time.time() - t_last_autosave > AUTOSAVE_PERIOD_S:
						
						log.debug("Autosaving data and logs")
						
						# Autosave data and logs
						save_data(dataset, calset, conf_data, __file__, operator_notes, sweep_name, autosave=True, log=log)
						
						# Change autosave time
						t_last_autosave = time.time()

# Turn off signal generators
sg1.set_enable_rf(False)
sg2.set_enable_rf(False)

# Save data and logs
save_data(dataset, calset, conf_data, __file__, operator_notes, sweep_name, autosave=False, log=log)

