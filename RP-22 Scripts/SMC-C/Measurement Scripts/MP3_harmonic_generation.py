''' The purpose of this script is to measure the harmonic generation capacity of the C2024Q1 chips.

Allows a temperature-wait condition to be set s.t. the sweep doesn't begin until the cryostat is properly cooled-down. 
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
CONF_DIRECTORY = "sweep_configs"
# DATA_DIRECTORY = "data"
# LOG_DIRECTORY = "logs"

# DATA_DIRECTORY = "C:\\Users\\gmg3\\Mega\\remote_data\\data"
# LOG_DIRECTORY = "C:\\Users\\gmg3\\Mega\\remote_data\\logs"

DATA_DIRECTORY = "C:\\Users\\gmg3\\OneDrive - UCB-O365\\remote_data\\data"
LOG_DIRECTORY = "C:\\Users\\gmg3\\OneDrive - UCB-O365\\remote_data\\logs"

# Set autosave period in seconds
TIME_AUTOSAVE_S = 600

# Time to wait in between temp-wait temperature checks
TEMPWAIT_PERIOD_S = 10

# Time after going normal to let the chip recover
RECOVERY_TIME_S = 1

# Fraction of set voltage which, if meas voltage is less than, chip is considered to have gone normal
THRESHOLD_NORMAL_V = 0.25

##======================================================
# Read user arguments

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--detail', help="Show detailed log messages.", action='store_true')
parser.add_argument('-x', '--disablerf', help="Disable RF power in sweep.", action='store_true')
parser.add_argument('--loglevel', help="Set the logging display level.", choices=['LOWDEBUG', 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], type=str.upper)
parser.add_argument('--tempwait', help="Set the logging display level.", type=float)
args = parser.parse_args()

disable_rf = args.disablerf

# Initialize log
log = LogPile()
if args.loglevel is not None:
	log.set_terminal_level(args.loglevel)
else:
	log.set_terminal_level("DEBUG")
log.str_format.show_detail = args.detail

# Check for tempwait argument
if args.tempwait is not None:
	log.info(f"Will wait to begin sweep until temperature >{args.tempwait} K< has been reached.")

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
	freq_rf = interpret_range(conf_data['frequency_RF'], print_err=True)
	power_rf_dBm = interpret_range(conf_data['power_RF'], print_err=True)
	
	idc_coarse_step_mA = float(conf_data['Idc_coarse_step_mA']) # Step used for step-1 climbing
	idc_resolution_step_mA = float(conf_data['Idc_resolution_step_mA']) # Step used for step-2, zeroing-in
	idc_fine_step_mA = float(conf_data['Idc_fine_step_mA']) # Step size used for step-3 fine-climbing
	
	norm_extra_Z = float(conf_data['extra_Z_norm_ohm'])
	Ic_uncert_backoff_mult = float(conf_data['Ic_uncert_backoff_mult'])
	
	sa_conf = conf_data['spectrum_analyzer']
except:
	log.critical(f"Corrupt configuration file >{conf_file_name}<.")
	exit()

# log.info(f"Detected {len(temp_sp_list)} measurement points.", detail=f"Measurement points = {temp_sp_list} [K]")

##======================================================
# Initialize instruments

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

mfli = ZurichInstrumentsMFLI("dev5652", "192.168.88.82", log)
if mfli.online:
	log.info("Zurich Instruments MFLI >ONLINE<")
else:
	log.critical("Failed to connect to Zurich Instruments MFLI!")
	exit()

sig_gen = Keysight8360L("GPIB::18::INSTR", log)
if sig_gen.online:
	log.info("Keysight signal generator >ONLINE<")
else:
	log.critical("Failed to connect to Keysight signal generator!")
	exit()
sig_gen.set_enable_rf(False)

spec_an = RohdeSchwarzFSE("GPIB::20::INSTR", log)
if spec_an.online:
	log.info("Rohde & Schwarz FSEK spectrum analyzer >ONLINE<")
else:
	log.critical("Failed to connect to Rohde & Schwarz FSEK spectrum analyzer!")
	exit()

##======================================================
# Get User Input re: sweep

sweep_name = input("Sweep name: ")
operator_notes = input("Operator notes: ")

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
# Configure spectrum analyzer

spec_an.set_ref_level(10)

spec_an.set_continuous_trigger(False) # Otherwise wait-ready will not work!
spec_an.set_y_div(20) # Set y scale so *everything* is visible

##======================================================
# Calibrate current-sense resistor

# Wait for user to connect through
print(markdown(f"Measuring impedance of current sense resistor. Leave the multimeter connected to the current sense resistor.\n\n>Please disconnect lines from CS resistor to bias tees.<"))
a = input("Press enter when ready:")

# Prepare DMM to measure resistance
dmm.set_measurement(DigitalMultimeterCtg1.MEAS_RESISTANCE_2WIRE, DigitalMultimeterCtg1.RANGE_AUTO)

# Measure resistance
cs_res = dmm.trigger_and_read()
cs_time = datetime.datetime.now()

log.info(f">Current-sense resistor impedance:< Measured R = {cs_res} ohms.")

# Have the script read itself (to put in output file)
try:
	self_contents = Path(__file__).read_text()
except Exception as e:
	self_contents = f"ERROR: Failed to read script source. ({e})"


# Define dataset dictionary
calibration_dict = {"current_sense_res_ohms": cs_res, "time_of_meas": str(cs_time)}
dataset_dict = {"freq_rf_GHz":[], "power_rf_dBm":[], "times":[], "MFLI_V_offset_V":[], "requested_Idc_mA":[], "raw_meas_Vdc_V":[], "Idc_mA":[], "detect_normal":[], "temperature_K":[], "waveform_f_Hz":[], "waveform_s_dBm":[], "waveform_rbw_Hz":[], "measurement_stage":[]}

summary_dict = {'Ic_meas_mA':[], "Ic_set_mA":[], "Ic_uncert_mA": [], "power_rf_dBm":[], "freq_rf_GHz":[], "times":[], "temperature_K":[]}

info_dict = {'source_script': __file__, 'operator_notes':operator_notes, 'configuration': json.dumps(conf_data), 'source_script_full':self_contents}

dataset = {"calibration":calibration_dict, "dataset":dataset_dict, "summary_dataset":summary_dict, 'info':info_dict}


# Prepare DMM to measure voltage
dmm.set_measurement(DigitalMultimeterCtg1.MEAS_VOLT_DC, DigitalMultimeterCtg1.RANGE_AUTO)

# Wait for user to connect through
print(markdown(f"Ready to perform primary sweep. >Reconnect current sense resistor<."))
a = input("Press enter when ready:")

##======================================================
# Wait for starting temperature (if requested)

if args.tempwait is not None:
	
	while True:
		
		# Measure temperature
		t_meas = tempctrl.get_temp()
		
		# Check break condition
		if t_meas < args.tempwait:
			log.debug(f"Exiting temperature-wait phase. Measured temperature (>{t_meas} K<) is below starting threshold (>:a{args.tempwait} K<).")
			break
		
		log.debug(f"Temperature (>{t_meas} K<) is above starting threshold (>:a{args.tempwait} K<).")
		time.sleep(TEMPWAIT_PERIOD_S)

##======================================================
# Measure real system impedance

time_last_save = time.time()

# Initialize current set resistance from the current sense resistor
current_set_res = None

idc_cs_res_meas_mA = 0.1
log.debug(f"Preparing to measure system impedance.")
Voffset = cs_res * idc_cs_res_meas_mA/1e3
mfli.set_offset(Voffset)
mfli.set_output_enable(True)

# Wait
time.sleep(0.01)

# Read current
v_cs_read = np.abs(dmm.trigger_and_read())
idc_read = v_cs_read/cs_res*1e3
log.info(f"With current target of {idc_cs_res_meas_mA} mA, set Vout={Voffset*1e3} mV, measured Idc={idc_read} mA.")

# Update resistance figure
current_set_res = Voffset/(idc_read/1e3)
log.info(f"Setting system impedance to {current_set_res} Ohms.")

mfli.set_offset(0)

##======================================================
# Begin logging data

if disable_rf:
	freq_rf = [1e9]
	power_rf_dBm = [-40]

# Scan over all frequencies
abort = False
for f_rf in freq_rf:
	
	if abort:
		break
	
	log.info(f"Setting freq_rf to >{f_rf/1e9}< GHz.")
	
	# Scan over all powers
	for p_rf in power_rf_dBm:
		
		if abort:
			break
		
		norm_detected = False
		
		log.info(f"Setting RF power to >{p_rf}< dBm.")
		
		log.info(f"Searching for Ic: Ramping up until normal.")
		
		# Prepare the signal generator
		if disable_rf:
			sig_gen.set_enable_rf(False)
		else:
			sig_gen.set_freq(f_rf)
			sig_gen.set_power(p_rf)
			sig_gen.set_enable_rf(True)
		
		# Prepare the spectrum analyzer
		sa_conditions = calc_sa_conditions(sa_conf, f_rf, f_lo=None)
		
		#------ Find critical current ----------
		# Step 1: Move up in steps until you go normal
		measurement_stage = 0
		
		highest_sc_meas = None
		highest_sc_set = 0
		lowest_norm_meas = None
		lowest_norm_set = None
		
		# Loop until normal
		idc_target = 0
		while True:
			
			# Pick new target current
			idc_target += idc_coarse_step_mA
			
			# Set offset voltage
			Voffset = current_set_res * idc_target/1e3
			log.debug(f"Selected offset voltage {Voffset} V from current_set_res={current_set_res} Ohms and idc={idc_target} mA")
			mfli.set_offset(Voffset)
			mfli.set_output_enable(True)
			
			# Read Vdc (-> Idc)
			v_cs_read = np.abs(dmm.trigger_and_read())
			
			# Calculate current
			I_meas_A = v_cs_read/cs_res
			log.debug(f"Measured bias current at {I_meas_A*1e3} mA", detail=f"V_current_sense_read = {v_cs_read} V, CS-resistor = {cs_res} ohms.")
			
			# Read temperature
			t_meas = tempctrl.get_temp()
			
			# Set conditions for spectrum analyzer
			wav_rbw = None
			wav_f = None
			wav_s = None
			for idx_sac, sac in enumerate(sa_conditions):
				
				fstart = sac['f_start']
				fend = sac['f_end']
				frbw = sac['rbw']
				
				log.debug(f"Spectrum analyzer measuring frequency range >{fstart/1e6}< MHz to >{fend/1e6}< MHz, RBW = >:q{frbw/1e3}< kHz.")
				
				# Configure spectrum analyzer
				spec_an.set_res_bandwidth(frbw)
				spec_an.set_freq_start(fstart)
				spec_an.set_freq_end(fend)
				
				# Start trigger on spectrum analyzer
				spec_an.send_manual_trigger()
				
				# Wait for FSQ to finish sweep
				spec_an.wait_ready()
				
				# Get waveform
				wvfrm = spec_an.get_trace_data(1)
				
				# Save data
				rbw_list = [sac['rbw']]*len(wvfrm['x'])
				if wav_f is None:
					wav_f = wvfrm['x']
					wav_s = wvfrm['y']
					wav_rbw = rbw_list
				else:
					wav_f_temp = wav_f + wvfrm['x']
					wav_s_temp = wav_s + wvfrm['y']
					wav_rbw_temp = wav_rbw+ rbw_list
					
					# Sort result
					wav_f = []
					wav_s = []
					wav_rbw = []
					for wx,wy,wr in sorted(zip(wav_f_temp, wav_s_temp, wav_rbw_temp)):
						wav_f.append(wx)
						wav_s.append(wy)
						wav_rbw.append(wr)
			
			# Calcualte additional resistance in system
			system_Z = Voffset/I_meas_A
			additional_Z = system_Z - current_set_res
			if additional_Z > norm_extra_Z:
				log.info(f"Chip detected as normal (>Z_extra = {additional_Z} Ohms<). Threshold is {norm_extra_Z} Ohms.")
				lowest_norm_meas = I_meas_A*1e3
				lowest_norm_set = idc_target
				
				# Let chip recover
				log.debug("Turning off DC bias and signal generator to let chip recover")
				sig_gen.set_enable_rf(False)
				mfli.set_offset(0)
				log.debug(f"Waiting {RECOVERY_TIME_S} seconds for chip to recover.")
				time.sleep(RECOVERY_TIME_S)
				log.debug("Post-normal recovery procedure complete. Proceeding to next sweep point.")
				
				# Quit climbing
				break
			else:
				highest_sc_meas = I_meas_A*1e3
				highest_sc_set = idc_target
			
			dataset['dataset']['freq_rf_GHz'].append(f_rf)
			dataset['dataset']['power_rf_dBm'].append(p_rf)
			
			dataset['dataset']['waveform_f_Hz'].append(wav_f)
			dataset['dataset']['waveform_s_dBm'].append(wav_s)
			dataset['dataset']['waveform_rbw_Hz'].append(wav_rbw)
			
			dataset['dataset']['MFLI_V_offset_V'].append(Voffset)
			dataset['dataset']['requested_Idc_mA'].append(idc_target)
			dataset['dataset']['raw_meas_Vdc_V'].append(v_cs_read)
			dataset['dataset']['Idc_mA'].append(v_cs_read/cs_res*1e3)
			dataset['dataset']['detect_normal'].append(norm_detected)
			
			dataset['dataset']['temperature_K'].append(t_meas)
			dataset['dataset']['times'].append(str(datetime.datetime.now()))
			dataset['dataset']['measurement_stage'].append(measurement_stage)
			
			# Check for autosave
			if time.time() - time_last_save > TIME_AUTOSAVE_S:
				
				# Save data and log
				log.save_hdf(os.path.join(LOG_DIRECTORY, f"{sweep_name}_autosave.log.hdf"))
				
				dict_to_hdf(dataset, os.path.join(DATA_DIRECTORY, f"{sweep_name}_autosave.hdf"))
				log.debug(f"Autosaved logs and data.")
				
				time_last_save = time.time()
		
		
		#------ Find critical current ----------
		# Step 2: Perform binary search from max to locate critical current with tolerance
		measurement_stage = 1
		
		log.info(f"Searching for Ic: Running binary search.")
		
		while True:
			
			# Calculate delta, check for end condition
			#NOTE: Cannot calculate (delta_meas = lowest_norm_meas - highest_sc_meas) because highest_sc_meas is meaningless (val taken when chip is doing god knows what)
			# INstead we look at how our measured values are missing the set point, then using that to extrapolate what real measured current CANNOT be hit because the chip would first go normal.
			correction_factor = highest_sc_meas/highest_sc_set # Calculate correction from SP to measured
			est_norm_current = correction_factor*lowest_norm_set
			delta_meas = est_norm_current - highest_sc_meas
			delta_set = lowest_norm_set - highest_sc_set
			if (delta_set < idc_resolution_step_mA):
				log.info(f"Met end condition. Final range [S.P. >{highest_sc_set} mA - {lowest_norm_set} mA<]. Measured range: [>:a{highest_sc_meas} mA - ({est_norm_current}) mA<]")
				log.info(f"Critical current uncertainty (>{delta_meas} mA< measured, >{delta_set} mA< S.P.) is within margin ({idc_resolution_step_mA} mA)")
				
				Ic = (est_norm_current + highest_sc_meas)/2
				Ic_uncert = delta_meas
				
				log.info(f"Ic = >{Ic}< mA +/- {Ic_uncert/2} mA")
				
				break
			
			# Else divide range by 2
			idc_target = (lowest_norm_set + highest_sc_set)/2
			log.info(f"Setting new target current to >{idc_target} mA<, splitting range [S.P. >{highest_sc_set} mA - {lowest_norm_set} mA<]. Measured range: [>:a{highest_sc_meas} mA - {lowest_norm_meas} mA<]")
			
			# Set current, take measurements
			
			# Set offset voltage
			Voffset = current_set_res * idc_target/1e3
			log.debug(f"Selected offset voltage {Voffset} V from current_set_res={current_set_res} Ohms and idc={idc_target} mA")
			mfli.set_offset(Voffset)
			mfli.set_output_enable(True)
			
			# Read Vdc (-> Idc)
			v_cs_read = np.abs(dmm.trigger_and_read())
			
			# Calculate current
			I_meas_A = v_cs_read/cs_res
			log.debug(f"Measured bias current at {I_meas_A*1e3} mA", detail=f"V_current_sense_read = {v_cs_read} V, CS-resistor = {cs_res} ohms.")
			
			# Read temperature
			t_meas = tempctrl.get_temp()
			
			# Set conditions for spectrum analyzer
			wav_rbw = None
			wav_f = None
			wav_s = None
			for idx_sac, sac in enumerate(sa_conditions):
				
				fstart = sac['f_start']
				fend = sac['f_end']
				frbw = sac['rbw']
				
				log.debug(f"Spectrum analyzer measuring frequency range >{fstart/1e6}< MHz to >{fend/1e6}< MHz, RBW = >:q{frbw/1e3}< kHz.")
				
				# Configure spectrum analyzer
				spec_an.set_res_bandwidth(frbw)
				spec_an.set_freq_start(fstart)
				spec_an.set_freq_end(fend)
				
				# Start trigger on spectrum analyzer
				spec_an.send_manual_trigger()
				
				# Wait for FSQ to finish sweep
				spec_an.wait_ready()
				
				# Get waveform
				wvfrm = spec_an.get_trace_data(1)
				
				# Save data
				rbw_list = [sac['rbw']]*len(wvfrm['x'])
				if wav_f is None:
					wav_f = wvfrm['x']
					wav_s = wvfrm['y']
					wav_rbw = rbw_list
				else:
					wav_f_temp = wav_f + wvfrm['x']
					wav_s_temp = wav_s + wvfrm['y']
					wav_rbw_temp = wav_rbw+ rbw_list
					
					# Sort result
					wav_f = []
					wav_s = []
					wav_rbw = []
					for wx,wy,wr in sorted(zip(wav_f_temp, wav_s_temp, wav_rbw_temp)):
						wav_f.append(wx)
						wav_s.append(wy)
						wav_rbw.append(wr)
			
			# Calcualte additional resistance in system
			system_Z = Voffset/I_meas_A
			additional_Z = system_Z - current_set_res
			if additional_Z > norm_extra_Z:
				log.info(f"Chip detected as normal (>Z_extra = {additional_Z} Ohms<). Threshold is {norm_extra_Z} Ohms.")
				lowest_norm_meas = np.min([I_meas_A*1e3, lowest_norm_meas])
				lowest_norm_set = idc_target
				
				# Let chip recover
				log.debug("Turning off DC bias and signal generator to let chip recover")
				sig_gen.set_enable_rf(False)
				mfli.set_offset(0)
				log.debug(f"Waiting {RECOVERY_TIME_S} seconds for chip to recover.")
				time.sleep(RECOVERY_TIME_S)
				log.debug("Post-normal recovery procedure complete. Proceeding to next sweep point.")
			else:
				highest_sc_meas = I_meas_A*1e3
				highest_sc_set = idc_target
				log.info(f"Chip remained superconducting.")
			
			log.debug(f"Saving datapoint.")
			
			dataset['dataset']['freq_rf_GHz'].append(f_rf)
			dataset['dataset']['power_rf_dBm'].append(p_rf)
			
			dataset['dataset']['waveform_f_Hz'].append(wav_f)
			dataset['dataset']['waveform_s_dBm'].append(wav_s)
			dataset['dataset']['waveform_rbw_Hz'].append(wav_rbw)
			
			dataset['dataset']['MFLI_V_offset_V'].append(Voffset)
			dataset['dataset']['requested_Idc_mA'].append(idc_target)
			dataset['dataset']['raw_meas_Vdc_V'].append(v_cs_read)
			dataset['dataset']['Idc_mA'].append(v_cs_read/cs_res*1e3)
			dataset['dataset']['detect_normal'].append(norm_detected)
			
			dataset['dataset']['temperature_K'].append(t_meas)
			dataset['dataset']['times'].append(str(datetime.datetime.now()))
			dataset['dataset']['measurement_stage'].append(measurement_stage)
			
			# Check for autosave
			if time.time() - time_last_save > TIME_AUTOSAVE_S:
				
				# Save data and log
				log.save_hdf(os.path.join(LOG_DIRECTORY, f"{sweep_name}_autosave.log.hdf"))
				
				dict_to_hdf(dataset, os.path.join(DATA_DIRECTORY, f"{sweep_name}_autosave.hdf"))
				log.debug(f"Autosaved logs and data.")
				
				time_last_save = time.time()
		
		#------ Find critical current ----------
		# Step 3: Fine stepped search
		measurement_stage = 2
		
		log.info(f"Searching for Ic: Running fine stepped search.")
		
		Ic = (est_norm_current + highest_sc_meas)/2
		Ic_uncert = delta_meas
		
		# Initialize target Idc from Ic and uncertainty.
		idc_target = Ic - Ic_uncert*Ic_uncert_backoff_mult
		last_sc_point_targ = None
		last_sc_point_meas = None
		
		running = True
		while running:
			
			# Increase target current
			idc_target += idc_fine_step_mA
			
			# Set current, take measurements
			
			# Set offset voltage
			Voffset = current_set_res * idc_target/1e3
			log.debug(f"Selected offset voltage {Voffset} V from current_set_res={current_set_res} Ohms and idc={idc_target} mA")
			mfli.set_offset(Voffset)
			mfli.set_output_enable(True)
			
			# Read Vdc (-> Idc)
			v_cs_read = np.abs(dmm.trigger_and_read())
			
			# Calculate current
			I_meas_A = v_cs_read/cs_res
			log.debug(f"Measured bias current at {I_meas_A*1e3} mA", detail=f"V_current_sense_read = {v_cs_read} V, CS-resistor = {cs_res} ohms.")
			
			# Read temperature
			t_meas = tempctrl.get_temp()
			
			# Set conditions for spectrum analyzer
			wav_rbw = None
			wav_f = None
			wav_s = None
			for idx_sac, sac in enumerate(sa_conditions):
				
				fstart = sac['f_start']
				fend = sac['f_end']
				frbw = sac['rbw']
				
				log.debug(f"Spectrum analyzer measuring frequency range >{fstart/1e6}< MHz to >{fend/1e6}< MHz, RBW = >:q{frbw/1e3}< kHz.")
				
				# Configure spectrum analyzer
				spec_an.set_res_bandwidth(frbw)
				spec_an.set_freq_start(fstart)
				spec_an.set_freq_end(fend)
				
				# Start trigger on spectrum analyzer
				spec_an.send_manual_trigger()
				
				# Wait for FSQ to finish sweep
				spec_an.wait_ready()
				
				# Get waveform
				wvfrm = spec_an.get_trace_data(1)
				
				# Save data
				rbw_list = [sac['rbw']]*len(wvfrm['x'])
				if wav_f is None:
					wav_f = wvfrm['x']
					wav_s = wvfrm['y']
					wav_rbw = rbw_list
				else:
					wav_f_temp = wav_f + wvfrm['x']
					wav_s_temp = wav_s + wvfrm['y']
					wav_rbw_temp = wav_rbw+ rbw_list
					
					# Sort result
					wav_f = []
					wav_s = []
					wav_rbw = []
					for wx,wy,wr in sorted(zip(wav_f_temp, wav_s_temp, wav_rbw_temp)):
						wav_f.append(wx)
						wav_s.append(wy)
						wav_rbw.append(wr)
			
			# Calcualte additional resistance in system
			system_Z = Voffset/I_meas_A
			additional_Z = system_Z - current_set_res
			if additional_Z > norm_extra_Z:
				log.info(f"Chip detected as normal (>Z_extra = {additional_Z} Ohms<). Threshold is {norm_extra_Z} Ohms.")
				
				correction_factor = last_sc_point_meas/last_sc_point_targ
				Ic_uncert_meas = correction_factor*idc_target - last_sc_point_meas
				Ic_uncert_targ = idc_target - last_sc_point_targ
				uncert_max = np.max([Ic_uncert_meas, Ic_uncert_targ])
				
				log.info(f"Saving summary data point: Ic_meas = >{last_sc_point_meas} mA< Ic_set = >{last_sc_point_targ} mA<, Ic_uncertainty = >:a{uncert_max} mA<.")
				
				# Save summary data
				dataset['summary_dataset']['Ic_meas_mA'].append(last_sc_point_meas)
				dataset['summary_dataset']['Ic_set_mA'].append(last_sc_point_targ)
				dataset['summary_dataset']['Ic_uncert_mA'].append( uncert_max )
				dataset['summary_dataset']['power_rf_dBm'].append(p_rf)
				dataset['summary_dataset']['freq_rf_GHz'].append(f_rf)
				dataset['summary_dataset']['times'].append(str(datetime.datetime.now()))
				dataset['summary_dataset']['temperature_K'].append(t_meas)
				
				# Let chip recover
				log.debug("Turning off DC bias and signal generator to let chip recover")
				sig_gen.set_enable_rf(False)
				mfli.set_offset(0)
				log.debug(f"Waiting {RECOVERY_TIME_S} seconds for chip to recover.")
				time.sleep(RECOVERY_TIME_S)
				log.debug("Post-normal recovery procedure complete.")
				
				running = False
			else:
				log.debug(f"Chip remained superconducting.")
				last_sc_point_targ = idc_target
				last_sc_point_meas = I_meas_A*1e3
			
			log.debug(f"Saving datapoint.")
			
			dataset['dataset']['freq_rf_GHz'].append(f_rf)
			dataset['dataset']['power_rf_dBm'].append(p_rf)
			
			dataset['dataset']['waveform_f_Hz'].append(wav_f)
			dataset['dataset']['waveform_s_dBm'].append(wav_s)
			dataset['dataset']['waveform_rbw_Hz'].append(wav_rbw)
			
			dataset['dataset']['MFLI_V_offset_V'].append(Voffset)
			dataset['dataset']['requested_Idc_mA'].append(idc_target)
			dataset['dataset']['raw_meas_Vdc_V'].append(v_cs_read)
			dataset['dataset']['Idc_mA'].append(v_cs_read/cs_res*1e3)
			dataset['dataset']['detect_normal'].append(norm_detected)
			
			dataset['dataset']['temperature_K'].append(t_meas)
			dataset['dataset']['times'].append(str(datetime.datetime.now()))
			dataset['dataset']['measurement_stage'].append(measurement_stage)
			
			# Check for autosave
			if time.time() - time_last_save > TIME_AUTOSAVE_S:
				
				# Save data and log
				log.save_hdf(os.path.join(LOG_DIRECTORY, f"{sweep_name}_autosave.log.hdf"))
				
				dict_to_hdf(dataset, os.path.join(DATA_DIRECTORY, f"{sweep_name}_autosave.hdf"))
				log.debug(f"Autosaved logs and data.")
				
				time_last_save = time.time()

if abort:
	log.info(f"Sweep has been aborted. Shutting off signal generator and DC bias.")
else:
	log.info("Sweep completed. Shutting off signal generator and DC bias.")

# sig_gen.set_enable_rf(False)
# sig_gen.set_power(-50)

mfli.set_offset(0)
mfli.set_output_enable(False)

print(f"Saving data...")

# Save data and log
log.save_hdf(os.path.join(LOG_DIRECTORY, f"{sweep_name}.log.hdf"))

dict_to_hdf(dataset, os.path.join(DATA_DIRECTORY, f"{sweep_name}.hdf"))

print(f" -> Data saved. Exiting.")
