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
CONF_DIRECTORY = "sweep_configs"
DATA_DIRECTORY = "data"
LOG_DIRECTORY = "logs"

# Set autosave period in seconds
TIME_AUTOSAVE_S = 600

# Time after going normal to let the chip recover
RECOVERY_TIME_S = 1

# Fraction of set voltage which, if meas voltage is less than, chip is considered to have gone normal
THRESHOLD_NORMAL_V = 0.1

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
	idc_list_mA = interpret_range(conf_data['dc_current_mA'], print_err=True)
	
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
# Configure spectrum analuzer

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
dataset = {"calibration":{"current_sense_res_ohms": cs_res, "time_of_meas": str(cs_time)}, "dataset":{"freq_rf_GHz":[], "power_rf_dBm":[], "times":[], "waveform_f_Hz":[], "waveform_s_dBm":[], "waveform_rbw_Hz":[], "MFLI_V_offset_V":[], "requested_Idc_mA":[], "raw_meas_Vdc_V":[], "Idc_mA":[], "detect_normal":[], "temperature_K":[]}, 'info': {'source_script': __file__, 'operator_notes':operator_notes, 'configuration': json.dumps(conf_data), 'source_script_full':self_contents}}

##======================================================
# Begin logging data

# Prepare DMM to measure voltage
dmm.set_measurement(DigitalMultimeterCtg1.MEAS_VOLT_DC, DigitalMultimeterCtg1.RANGE_AUTO)

# Wait for user to connect through
print(markdown(f"Ready to perform primary sweep. >Reconnect current sense resistor<."))
a = input("Press enter when ready:")

time_last_save = time.time()


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
		
		# Scan over each bias current
		for idc in idc_list_mA:
			
			if abort:
				break
			
			log.info(f"Setting bias current to >{idc}< mA.")
			
			# Set offset voltage
			Voffset = cs_res * idc/1e3
			mfli.set_offset(Voffset)
			mfli.set_output_enable(True)
			
			# Prepare the signal generator
			sig_gen.set_freq(f_rf)
			sig_gen.set_power(p_rf)
			sig_gen.set_enable_rf(True)
			
			# Prepare the spectrum analyzer
			sa_conditions = calc_sa_conditions(sa_conf, f_rf, f_lo=None)
			
			# Read Vdc (-> Idc)
			v_cs_read = np.abs(dmm.trigger_and_read())
			
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
				
				log.debug(f"Measuring frequency range >{fstart/1e6}< MHz to >{fend/1e6}< MHz, RBW = >:q{frbw/1e3}< kHz.")
				
				# count += 1
				# log.debug(f"Beginning measurement {count} of {npts}.")
				
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
					
					# # Find duplicate frequencies
					# dupl_freqs = [k for k,v in Counter(wav_x).items() if v>1]
					
					# # Found duplicates - resolve duplicates
					# if len(dupl_freqs) > 0:
					# 	pass
					
					# Sort result
					wav_f = []
					wav_s = []
					wav_rbw = []
					for wx,wy,wr in sorted(zip(wav_f_temp, wav_s_temp, wav_rbw_temp)):
						wav_f.append(wx)
						wav_s.append(wy)
						wav_rbw.append(wr)
			
			# Check for chip going normal
			if v_cs_read < THRESHOLD_NORMAL_V * Voffset:
				# Chip has gone normal
				log.warning(f"Chip has gone normal. Moving to next power condition. {v_cs_read} > {THRESHOLD_NORMAL_V}*{Voffset}.")
				norm_detected = True
			else:
				log.debug(f"Chip has not gone normal. {v_cs_read} > {THRESHOLD_NORMAL_V}*{Voffset}.")
				norm_detected = False
			
			log.debug(f"Saving datapoint.")
			
			# After all spectrum analyzer data has been captured, save all data to master dataset
			dataset['dataset']['freq_rf_GHz'].append(f_rf/1e9)
			dataset['dataset']['power_rf_dBm'].append(p_rf)
			
			dataset['dataset']['waveform_f_Hz'].append(wav_f)
			dataset['dataset']['waveform_s_dBm'].append(wav_s)
			dataset['dataset']['waveform_rbw_Hz'].append(wav_rbw)
			
			dataset['dataset']['MFLI_V_offset_V'].append(Voffset)
			dataset['dataset']['requested_Idc_mA'].append(idc)
			dataset['dataset']['raw_meas_Vdc_V'].append(v_cs_read)
			dataset['dataset']['Idc_mA'].append(v_cs_read/cs_res*1e3)
			dataset['dataset']['detect_normal'].append(norm_detected)
			
			dataset['dataset']['temperature_K'].append(t_meas)
			dataset['dataset']['times'].append(str(datetime.datetime.now()))
			
			# Check for autosave
			if time.time() - time_last_save > TIME_AUTOSAVE_S:
				
				# Save data and log
				log.save_hdf(os.path.join(LOG_DIRECTORY, f"{sweep_name}_autosave.log.hdf"))
				
				dict_to_hdf(dataset, os.path.join(DATA_DIRECTORY, f"{sweep_name}_autosave.hdf"))
				log.debug(f"Autosaved logs and data.")
				
				time_last_save = time.time()
			
			# Wait for user input - check to quit
			try:
				usr_input = inputimeout(prompt="Enter 'q' to quit (will work next cycle): ", timeout=0.01)
			except TimeoutOccurred:
				usr_input = ''
				
			# Quit if told
			if usr_input.lower() == "q":
				log.info(f"Received exit signal quitting.")
				abort = True
			
			# Check for skip to next power condition
			if norm_detected:
				
				# Let chip recover
				log.info("Turning off DC bias and signal generator to let chip recover")
				sig_gen.set_enable_rf(False)
				mfli.set_offset(0)
				log.debug(f"Waiting {RECOVERY_TIME_S} seconds for chip to recover.")
				time.sleep(RECOVERY_TIME_S)
				log.debug("Post-normal recovery procedure complete. Proceeding to next sweep point.")
				break

if abort:
	log.info(f"Sweep has been aborted. Shutting off signal generator and DC bias.")
else:
	log.info("Sweep completed. Shutting off signal generator and DC bias.")

sig_gen.set_enable_rf(False)
sig_gen.set_power(-50)

mfli.set_offset(0)
mfli.set_output_enable(False)

print(f"Saving data...")

# Save data and log
log.save_hdf(os.path.join(LOG_DIRECTORY, f"{sweep_name}.log.hdf"))

dict_to_hdf(dataset, os.path.join(DATA_DIRECTORY, f"{sweep_name}.hdf"))

print(f" -> Data saved. Exiting.")
