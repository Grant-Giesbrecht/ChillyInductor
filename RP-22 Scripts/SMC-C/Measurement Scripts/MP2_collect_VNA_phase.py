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
from jarnsaxa import dict_to_hdf

class SystemRP22:
	
	def __init__(self, dmm, tempctrl, mfli, vna):
		self.dmm = dmm
		self.tempctrl = tempctrl
		self.mfli = mfli
		self.vna = vna
		
		# Define dataset dictionary
		self.calibration_dict = {"current_sense_res_ohms": cs_res, "time_of_meas": str(cs_time)}
		self.dataset_dict = {"freq_rf_GHz":[], "power_rf_dBm":[], "times":[], "MFLI_V_offset_V":[], "requested_Idc_mA":[], "raw_meas_Vdc_V":[], "Idc_mA":[], "detect_normal":[], "temperature_K":[], "waveform_f_Hz":[], "waveform_s_dBm":[], "waveform_rbw_Hz":[], "measurement_stage":[]}

		self.summary_dict = {'Ic_meas_mA':[], "Ic_set_mA":[], "Ic_uncert_mA": [], "power_rf_dBm":[], "freq_rf_GHz":[], "times":[], "temperature_K":[]}

		self.info_dict = {'source_script': __file__, 'operator_notes':operator_notes, 'configuration': json.dumps(conf_data), 'source_script_full':self_contents}

		self.dataset = {"calibration":self.calibration_dict, "dataset":self.dataset_dict, "summary_dataset":self.summary_dict, 'info':self.info_dict}
	
	def autosave(self):
		# Save data and log
		log.save_hdf(os.path.join(LOG_DIRECTORY, f"{sweep_name}_autosave.log.hdf"))
		
		dict_to_hdf(dataset, os.path.join(DATA_DIRECTORY, f"{sweep_name}_autosave.hdf"))
		log.debug(f"Autosaved logs and data.")
	
	def measure_point(self, f_rf, p_rf, idc):
		
		log.info(f"Setting bias current to >{idc}< mA.")
		
		# Set offset voltage
		Voffset = current_set_res * idc/1e3
		log.debug(f"Selected offset voltage {Voffset} V from current_set_res={current_set_res} Ohms and idc={idc} mA")
		mfli.set_offset(Voffset)
		mfli.set_output_enable(True)
		
		# Prepare the VNA
		vna.set_freq_start(f_rf)
		vna.set_freq_end(f_rf)
		vna.set_power(p_rf)
		vna.set_rf_enable(True)
		
		# Read Vdc (-> Idc)
		v_cs_read = np.abs(dmm.trigger_and_read())
		
		log.info(f"Measured bias current at {v_cs_read/cs_res*1e3} mA", detail=f"V_current_sense_read = {v_cs_read} V, CS-resistor = {cs_res} ohms.")
		if (idc != 0) and (np.abs((v_cs_read/cs_res*1e3) - idc)/idc > 0.2):
			log.warning(f"Measured Idc missed target by more than 20%. (Error = {np.abs((v_cs_read/cs_res*1e3) - idc)/idc*100} %)")
		
		# Read temperature
		t_meas = tempctrl.get_temp()
		
		vna.send_manual_trigger()
		vna.get_channel_data()
		
		# Check for chip going normal
		if v_cs_read < THRESHOLD_NORMAL_V * Voffset:
			# Chip has gone normal
			log.warning(f"Chip has gone normal. Moving to next power condition. {v_cs_read} > {THRESHOLD_NORMAL_V}*{Voffset}.")
			norm_detected = True
		else:
			log.debug(f"Chip has not gone normal. {v_cs_read} > {THRESHOLD_NORMAL_V}*{Voffset}.")
			norm_detected = False
		
		self.dataset['dataset']['freq_rf_GHz'].append(f_rf/1e9)
		self.dataset['dataset']['power_rf_dBm'].append(p_rf)
		
		self.dataset['dataset']['MFLI_V_offset_V'].append(Voffset)
		self.dataset['dataset']['requested_Idc_mA'].append(idc)
		self.dataset['dataset']['raw_meas_Vdc_V'].append(v_cs_read)
		self.dataset['dataset']['Idc_mA'].append(v_cs_read/cs_res*1e3)
		self.dataset['dataset']['detect_normal'].append(norm_detected)
		
		self.dataset['dataset']['temperature_K'].append(t_meas)
		self.dataset['dataset']['times'].append(str(datetime.datetime.now()))
		
		self.dataset['dataset']
		
	def recover_normal(self):
		
		# Let chip recover
		log.info("Turning off DC bias and signal generator to let chip recover")
		
		vna.set_enable_rf(False)
		mfli.set_offset(0)
		
		log.debug(f"Waiting {RECOVERY_TIME_S} seconds for chip to recover.")
		time.sleep(RECOVERY_TIME_S)
		
		log.debug("Post-normal recovery procedure complete. Proceeding to next sweep point.")
	

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
parser.add_argument('--loglevel', help="Set the logging display level.", choices=['LOWDEBUG', 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], type=str.upper)
parser.add_argument('--tempwait', help="Set the logging display level.", type=float)
args = parser.parse_args()

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

##======================================================
# Interpret data

try:
	freq_rf = interpret_range(conf_data['frequency_RF'], print_err=True)
	power_rf_dBm = interpret_range(conf_data['power_RF'], print_err=True)
	
	extra_z_norm = float(conf_data['extra_Z_norm_ohm'])
	idc_points = int(conf_data['Idc_points'])
	
	vna_rbw = conf_data['VNA']['resolution_bw_Hz']
	vna_npoints = conf_data['VNA']['num_points']
except:
	log.critical(f"Corrupt configuration file >{conf_file_name}<.")
	exit()

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

vna = RohdeSchwarzZVA("TCPIP0::something:INSTR", log)
if vna.online:
	log.info("Rohde & Schwarz ZVA >ONLINE<")
else:
	log.critical("Failed to connect to Rohde & Schwarz ZVA!")
	exit()

# Initialize system object
meas_sys = SystemRP22(dmm, tempctrl, mfli, vna)

##======================================================
# Get User input re: sweep

sweep_name = input("Sweep name: ")
operator_notes = input("Operator notes: ")
while True:
	
	# Get user input
	try:
		Ic_expected_mA = float(input("Expected critical current (mA): "))
	except Exception as e:
		print(f"Failed to interpret data. ({e})")
		continue
	
	# Verify units
	if (Ic_expected_mA < 0.1) or (Ic_expected_mA > 10):
		print(f"{Fore.RED}CHECK UNITS:{Style.RESET_ALL} Expected critical current set to {Ic_expected_mA} mA.")
		confirm_string = input(f"Y to confirm, N to re-enter:")
		if confirm_string != "Y":
			continue
	
	log.info(f"Set expected critical current to >{Ic_expected_mA}< mA.")
	break

##======================================================
# Configure VNA

# Reset VNA
vna.send_preset()
vna.clear_traces()

# Add trace and initialize with reasonable values
vna.add_trace(1, 1, VectorNetworkAnalyzerCtg1.MEAS_S21)
vna.set_freq_start(1e9)
vna.set_freq_end(1e9)
vna.set_power(-50)
vna.set_num_points(vna_npoints)
vna.set_res_bandwidth(vna_rbw)
vna.set_rf_enable(False)
vna.set_continuous_trigger(False)
vna.set_averaging_enable(False)

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
# Begin logging data

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

# Pick Idc list
idc_list_mA = np.linspace(0, Ic_expected_mA, num=idc_points)
log.debug(f"Selected Idc values: {idc_list_mA}")

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
			
			# Set instruments and measure point
			SystemRP22.measure_point(f_rf, p_rf, idc)
			
			# Check for autosave
			if time.time() - time_last_save > TIME_AUTOSAVE_S:
				SystemRP22.autosave()
				time_last_save = time.time()
			
			# Check for skip to next power condition
			if norm_detected:
				SystemRP22.recover_normal()
				break

if abort:
	log.info(f"Sweep has been aborted. Shutting off signal generator and DC bias.")
else:
	log.info("Sweep completed. Shutting off signal generator and DC bias.")

vna.set_enable_rf(False)
vna.set_power(-50)

mfli.set_offset(0)
mfli.set_output_enable(False)

print(f"Saving data...")

# Save data and log
log.save_hdf(os.path.join(LOG_DIRECTORY, f"{sweep_name}.log.hdf"))

dict_to_hdf(dataset, os.path.join(DATA_DIRECTORY, f"{sweep_name}.hdf"))

print(f" -> Data saved. Exiting.")
