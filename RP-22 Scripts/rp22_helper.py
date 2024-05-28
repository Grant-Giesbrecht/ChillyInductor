

def calc_sa_conditions(sa_conf, f_rf:float, f_lo:float, print_error:bool=False, remove_duplicates:bool=True) -> list:
	''' Generates a list of dictionsaries with keys:
		rbw: Resolution bandwidth in Hz
		f_start: Start frequency in Hz
		f_end: End frequency in Hz
	for a given set of spectrum analyzer configution blocks.
	
	--------------------------------------------------------
	Configuration example block:
	"spectrum_analyzer":{
		[
			"mode": "dynamic",
			"span_Hz": 30e3,
			"RBW_Hz": 1e3,
			"mixing_products_order": 2,
			"lo_harmonics": 5,
			"rf_harmonics": 2
		],
		[
			"mode": "fixed",
			"freq_start_Hz": 100e6,
			"freq_end_Hz": 6e9,
			"RBW_Hz": 100e3
		]
	}
	
	Returns None if error, otherwise returns list of dictionaries
	'''
	
	# If sa_conf input is a list, recursively call each
	if type(sa_conf) is list:
				
		# Iterate over each element of list
		sacl = None
		for sacb in sa_conf:
			if sacl is None:
				sacl = calc_sa_conditions(sacb, f_rf, f_lo, print_error=print_error, remove_duplicates=remove_duplicates)
			else:
				sacl = sacl + calc_sa_conditions(sacb, f_rf, f_lo, print_error=print_error, remove_duplicates=remove_duplicates)
				
		return sacl
	
	# Otherwise generate the dictionary list from this block
	elif type(sa_conf) is dict:
		
		# Generate list of measurement regions
		sac = []
		
		# Validate conf
		if 'mode' not in sa_conf.keys():
			if print_error:
				print(f"Missing key 'mode'.")
			return None
		
		# Split into modes
		if sa_conf['mode'].lower() == "dynamic":
			
			# Validate conf
			test_key_list = ['span_Hz', 'RBW_Hz', 'mixing_products_order', 'lo_harmonics', 'rf_harmonics']
			if not all(test_key in sa_conf for test_key in test_key_list):
				if print_error:
					print(f"Missing keys")
				return None
			
			# Interpret conf
			try:
				lo_harmonics = int(sa_conf['lo_harmonics'])
				rf_harmonics = int(sa_conf['rf_harmonics'])
				mixing_products_order = int(sa_conf['mixing_products_order'])
				RBW_Hz = int(sa_conf['RBW_Hz'])
				span_Hz = int(sa_conf['span_Hz'])
			except Exception as e:
				if print_error:
					print(f"Invalid data in configuration block ({e}).")
				return None
			
			# Create conditions for each LO harmonics
			for i in range(1, 1+lo_harmonics):
				
				# Get frequencies
				f_center = f_lo*i
				f_start = f_center - span_Hz/2
				f_end = f_start + span_Hz
				
				# Create conf dictionary
				cd = {'rbw':RBW_Hz, 'f_start':f_start, 'f_end':f_end}
				
				# Add to list
				sac.append(cd)
			
			# Create conditions for each RF harmonics
			for i in range(1, 1+rf_harmonics):
				
				# Get frequencies
				f_center = f_rf*i
				f_start = f_center - span_Hz/2
				f_end = f_start + span_Hz
				
				# Create conf dictionary
				cd = {'rbw':RBW_Hz, 'f_start':f_start, 'f_end':f_end}
				
				# Add to list
				sac.append(cd)
			
			## Create conditions for mixing products
			# Get center frequencies
			cfl = []
			for i in range(1, 1+mixing_products_order):
				cfl.append(f_rf + i*f_lo)
				cfl.append(f_rf - i*f_lo)
			
			## Create conditions for mixing products
			# Generate conditions
			for f_center in cfl:
				
				# Get frequencies
				f_start = f_center - span_Hz/2
				f_end = f_start + span_Hz
				
				# Create conf dictionary
				cd = {'rbw':RBW_Hz, 'f_start':f_start, 'f_end':f_end}
				
				# Add to list
				sac.append(cd)
				
		else:
			
			# Validate conf
			test_key_list = ['freq_start_Hz', 'RBW_Hz', 'freq_end_Hz']
			if not all(test_key in sa_conf for test_key in test_key_list):
				if print_error:
					print(f"Missing keys")
				return None
			
			# Interpret conf
			try:
				freq_start_Hz = int(sa_conf['freq_start_Hz'])
				freq_end_Hz = int(sa_conf['freq_end_Hz'])
				RBW_Hz = int(sa_conf['RBW_Hz'])
			except Exception as e:
				if print_error:
					print(f"Invalid data in configuration block ({e}).")
				return None
			
			# Generate condition
			cd = {'rbw':RBW_Hz, 'f_start':freq_start_Hz, 'f_end':freq_end_Hz}
			
			# Add to list
			sac.append(cd)
			
		# Return spectrum analyzer condition list
		return sac
		
	# Otherwise invalid type
	else:
		if print_error:
			print(f"Invalid data type")
		return None
	