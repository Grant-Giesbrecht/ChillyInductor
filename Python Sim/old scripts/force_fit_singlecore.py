from core import *
import time
import threading

# Get target values from MATLAB
Idc_A = np.array([-0.03325, -0.030875, -0.0285, -0.026125, -0.02375, -0.021375, -0.019, -0.016625, -0.01425, -0.011875, -0.0095, -0.007125, -0.00475, -0.002375, 0, 0.002375, 0.00475, 0.007125, 0.0095, 0.011875, 0.01425, 0.016625, 0.019, 0.021375, 0.02375, 0.026125, 0.0285, 0.030875, 0.03325])
Ifund_mA = np.array([1.5131, 1.6246, 1.4932, 1.6711, 1.7109, 1.817, 1.7177, 1.6502, 1.6852, 1.7652, 1.8488, 1.9048, 1.9497, 1.9705, 1.9756, 1.9717, 1.9531, 1.9116, 1.8525, 1.7779, 1.6928, 1.6577, 1.7247, 1.8184, 1.7306, 1.6805, 1.5131, 1.6378, 1.5375])
Ifund_A = Ifund_mA/1e3

mid_idx = int(np.floor(len(Idc_A)/2))

# Pgen = -4 # dBm
Pgen = 0 # dBm
C_ = 121e-12
l_phys = 0.5
freq = 10e9
q = 0.190

Nthread = 4
Nq = 10
NL0 = 16
NL01 = NL0//Nthread
NL02 = NL0//Nthread
NL03 = NL0//Nthread
NL04 = NL0 - NL01 - NL02 - NL03


# Create array of L0 values to try
L0_list = np.linspace(1.0e-9, 2.0e-9, NL0)
q_list = np.linspace(0.18, 0.2, Nq)

# Global data
master_mutex = threading.Lock()
master_Iac = []
master_rmse1 = []
master_rmse2 = []
master_coefs = []
master_conditions = []

class SimBasic():
	
	def __init__(self, L0_list:list, q_list:list, Pgen, C_, l_phys, freq):
		
		super().__init__()
		
		self.L0_list = L0_list
		self.q_list = q_list
		
		self.Pgen = Pgen
		self.C_ = C_
		self.l_phys = l_phys
		self.freq = freq
	
	def run(self):
		global master_mutex, master_coefs, master_conditions, master_Iac, master_rmse1, master_rmse2
		global Idc_A
		
		logging.main(f"{Fore.GREEN}[TID={hex(threading.get_ident())}]{standard_color} Beginning sweep.")
		
		# Create array of bias values
		Ibias = Idc_A
		
		Iac_results = []
		rmse1_results = []
		rmse2_results = []
		coefs = []
		conditions = []
		
		for idx, L0 in enumerate(self.L0_list):
			for idx2, q in enumerate(self.q_list):
				
				logging.main(f"{Fore.GREEN}[TID={hex(threading.get_ident())}]{Fore.LIGHTBLUE_EX}Beginning simulation of L0 = {L0*1e9} nH{standard_color}")
				
				lks = LKSystem(Pgen, C_, l_phys, freq, q, L0)
				lks.opt.start_guess_method = GUESS_USE_LAST
				lks.solve(Ibias, show_plot_on_conv=False)
				
				# Calculate current array
				Iac = np.array([x.Iac for x in lks.solution])
				
				# Calculate error basic error
				rmse1 = np.sqrt( np.mean( np.abs(Ifund_A - Iac)**2 ) )
				
				# Calculate auto-scaled error
				coef = Ifund_A[mid_idx]/Iac[mid_idx]
				Iac_scaled = Iac * coef
				rmse2 = np.sqrt( np.mean( np.abs(Ifund_A - Iac_scaled)**2 ) )
				
				# Add to output data
				Iac_results.append(Iac)
				rmse1_results.append(rmse1)
				rmse2_results.append(rmse2)
				coefs.append(coef)
				conditions.append( (q, L0) )
		
		logging.main(f"{Fore.GREEN}[TID={hex(threading.get_ident())}]{standard_color} Finished simulation sweep.")
		
		with master_mutex:
			logging.main(f"{Fore.GREEN}[TID={hex(threading.get_ident())}]{standard_color} Acquired mutex. Saving to global data arrays")
			master_Iac.extend(Iac_results)
			master_rmse1.extend(rmse1_results)
			master_rmse2.extend(rmse2_results)
			master_coefs.extend(coefs)
			master_conditions.extend(conditions)
		
		logging.main(f"{Fore.GREEN}[TID={hex(threading.get_ident())}]{standard_color} Exiting")

# Create threads
st1 = SimBasic(L0_list, q_list, Pgen, C_, l_phys, freq)

t0 = time.time()

# Begin all threads
st1.run()

# Wait for threads to complete

tf = time.time()

# Print stats
print(f"Finished sweep in {tf-t0} seconds")
npoints = len(L0_list)*len(q_list)
print(f"Number of sweep points: {npoints}")

# Find minimum error 
idx_best1 = master_rmse1.index(min(master_rmse1))
idx_best2 = master_rmse2.index(min(master_rmse2))

# Get L0 conditions
L0_best1 = master_conditions[idx_best1][1]
L0_best2 = master_conditions[idx_best2][1]

# Get Pg conditions
q_best1 = master_conditions[idx_best1][0]
q_best2 = master_conditions[idx_best2][0]

coef_best = master_coefs[idx_best2]

plt.plot(Idc_A, Ifund_mA, color='r', label="Measurement", linestyle='dashed', marker='o')
plt.plot(Idc_A, 1e3*master_Iac[idx_best1], color='b', label=f"Sim (L0={rd(L0_best1*1e9,1)} nH) (q = {q_best1} A)", linestyle='dashed', marker='o')
plt.plot(Idc_A, 1e3*master_Iac[idx_best2]*coef_best, color='g', label=f"Scaled Sim (L0={rd(L0_best2*1e9, 1)} nH) (q = {q_best2} A)", linestyle='dashed', marker='o')
plt.grid()
plt.xlabel("Bias Current (mA)")
plt.ylabel("AC Current Amplitude (mA)")
plt.title(f"Closest fit")
plt.legend()

print(f"Scaling coefficient used in plot: {coef_best}")

# Save data to Pickle for further analysis
master_data = {"Iac_results":master_Iac, "rmse1_results":master_rmse1, "rmse2_results":master_rmse2, "Idc_A":Idc_A, "coefs":master_coefs, "conditions":master_conditions}
with open("multithread_sim_last_data.pkl", 'wb') as f:
	pickle.dump(master_data, f)

plt.show()
























# from core import *
# import time
# import threading

# # Pgen = -4 # dBm
# Pgen = 0 # dBm
# C_ = 121e-12
# l_phys = 0.5
# freq = 10e9
# q = 0.190

# Nthread = 4
# Nq = 2
# NL0 = 8
# NL01 = NL0//Nthread
# NL02 = NL0//Nthread
# NL03 = NL0//Nthread
# NL04 = NL0 - NL01 - NL02 - NL03


# # Create array of L0 values to try
# L0_list = np.linspace(1.0e-9, 2.0e-9, NL0)
# q_list = np.linspace(0.18, 0.2, 5)

# # Global data
# master_mutex = threading.Lock()
# master_data = []
# master_conditions = []

# class SimBasic():
	
# 	def __init__(self, L0_list:list, q_list:list, Pgen, C_, l_phys, freq):
		
# 		self.L0_list = L0_list
# 		self.q_list = q_list
		
# 		self.Pgen = Pgen
# 		self.C_ = C_
# 		self.l_phys = l_phys
# 		self.freq = freq
	
# 	def run(self):
# 		global master_data, master_mutex
		
# 		logging.main(f"{Fore.GREEN}[TID={threading.get_ident()}]{standard_color} Beginning sweep.")
		
# 		# Create array of bias values
# 		Ibias = np.linspace(1, 33, 34)*1e-3

# 		# cm = CMap('plasma', data=L0_list)
# 		cm = CMap('cividis', data=self.L0_list)
# 		results = []
# 		conditions = []
# 		for idx, L0 in enumerate(self.L0_list):
# 			for idx2, q in enumerate(self.q_list):
			
# 				logging.info(f"{Fore.GREEN}[TID={threading.get_ident()}]{Fore.LIGHTBLUE_EX}Beginning simulation of L0 = {L0*1e9} nH{standard_color}")

# 				lks = LKSystem(Pgen, C_, l_phys, freq, q, L0)
# 				lks.opt.start_guess_method = GUESS_USE_LAST
# 				lks.solve(Ibias, show_plot_on_conv=False)

# 				Iac = np.array([x.Iac for x in lks.solution])
				
# 				results.append( Iac )
# 				conditions.append( (L0, q) )

# 				# plt.plot(Ibias*1e3, Iac*1e3, label=f"{rd(L0*1e9,2)} nH", color=cm(idx))
		
# 		logging.main(f"{Fore.GREEN}[TID={threading.get_ident()}]{standard_color} Finished sweep. Saving data")
		
# 		with master_mutex:
			
# 			master_data.extend(results)
# 			master_conditions.extend(results)

# t0 = time.time()

# st4 = SimBasic(L0_list, q_list, Pgen, C_, l_phys, freq)
# st4.run()

# tf = time.time()

# print(f"Finished sweep in {tf-t0} seconds")

# npoints = len(L0_list)*len(q_list)
# print(f"Number of sweep points: {npoints}")
# # plt.grid()
# # plt.xlabel("Bias Current (mA)")
# # plt.ylabel("AC Current Amplitude (mA)")
# # plt.title("Simulation Summary")
# # plt.legend()

# # print(f"L0 values: {L0_list}")

# # plt.show()