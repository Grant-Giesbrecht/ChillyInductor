from core import *
import time

Pgen = 0 # dBm
C_ = 121e-12
l_phys = 0.5
freq = 10e9
q = 0.190

# Create array of L0 values to try
L0_list = np.linspace(300e-9, 500e-9, 11)

# Get target values from MATLAB
Idc_A = np.array([-0.03325, -0.030875, -0.0285, -0.026125, -0.02375, -0.021375, -0.019, -0.016625, -0.01425, -0.011875, -0.0095, -0.007125, -0.00475, -0.002375, 0, 0.002375, 0.00475, 0.007125, 0.0095, 0.011875, 0.01425, 0.016625, 0.019, 0.021375, 0.02375, 0.026125, 0.0285, 0.030875, 0.03325])
Ifund_mA = np.array([1.5131, 1.6246, 1.4932, 1.6711, 1.7109, 1.817, 1.7177, 1.6502, 1.6852, 1.7652, 1.8488, 1.9048, 1.9497, 1.9705, 1.9756, 1.9717, 1.9531, 1.9116, 1.8525, 1.7779, 1.6928, 1.6577, 1.7247, 1.8184, 1.7306, 1.6805, 1.5131, 1.6378, 1.5375])
Ifund_A = Ifund_mA/1e3

mid_idx = int(np.floor(len(Idc_A)/2))

# Create array of L0 values to try
L0_list = np.linspace(1e-9, 600e-9, 5)

# Create colormapper
cm = CMap('plasma', data=L0_list)

t0 = time.time()

# Loop over all L0 values
Iac_results = []
rmse1_results = []
rmse2_results = []
coefs = []
for idx, L0 in enumerate(L0_list):
	
	t0_ = time.time()
	
	# Report scan iteration
	logging.main(f"{Fore.LIGHTBLUE_EX}Beginning simulation of L0 = {rd(L0*1e9,1)} nH{standard_color}")
	
	# Prepare simualtion
	lks = LKSystem(Pgen, C_, l_phys, freq, q, L0)
	lks.opt.start_guess_method = GUESS_USE_LAST
	lks.solve(Idc_A, show_plot_on_conv=False)
	
	# Get results
	Iac = np.array([x.Iac for x in lks.solution])
	
	# Calculate error basic error
	rmse1 = np.sqrt( np.mean( np.abs(Ifund_A - Iac)**2 ) )
	
	# Calculate scaled error
	coef = Ifund_A[mid_idx]/Iac[mid_idx]
	Iac_scaled = Iac * coef
	rmse2 = np.sqrt( np.mean( np.abs(Ifund_A - Iac_scaled)**2 ) )
	
	# Add to output data
	Iac_results.append(Iac)
	rmse1_results.append(rmse1)
	rmse2_results.append(rmse2)
	coefs.append(coef)
	
	# Report stats and estimated completion time
	logging.main(f"[RMSE = {Fore.LIGHTRED_EX}{rd(rmse1*1e3,3)}m{standard_color}]. Minimum RMSE = {Fore.LIGHTRED_EX}{rd(np.min(rmse1_results)*1e3,3)}m{standard_color}].")
	if idx > 0:
		iter_per_sec = idx/(time.time()-t0)
		t_est = (len(L0_list)-idx)/iter_per_sec
		logging.main(f"Iteration time = {Fore.LIGHTRED_EX}{rd(time.time()-t0_,1)}{standard_color} sec. Est time remaining = {Fore.LIGHTRED_EX}{rd(t_est, 2)}{standard_color} sec.")

# Find minimum error 
idx_best1 = rmse1_results.index(min(rmse1_results))
idx_best2 = rmse2_results.index(min(rmse2_results))
L0_best1 = L0_list[idx_best1]
L0_best2 = L0_list[idx_best2]

plt.plot(Idc_A, Ifund_mA, color='r', label="Measurement", linestyle='dashed', marker='o')
plt.plot(Idc_A, 1e3*Iac_results[idx_best1], color='b', label=f"Sim (L0={rd(L0_best1*1e9,1)} nH)", linestyle='dashed', marker='o')
plt.plot(Idc_A, 1e3*Iac_results[idx_best2], color='g', label=f"Scaled Sim (L0={rd(L0_best2*1e9, 1)} nH)", linestyle='dashed', marker='o')
plt.grid()
plt.xlabel("Bias Current (mA)")
plt.ylabel("AC Current Amplitude (mA)")
plt.title(f"Closest fit")
plt.legend()

# print(f"L0 values: {L0_best}")

plt.show()