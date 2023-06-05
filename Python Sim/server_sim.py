from core import *
import time
import threading
import socket

#################### SIMULATION OPTIONS ##############################

# Pgen = -4 # dBm
Pgen = 0 # dBm
C_ = 121e-12
l_phys = 0.5
freq = 10e9
q = 0.190

NUM_CLIENTS = 4
Nthread = 2
Nq = 3
NL0 = 4

#######################################################################

########################## CREATE LISTS AND BREAK INTO SUBLISTS ################

# Create L0 lists
L0_sub_lists = []
L0_list = np.linspace(250e-9, 350e-9, NL0)

nL = NL0//Nthread

# Break into sub lists
i = None
for idx in range(Nthread):
	
	# Get slice of list
	i = idx + 1
	L0_list_temp = L0_list[int(nL*(i-1)):int((nL*i))]
	
	# Add to list of sub lists
	L0_sub_lists.append(L0_list_temp)

# Create q list
q_list = np.linspace(0.18, 0.2, Nq)

print(f"Will create {Nthread} threads. Number of simulations per thread:")
for idx in range(Nthread):
	npts = len(L0_sub_lists[idx])*len(q_list)
	print(f"\t[thread {idx}]: {npts} sims")

##########################################################################

########################## LAUNCH SERVER ##################################

client_idx = 0
id_str = "MIAN"

def main():
	global next_thread_id, sock, server_opt
	
	threads = []
	
	# Loop accept client connections
	while client_idx < NUM_CLIENTS:
				
		# Accept a new client connection
		try:
			client_socket, client_addr = sock.accept()
		except socket.timeout:
			logging.info(f"{id_str}Timed out waiting for client")
			continue
		logging.info(f"{id_str}Accepted client connected on address <{client_addr}>")
		
		# Create server agent class
		sa = ServerAgent(L0_sub_lists[client_idx], q_list, Pgen, C_, l_phys, freq)
		sa.append(sa)
		
		# Increment counter
		client_idx += 1
		
		 # Update thread_id
		next_thread_id += 1
			
		# Start client thread
		sa.start()
	
	logging.info(f"{id_str}Server shutting down")
	server_opt.kill_stat_thread = True
	server_opt.kill_distribution_thread = True
	
	tf = time.time()

	# Print stats
	print(f"Finished sweep in {tf-t0} seconds")

	npoints = 0
	for idx in range(Nthread):
		npoints += len(L0_sub_lists[idx])*len(q_list)

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

	plt.figure(1)
	plt.plot(Idc_A, Ifund_mA, color='r', label="Measurement", linestyle='dashed', marker='o')
	plt.plot(Idc_A, 1e3*master_Iac[idx_best1], color='b', label=f"Sim (L0={rd(L0_best1*1e9,1)} nH) (q = {q_best1} A)", linestyle='dashed', marker='+')
	plt.plot(Idc_A, 1e3*master_Iac[idx_best2]*coef_best, color=(0.35, 0, 0.6), label=f"Scaled Sim (L0={rd(L0_best2*1e9, 1)} nH) (q = {q_best2} A)", linestyle='dashed', marker='x')
	plt.grid()
	plt.xlabel("Bias Current (mA)")
	plt.ylabel("AC Current Amplitude (mA)")
	plt.title(f"Fundamental - Closest fit")
	plt.legend()

	plt.figure(2)
	plt.plot(Idc_A, 1e3*I2h, color='r', label="Measurement", linestyle='dashed', marker='o')
	plt.plot(Idc_A, 1e3*master_I2h[idx_best1], color='b', label=f"Sim (L0={rd(L0_best1*1e9,1)} nH) (q = {q_best1} A)", linestyle='dashed', marker='+')
	plt.grid()
	plt.xlabel("Bias Current (mA)")
	plt.ylabel("AC Current Amplitude (mA)")
	plt.title(f"2nd Harmonics - Closest fit")
	plt.legend()

	print(f"Scaling coefficient used in plot: {coef_best}")

	# Save data to Pickle for further analysis
	master_data = {"Iac_results":master_Iac, "I2h_results":master_I2h, "rmse1_results":master_rmse1, "rmse2_results":master_rmse2, "rmse3_results":master_rmse3, "Idc_A":Idc_A, "coefs":master_coefs, "conditions":master_conditions}
	with open("harm_sim_last_data.pkl", 'wb') as f:
		pickle.dump(master_data, f)

	plt.show()
	

if __name__ == "__main__":
	
	main()