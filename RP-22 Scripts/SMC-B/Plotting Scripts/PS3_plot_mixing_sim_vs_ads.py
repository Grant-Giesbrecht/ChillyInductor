import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.family'] = 'Aptos'

current_mA_meas = [0.049477, 0.62933, 1.2255, 1.8416, 2.4026, 2.9788, 3.5918, 4.1775, 4.7627, 5.3638, 5.9295, 6.5237, 7.1274, 7.7013, 8.2815, 8.8733]
current_mA_sim = np.array([ 2.434e-17, 0.495, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10])
mx1h_meas = [-60.3831, -44.9189, -38.8115, -35.2555, -32.7306, -30.8298, -29.2586, -27.8516, -26.6879, -25.5979, -24.7294, -23.872, -23.1157, -22.4057, -21.7236, -21.1034]
mx1h_sim = np.array([ -136.327, -43.598, -37.576, -34.052, -31.55, -29.608, -28.019, -26.674, -25.507, -24.475, -23.55, -22.711, -21.943, -21.234, -20.574, -19.958, -19.378, -18.831, -18.312, -17.819, -17.348])

sim_unique = np.array([True, True, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True])


plt.figure(1)
plt.plot(current_mA_meas, mx1h_meas, linestyle=':', marker='s', color=(0.7, 0.2, 0.2), label="Measured")
plt.plot(current_mA_sim, mx1h_sim, linestyle=':', marker='o', color=(0.2, 0.2, 0.65), label="Simulated")
plt.grid(True)
plt.xlabel("Bias Current (mA)")
plt.ylabel("Mixing Gain (dB)")
plt.ylim([-65, -10])
plt.legend()

plt.figure(2, figsize=(6, 4.5))
plt.plot(current_mA_meas, mx1h_meas, linestyle=':', marker='s', color=(0.7, 0.2, 0.2), label="Measured")
plt.plot(current_mA_sim[sim_unique], mx1h_sim[sim_unique], linestyle=':', marker='o', color=(0.2, 0.2, 0.65), label="Simulated")
plt.grid(True)
plt.xlabel("Bias Current (mA)")
plt.ylabel("Mixing Gain (dB)")
plt.ylim([-65, -10])
plt.legend()
plt.title("Mixing Gain, $P(f_{RF} - f_{LO}) - P(f_{RF})$")

plt.tight_layout()

plt.savefig("PS3_fig2.png", dpi=700)

plt.show()