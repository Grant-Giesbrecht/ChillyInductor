import matplotlib.pyplot as plt
import numpy as np

pi_opt = np.array([0.965, 0.97, 0.975, 0.98, 0.985])
erate = np.array([0.017571, 1.67e-2, 1.784e-2, 0.162e-2, 1.655e-2]) # Check 0.97
erate_new = np.array([0.017571, 1.67e-2, 1.71e-2, 1.62e-2, 1.655e-2]) # Check 0.97

plt.plot(1e2*(0.975-pi_opt), erate_new, linestyle=':', marker='+', color=(0.75, 0, 0))
# plt.ylim([0, 1e-2])
plt.grid(True)
plt.xlabel("Optimization: $\\Delta$ From Ideal (%)")
plt.ylabel("Error Rate")

plt.show()

