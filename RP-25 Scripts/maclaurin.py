import numpy as np
import matplotlib.pyplot as plt

A = 130e-18
B = 65e-18
# x = np.array([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1])
x = np.logspace(-6, 1, 81)
v0 = 3e8

coef_0 = v0*np.sqrt(A)
coef_1 = v0*B/2/np.sqrt(A)
coef_2 = v0*B**2/8/A**1.5
coef_3 = v0*B**3/16/A**2.5

n_elec = v0*np.sqrt(A+B*x)
n_elec_approx = coef_0 + coef_1*x # - coef_2*x**2 #+ coef_3*x**3

print(f"n_elec = {coef_0} + {coef_1}*x - {coef_2}*x^2 + {coef_3}*x^3")

plt.semilogx(x, n_elec_approx, label=f"Approx.")
plt.semilogx(x, n_elec, label=f"Exact")
plt.xlabel("Power (W)")
plt.ylabel("$n_{elec}$")

plt.legend()
plt.grid(True)
plt.show()