import matplotlib.pyplot as plt

Vbias = [0.2, 0.3, 0.4, 0.5, 0.6, 0.8]
erate = [3.7452e-3, 5.8679e-3, 5.008e-3, 3.614e-3, 3.705e-3, 5.9808e-3]

plt.plot(Vbias, erate, linestyle=':', marker='+', color=(0.75, 0, 0))
plt.ylim([0, 1e-2])
plt.grid(True)
plt.xlabel("Bias Voltage")
plt.ylabel("Error Rate")

plt.show()