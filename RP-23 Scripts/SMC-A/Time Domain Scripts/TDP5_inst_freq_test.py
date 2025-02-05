import numpy as np
from scipy.signal import hilbert
import matplotlib.pyplot as plt

# Generate a sample signal (e.g., a chirp signal)
t = np.linspace(0, 1, 1000)
signal = np.sin(2 * np.pi * (5 * t + 10 * t**2))

freq_ans = (5 + 10*t)

# Compute the analytic signal using the Hilbert transform
analytic_signal = hilbert(signal)
instantaneous_phase = np.unwrap(np.angle(analytic_signal))
instantaneous_frequency = np.diff(instantaneous_phase) / (2.0 * np.pi * np.diff(t))

# Plot the results
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(t, signal, label='Original Signal')
plt.title('Original Signal')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(t[1:], instantaneous_frequency, label='Instantaneous Frequency')
plt.plot(t[1:], freq_ans[1:], label='Actual Frequency')
plt.title('Instantaneous Frequency')
plt.xlabel('Time [s]')
plt.ylabel('Frequency [Hz]')
plt.legend()

plt.tight_layout()
plt.show()
