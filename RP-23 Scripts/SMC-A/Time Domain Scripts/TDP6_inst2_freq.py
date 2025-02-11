import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert, chirp

DATADIR = "G:\ARC0 PhD Data\RP-23 Qubit Readout\Data\SMC-A\Time Domain Measurements"

duration, fs = 1, 400  # 1 s signal with sampling frequency of 400 Hz
t = np.arange(int(fs*duration)) / fs  # timestamps of samples
signal = chirp(t, 20.0, t[-1], 100.0)
signal *= (1.0 + 0.5 * np.sin(2.0*np.pi*3.0*t) )

analytic_signal = hilbert(signal)
amplitude_envelope = np.abs(analytic_signal)
instantaneous_phase = np.unwrap(np.angle(analytic_signal))
instantaneous_frequency = np.diff(instantaneous_phase) / (2.0*np.pi) * fs

fig, (ax0, ax1) = plt.subplots(nrows=2, sharex='all', tight_layout=True)
ax0.set_title("Amplitude-modulated Chirp Signal")
ax0.set_ylabel("Amplitude")
ax0.plot(t, signal, label='Signal')
ax0.plot(t, amplitude_envelope, label='Envelope')
ax0.legend()
ax1.set(xlabel="Time in seconds", ylabel="Phase in rad", ylim=(0, 120))
ax1.plot(t[1:], instantaneous_frequency, 'C2-', label='Instantaneous Phase')
ax1.legend()
plt.show()