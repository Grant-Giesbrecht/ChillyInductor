from RP23SMCD import *

file_trad = os.path.join("G:\\", "ARC0 PhD Data", "RP-23 Qubit Readout", "Data", "SMC-D Med Trace Campaign", "Time Domain Measurements", "C1RP23Dset2_f07_00000.txt")
file_doubler = os.path.join("G:\\", "ARC0 PhD Data", "RP-23 Qubit Readout", "Data", "SMC-D Med Trace Campaign", "Time Domain Measurements", "C1RP23Dset2_f10_00000.txt")
file_tripler = os.path.join("G:\\", "ARC0 PhD Data", "RP-23 Qubit Readout", "Data", "SMC-D Med Trace Campaign", "Time Domain Measurements", "C1RP23Dset2_f14_00000.txt")




p_trad = LoadParameters(file_trad)
p_trad.t_start = -1435*1e-9
p_trad.t_end = 0
r_trad = full_analysis(p_trad, fignum=1)

p_doub = LoadParameters(file_doubler)
p_doub.t_start = -1600*1e-9
p_doub.t_end = -100*1e-9
r_doub = full_analysis(p_doub, fignum=2)

fig4, ax4a = plot_spectrum_overlay([r_trad, r_doub], fignum=4)
ax4a.legend(["Traditional", "Doubler"])



# Calculate time points
t_start= -100e-9
t_end = 100e-9
sample_rate = 40e9
n_pts = int((t_end - t_start)*sample_rate+1)
t_series = np.linspace(t_start, t_end, n_pts)

# Calculate gaussian envelope
sigma = 30e-9
f1 = 400e6
f2 = 600e6
amp1 = 1
amp2 = 0.1
envelope = np.exp( -(t_series)**2/2/sigma**2 )
carrier = amp1*np.sin(2*np.pi*f1*t_series)+amp2*np.sin(2*np.pi*f2*t_series)

wave = carrier*envelope

plt.figure(5)
plt.plot(t_series, wave)

fig6 = plt.figure(6)
gs6 = fig6.add_gridspec(2, 1)
ax6a = fig6.add_subplot(gs6[0, 0])
ax6b = fig6.add_subplot(gs6[1, 0])

ax6a.set_xlim([-1480*1e-9, -1450e-9])
ax6a.plot(r_doub.t_si, r_doub.v_si, linestyle=':', marker='.')
ax6b.plot(t_series, carrier)
ax6b.set_xlim([-15*1e-9, 15*1e-9])

plt.tight_layout()

plt.show()