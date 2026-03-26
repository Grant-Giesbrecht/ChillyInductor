from RP23SMCD import *
import mplcursors

file_trad = os.path.join("F:\\", "ARC0 PhD Data", "RP-23 Qubit Readout", "Data", "SMC-D Med Trace Campaign", "Time Domain Measurements", "C1RP23Dset2_f07_00000.txt")
file_doubler = os.path.join("F:\\", "ARC0 PhD Data", "RP-23 Qubit Readout", "Data", "SMC-D Med Trace Campaign", "Time Domain Measurements", "C1RP23Dset2_f10_00000.txt")
file_tripler = os.path.join("F:\\", "ARC0 PhD Data", "RP-23 Qubit Readout", "Data", "SMC-D Med Trace Campaign", "Time Domain Measurements", "C1RP23Dset2_f14_00000.txt")




p_trad = LoadParameters(file_trad)
p_trad.t_start = -1435*1e-9
p_trad.t_end = 0
r_trad = full_analysis(p_trad, fignum=1)

p_doub = LoadParameters(file_doubler)
p_doub.t_start = -1600*1e-9
p_doub.t_end = -100*1e-9
r_doub = full_analysis(p_doub, fignum=2)

p_trip = LoadParameters(file_tripler)
p_trip.t_start = -1600*1e-9
p_trip.t_end = -100*1e-9
r_trip = full_analysis(p_trip, fignum=3)

fig4, ax4a = plot_spectrum_overlay([r_trad, r_doub, r_trip], fignum=4)
ax4a.legend(["Traditional", "Doubler", "Tripler"])

ax4a.set_xlim([100, 700])
ax4a.set_ylim([-190, -110])

fig4.savefig("fft_100MHz_to_700MHz.svg")

mplcursors.cursor(multiple=True)

plt.show()