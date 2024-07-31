import numpy as np
from ganymede import *

# # Frequency nulls - these values were manually picked off the graph (unit GHz) 
# # for data file 25June2024_Mid.csv.
# freqs = [8.619,8.817, 9, 9.18, 9.364]
# GMAX_DB = -6.4

# Frequency nulls - these values were manually picked off the graph (unit GHz) 
# for data file Sparam_31July2024_-30dBm_R4C4T1.csv.
freqs = [0.876, 1.220, 1.542]
GMAX_DB = -9.7

# Get approx spacing
deltas = np.diff(freqs)
f_mean = np.mean(deltas)

# Estimate null number
approx_null_no = freqs/f_mean
print(f"Initial guess, null-number: {approx_null_no}")

# Calculate approx freq different for each to be an integer null number
fd = np.array([ f/round(approx_null_no[idx]) for idx, f in enumerate(freqs)])

# Get mean difference
fdm = np.mean(fd)

# Approx null number with new mean
print(f"Enhanced guess, null-number: {freqs/fdm}")

ZL = 50
IDX = 0

GMAX = dB_to_lin(GMAX_DB)

L = approx_null_no[IDX]*ZL/2/(freqs[IDX]*1e9)*np.sqrt((1+GMAX)/(1-GMAX))
C = (approx_null_no[IDX]/2/(freqs[IDX]*1e9)/np.sqrt(L))**2

print(f"\nEstimated L = {L*1e6} uH")
print(f"Estimated C = {C*1e12} pF")
print(f"Estimated Z0 = {np.sqrt(L/C)} pF")