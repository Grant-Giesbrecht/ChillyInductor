import numpy as np

# Frequency nulls
freqs = [8.619,8.817, 9, 9.18, 9.364]

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
