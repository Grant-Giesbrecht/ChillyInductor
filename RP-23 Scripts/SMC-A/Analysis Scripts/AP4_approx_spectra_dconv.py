import matplotlib.pyplot as plt
import mplcursors

df = 0.001
noise_floor = -100

f_start = 0
f_end = 10

f_sgma = [3.8, 7.6] # GHz
p_sgma = [-23.7, -70.5] #dBm

f_agil = [4.8]
p_agil = [-34.9]

f_all = [0.2, 0.8, 1, 1.8, 2, 2.8, 3.8, 4.6, 4.8, 5.6, 5.8, 6.6, 7.61, 8.6, 9.6]
p_all = [-79, -75, -58.6, -64.3, -72.2, -50.7, -28.6, -76.3, -40.7, -73.5, -64.8, -64.5, -47.1, -47, -67.8]

def discrete_points_to_spectrum(freqs, powers, offset):
	
	f_out = [f_start]
	p_out = [noise_floor]
	
	for f, p in zip(freqs, powers):
		f_out.append(f-df+offset)
		p_out.append(noise_floor)
		
		f_out.append(f+offset)
		p_out.append(p)
		
		f_out.append(f+df+offset)
		p_out.append(noise_floor)
	
	f_out.append(f_end)
	p_out.append(noise_floor)
	
	return f_out, p_out

f_sgma, p_sgma = discrete_points_to_spectrum(f_sgma, p_sgma, 0)
f_agil, p_agil = discrete_points_to_spectrum(f_agil, p_agil, 0)
f_all, p_all = discrete_points_to_spectrum(f_all, p_all, 0)

plt.figure(1)

LW = 2
ALP = 0.6



# plt.plot(f_sgma, p_sgma, label="RF Only", color="#364156", linewidth=LW, alpha=ALP)
# plt.plot(f_agil, p_agil, label="LO Only", color="#7D4E57", linewidth=LW, alpha=ALP)
# plt.plot(f_all, p_all, label="RF + LO + Bias", color="#212D40", linewidth=LW, alpha=ALP)
plt.plot(f_sgma, p_sgma, label="RF Only", linewidth=LW, alpha=ALP, marker='+')
plt.plot(f_agil, p_agil, label="LO Only", linewidth=LW, alpha=ALP, marker='+')
plt.plot(f_all, p_all, label="RF + LO + Bias", linewidth=LW, alpha=ALP, marker='+')
plt.grid(True)
plt.xlabel("Frequency (GHz)")
plt.ylabel("Power (dBm)")
plt.legend()


plt.figure(2)
LW = 2
ALP = 1
plt.plot(f_sgma, p_sgma, label="LO Only", linewidth=LW, alpha=ALP, marker='x')
plt.plot(f_agil, p_agil, label="RF Only", linewidth=LW, alpha=ALP, marker='x')
plt.plot(f_all, p_all, label="RF + LO + Bias", linewidth=1, alpha=ALP, marker='x', linestyle=':', markersize=7)
plt.grid(True)
plt.xlabel("Frequency (GHz)")
plt.ylabel("Power (dBm)")
plt.legend()

mplcursors.cursor(multiple=True)

plt.show()

