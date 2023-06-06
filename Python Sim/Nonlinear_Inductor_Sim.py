from core import *

# Pgen = -4 # dBm
Pgen = 0 # dBm
C_ = 121e-12
l_phys = 0.5
freq = 10e9
q = 0.190
L0 = 269e-9

# lks.configure_time_domain()
# lks.convergence_settings()

# lks.solve(Ibias=[-10, 0, 10])
# lks.solution

# lks.crunch(.01, .03, show_plot_td=True, show_plot_spec=True)

# print(f"{Fore.RED}The problem is that the FFT on Iac acts as if Iac is I vs t, when it's really")
# print(f"{Fore.RED}the amplitude multiplying a sine term. I think I need to add an extra sine in")
# print(f"{Fore.RED}there or similar. But Iac_td is sitting at 2.something and FFT is showing Iac")
# print(f"{Fore.RED} = 0, which agrees with the guess of Iac=0 (which is really setting the osc.")
# print(f"{Fore.RED}amplitude of Iac).")

# Ibias = [.01, .02, .03]
Ibias = np.linspace(1, 33, 34)*1e-3

lks = LKSystem(Pgen, C_, l_phys, freq, q, L0)
lks.opt.start_guess_method = GUESS_USE_LAST
lks.opt.tol_pcnt = 0.1
# lks.opt.start_guess_method = GUESS_ZERO_REFLECTION
lks.solve(Ibias, show_plot_on_conv=True)

Iac = np.array([x.Iac for x in lks.solution])

plt.plot(Ibias*1e3, Iac*1e3, '-g')
plt.grid()
plt.xlabel("Bias Current (mA)")
plt.ylabel("AC Current Amplitude (mA)")
plt.title("Simulation Summary")

plt.show()