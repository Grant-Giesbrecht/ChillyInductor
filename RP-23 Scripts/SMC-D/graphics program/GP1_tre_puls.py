import matplotlib.pyplot as plt
import numpy as np
from stardust.algorithm import linstep
import matplotlib.ticker as ticker

t = np.array(linstep(start=-6, stop=6, step=0.01))

f0 = 1
sigma = 1

sigma1 = sigma
sigma2 = sigma*np.sqrt(2)
sigma3 = sigma*np.sqrt(3)

env1 = np.exp(-(t**2)/(2*sigma1**2))
env2 = np.exp(-(t**2)/(2*sigma2**2))
env3 = np.exp(-(t**2)/(2*sigma3**2))

v1 = np.cos(2*np.pi*f0*t) * env1
v2 = np.cos(2*np.pi*f0/2*t) * env2
v3 = np.cos(2*np.pi*f0/3*t) * env3

env2_final = env2**2
env3_final = env3**3

v2_final = (v2**2  )*2 - env2_final
v3_final = (v3**3)#-env3_final

fig1 = plt.figure(1)
gs1 = fig1.add_gridspec(1, 1)
ax1a = fig1.add_subplot(gs1[0, 0])

alpha_env = 1
alpha_sine = 0.6

color1 = (255/255, 185/255, 60/255)
color2 = (0, 130/255, 197/255)
color3 = (208/255, 10/255, 17/255)
color4 = (69/255, 135/255, 58/255)

ax1a.plot(t, env1, color=color1, alpha=alpha_env)
ax1a.plot(t, env2, color=color2, alpha=alpha_env)
ax1a.plot(t, env3, color=color3, alpha=alpha_env)

ax1a.plot(t, v1, color=color1, alpha=alpha_sine)
ax1a.plot(t, v2, color=color2, alpha=alpha_sine)
ax1a.plot(t, v3, color=color3, alpha=alpha_sine*0.7)

ax1a.grid(True)

# fig2 = plt.figure(2)
# gs2 = fig2.add_gridspec(1, 1)
# ax2a = fig2.add_subplot(gs2[0, 0])
#
# alpha_sine = 0.8
# 
# ax2a.plot(t, env1, color=color1, alpha=alpha_env)
# ax2a.plot(t, env2_final, color=color2, alpha=alpha_env)
# ax2a.plot(t, env3_final, color=color3, alpha=alpha_env)
# 
# ax2a.plot(t, v1, color=color1, alpha=alpha_sine)
# ax2a.plot(t, v2_final, color=color2, alpha=alpha_sine)
# ax2a.plot(t, v3_final, color=color3, alpha=alpha_sine*0.7)
# 
# ax2a.grid(True)
# 
# fig2.tight_layout()

fig3 = plt.figure(3)
gs3 = fig3.add_gridspec(3, 1)
ax3a = fig3.add_subplot(gs3[0, 0])
ax3b = fig3.add_subplot(gs3[1, 0])
ax3c = fig3.add_subplot(gs3[2, 0])

ax3a.plot(t, env1, color=color1, alpha=alpha_env)
ax3b.plot(t, env2, color=color2, alpha=alpha_env)
ax3c.plot(t, env3, color=color3, alpha=alpha_env)

ax3a.plot(t, v1, color=color1, alpha=alpha_sine)
ax3b.plot(t, v2, color=color2, alpha=alpha_sine)
ax3c.plot(t, v3, color=color3, alpha=alpha_sine*0.7)

ylim_tuple = [-1.1, 1.1]
ax_list = [ax3a, ax3b, ax3c]
for ax_ in ax_list:
	ax_.set_ylim(ylim_tuple)
	
	# Set major tick spacing for x-axis to 2 units
	ax_.xaxis.set_major_locator(ticker.MultipleLocator(1))
	# Set major tick spacing for y-axis to 0.5 units
	ax_.yaxis.set_major_locator(ticker.MultipleLocator(1))

ax3a.grid(True)
ax3b.grid(True)
ax3c.grid(True)

fig3.tight_layout()


fig4 = plt.figure(4, figsize=([6.4, 4]))
gs4 = fig4.add_gridspec(1, 1)
ax4a = fig4.add_subplot(gs4[0, 0])

ax4a.plot(t, env1, color=color4, alpha=alpha_env)

ax4a.plot(t, v1, color=color4, alpha=alpha_sine)

ylim_tuple = [-1.1, 1.1]
ax_list = [ax4a]
for ax_ in ax_list:
	ax_.set_ylim(ylim_tuple)
	
	# Set major tick spacing for x-axis to 2 units
	ax_.xaxis.set_major_locator(ticker.MultipleLocator(1))
	# Set major tick spacing for y-axis to 0.5 units
	ax_.yaxis.set_major_locator(ticker.MultipleLocator(0.5))

	ax_.grid(True)
	
fig4.tight_layout()



fig1.savefig('GP1_fig1.pdf')
fig3.savefig('GP1_fig3.pdf')
fig4.savefig('GP1_fig4.pdf')

plt.show()