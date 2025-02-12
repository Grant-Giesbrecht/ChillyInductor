import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import time

# Create initial plot
x = np.arange(0, 10, 0.1)
y = np.sin(x)

fig=  plt.figure()

# Update plot in loop
ims = []
for i in range(100):
	y = np.sin(x + i/10.0)
	im = plt.plot(y, animated=True, color=(0.6, 0, 0))
	ims.append([im[0]])

ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)

FFwriter = animation.FFMpegWriter(fps=10)
ani.save('./test2.mp4', writer=FFwriter, dpi=900)

plt.show()