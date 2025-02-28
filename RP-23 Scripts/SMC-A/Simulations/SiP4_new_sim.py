'''
Modified version of SiP2a. Whereas SiP2a simulated the entire waveform, it was just for visual
coolness. The envelope and thus the phase velocity was calcualted once at the beginning.

SiP3 supports multiple tones by re-calculating the envelope and thus the phase velocity at each
frame of the simualtiosn.
'''

import matplotlib.pyplot as plt
# from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import Image
import pandas as pd
import numpy as np
import os
import datetime
import hashlib
import matplotlib.animation as animation
import copy
import time
from matplotlib.animation import FFMpegWriter
import argparse
from chirp_sim import *

from pylogfile.base import markdown
import pylogfile.base as plf

#======================= Check for command line arguments ==================

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--detail', help="Show detailed log messages.", action='store_true')
parser.add_argument('--loglevel', help="Set the logging display level.", choices=['LOWDEBUG', 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], type=str.upper)
parser.add_argument('-s', '--savemp4', help="Save MP4 of animation.", action='store_true')
args = parser.parse_args()

log = plf.LogPile()
if args.loglevel is not None:
	log.set_terminal_level(args.loglevel)
if args.detail:
	log.str_format.show_detail = args.detail

#====================== Create Simulation Objects =======================

x_init = np.linspace(0, 10, 101)
waveform = (1+x_init/10)*np.sin(x_init*2*np.pi*0.25)

wav = Waveform(x_init, waveform, log)
sa = SimArea(bin_size=0.1, log=log)

sim = ChirpSimulation(wav, sa, log, t_stop=10, dt=0.1)

#================== Prepare simulation graphics ========================

# Create figure and turn on interactive mode
plt.ion()
fig1 = plt.figure(figsize=(12,6))
fig1.suptitle(f"ID hash: {sim.hash_id}, ({sim.hash_id_short})", fontsize=8)
gs = fig1.add_gridspec(1, 1)
ax0 = fig1.add_subplot(gs[0, 0])

# Initialize axes
ax0.set_xlabel("Position (nm)")
ax0.set_ylabel("Amplitude (1)")
ax0.grid(True)
# ax0.set_xlim([x_pos[0]/1e-9, x_pos[-1]/1e-9])
ax0.set_ylim([-3, 3])
# ax0.set_xlim([0, 4])

# Initialize artists
pulse_artist, = ax0.plot([], [], linestyle=':', marker='.', color=(0.35, 0.3, 0.65))
# region_artist = ax0.fill_between([nonlinear_region[0]/1e-9, nonlinear_region[1]/1e-9], [-100, -100], [100, 100], color=(0.4, 0.4, 0.4), alpha=0.2)

#================== Run Sim ========================

# pulse_artist.set_data([1, 2, 3], [-1, .5, -.5])
# fig1.canvas.draw()
# fig1.canvas.flush_events()
# time.sleep(10)

sim.run(artist=pulse_artist, fig=fig1)




