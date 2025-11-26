#!/usr/bin/env python

import argparse
import pylogfile.base as plf
from graf.base import *
from stardust.io import dict_summary

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--sanserif', help="Force use of SanSerif font family.", action='store_true')
parser.add_argument('--serif', help="Force use of Serif font family.", action='store_true')
parser.add_argument('--mono', help="Force use of Monospace font family.", action='store_true')
parser.add_argument('--bold', help="Force use of bold fonts.", action='store_true')
parser.add_argument('--italic', help="Force use of italic fonts.", action='store_true')
parser.add_argument('-s', '--struct', help="Show internal strucutre of GrAF file.", action='store_true')
parser.add_argument('-S', '--structure', help="Show internal strucutre of GrAF file, with verbose options.", action='store_true')
args = parser.parse_args()

filenames = [".\R3C3W1T2_f01,5GHz_P-15dBm_bias_sweep.graf"]

def main():
	
	log = plf.LogPile()
	
	graphs = []
	figs = []
	
	# Get filename from arguments
	for filename in filenames:
	# filename = args.filename
	
		len_gt5 = len(filename) > 5
		len_gt7 = len(filename) > 7
		
		# Read file
		if len_gt5 and filename[-5:].upper() == ".GRAF":
			graf1 = Graf()
			graf1.load_hdf(filename)
		elif len_gt5 and filename[-5:].upper() == ".JSON":
			print(f"JSON")
			# if not log.load_hdf(filename):
			# 	print("\tFailed to read JSON file.")
			pass
		elif len_gt7 and filename[-7:].upper() == ".PKLFIG":
			print(f"PklFig")
			pass
		else:
			print(f"Other")
		
		# Print strucutre if requested
		if args.structure:
			dict_summary(graf1.pack(), verbose=2)
		elif args.struct:
			dict_summary(graf1.pack(), verbose=1)
		
		# Apply styling
		if args.serif:
			print("Setting to serif")
			graf1.style.set_all_font_families("serif")
		elif args.sanserif:
			print("Setting to sanserif")
			graf1.style.set_all_font_families("sanserif")
		elif args.mono:
			print("Setting to monospace")
			graf1.style.set_all_font_families("monospace")
		
		if args.italic:
			# print("Setting to serif")
			graf1.style.title_font.italic = True
			graf1.style.graph_font.italic = True
			graf1.style.label_font.italic = True
		if args.bold:
			# print("Setting to sanserif")
			graf1.style.title_font.bold = True
			graf1.style.graph_font.bold = True
			graf1.style.label_font.bold = True
		
		graphs.append(graf1)
		
		# Generate plot
		figs.append(graf1.to_fig(filename))
		
		fig1 = figs[0]
		fig1_axes = fig1.get_axes()
		ax1a = fig1_axes[0]
		
		ax1a_lines = ax1a.get_lines()
		ax1a_lines[0].set_linestyle(':')
		ax1a_lines[1].set_linestyle(':')
		ax1a_lines[2].set_linestyle(':')
		
		COLOR_TRAD = (0.3, 0.3, 0.3)
		COLOR_DOUB = (0, 119/255, 179/255) # From TQE template section header color
		COLOR_TRIPLE = (0, 128/255, 35/255) # From RB schema figure
		
		def set_color(line, color):
			line.set_color(color)
			line.set_markerfacecolor(color)
			line.set_markeredgecolor(color)
		
		set_color(ax1a_lines[0], COLOR_TRAD)
		set_color(ax1a_lines[1], COLOR_DOUB)
		set_color(ax1a_lines[2], COLOR_TRIPLE)
		# ax1a_lines[0].set_color(COLOR_TRAD)
		# ax1a_lines[1].set_color(COLOR_DOUB)
		# ax1a_lines[2].set_color(COLOR_TRIPLE)
		
		ax1a.set_xlim([0, 0.9])
		ax1a.set_ylim([-80, -20])
		ax1a.legend(["Fundamental", "2nd Harmonic", '3rd Harmonic'], loc="lower right")
		
		# Adjust lines
		
		ax1a_lines[0].set_marker('s')
		ax1a_lines[1].set_marker('o')
		ax1a_lines[2].set_marker('v')
		
	# # Make figure
	# pltfig = graphs[0].to_fig()
	# ax = pltfig.gca()
	# idx = 0
	# for grf in graphs[1:]:
	# 	idx += 1
	# 	ax.plot(grf.get_xdata(), grf.get_ydata(), linestyle=':', marker='.', label=args.filenames[idx])
	
	
	plt.show()

if __name__ == "__main__":
	main()