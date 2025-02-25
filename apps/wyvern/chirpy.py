import blackhole.base as bh

from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtGui import QAction, QActionGroup, QDoubleValidator, QIcon, QFontDatabase, QFont, QPixmap
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtWidgets import QWidget, QTabWidget, QLabel, QGridLayout, QLineEdit, QCheckBox, QSpacerItem, QSizePolicy, QMainWindow, QSlider, QPushButton, QGroupBox, QListWidget, QFileDialog, QProgressBar, QStatusBar

import pylogfile.base as plf
import numpy as np
import sys

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--detail', help="Show log details.", action='store_true')
parser.add_argument('--loglevel', help="Set the logging display level.", choices=['LOWDEBUG', 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], type=str.upper)
args = parser.parse_args()

# Initialize log
log = plf.LogPile()
if args.loglevel is not None:
	print(f"\tSetting log level to {args.loglevel}")
	log.set_terminal_level(args.loglevel)
else:
	log.set_terminal_level("DEBUG")
log.str_format.show_detail = args.detail

# class ChirpDataManager(bh.BHDatasetManager):
	
# 	def __init__(self, log):
# 		super().__init__(log)

class ChirpDataset(bh.BHDataset):
	
	def __init__(self, log):
		super().__init__(log)

class ChirpAnalyzerMainWindow(bh.BHMainWindow):
	
	def __init__(self, log, app, data_manager):
		super().__init__(log, app, data_manager, window_title="Chirp Analyzer")
		
		self.show()

# Create app object
app = QtWidgets.QApplication(sys.argv)
app.setStyle(f"Fusion")
# app.setWindowIcon

# Create Data Manager
data_manager = bh.BHDatasetManager(log)
data_manager.load_configuration("chirpy_conf.json")

window = ChirpAnalyzerMainWindow(log, app, data_manager)

app.exec()