import blackhole.base as bh
import pandas as pd

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


##==================== Create custom classes for Black-Hole ======================

class ChirpDataset(bh.BHDataset):
	
	def __init__(self, log:plf.LogPile, source_info:bh.BHDataSource):
		super().__init__(log, source_info.unique_id)
		
		self.time_ns = []
		self.volt_mV = []
		
		master_df = pd.read_csv(source_info.file_fullpath, skiprows=4, encoding='utf-8')
		# trim_regions_orig = [[-400, -350], [-300, -225], [-190, -125]]
		
		self.time_ns = master_df['Time']*1e9
		self.volt_mV = master_df['Ampl']*1e3
		
class ChirpAnalyzerMainWindow(bh.BHMainWindow):
	
	def __init__(self, log, app, data_manager):
		super().__init__(log, app, data_manager, window_title="Chirp Analyzer")
		
		self.main_grid = QGridLayout()
		
		self.select_widget = bh.BHDatasetSelectBasicWidget(data_manager, log)
		
		self.main_grid.addWidget(self.select_widget, 0, 0)
		
		# Create central widget
		self.central_widget = QtWidgets.QWidget()
		self.central_widget.setLayout(self.main_grid)
		self.setCentralWidget(self.central_widget)
		
		self.show()

##==================== Create custom functions for Black-Hole ======================

def load_chirp_dataset(source, log):
	return ChirpDataset(log, source)



#================= Basic PyQt App creation things =========================

# Create app object
app = QtWidgets.QApplication(sys.argv)
app.setStyle(f"Fusion")
# app.setWindowIcon

# Create Data Manager
data_manager = bh.BHDatasetManager(log, load_function=load_chirp_dataset)
if not data_manager.load_configuration("chirpy_conf.json"):
	exit()

window = ChirpAnalyzerMainWindow(log, app, data_manager)

app.exec()