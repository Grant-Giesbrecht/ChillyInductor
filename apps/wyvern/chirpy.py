import blackhole.base as bh

from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtGui import QAction, QActionGroup, QDoubleValidator, QIcon, QFontDatabase, QFont, QPixmap
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtWidgets import QWidget, QTabWidget, QLabel, QGridLayout, QLineEdit, QCheckBox, QSpacerItem, QSizePolicy, QMainWindow, QSlider, QPushButton, QGroupBox, QListWidget, QFileDialog, QProgressBar, QStatusBar

import pylogfile.base as plf
import numpy as np
import sys

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

# Create Log
log = plf.LogPile()

# Create Data Manager
data_manager = bh.BHDatasetManager(log)

window = ChirpAnalyzerMainWindow(log, app, data_manager)

app.exec()