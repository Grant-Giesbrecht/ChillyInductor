from heimdallr.all import *
from chillyinductor.rp22_helper import *
import numpy as np
import os
from inputimeout import inputimeout, TimeoutOccurred
from ganymede import *
import datetime
from pathlib import Path
import argparse
import sys
import traceback
from dataclasses import dataclass

from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtWidgets import (
	QApplication, QWidget, QGridLayout, QLabel, QDoubleSpinBox, QPushButton,
	QGroupBox, QVBoxLayout, QHBoxLayout, QMessageBox
)

dmm = Keysight34400("GPIB0::28::INSTR", log)
if dmm.online:
	log.info("Multimeter >ONLINE<")
else:
	log.critical("Failed to connect to multimeter!")
	exit()

tempctrl = LakeShoreModel335("GPIB::12::INSTR", log)
if tempctrl.online:
	log.info("Temperature controller >ONLINE<")
else:
	log.critical("Failed to connect to temperature controller!")
	exit()

mfli = ZurichInstrumentsMFLI("dev5652", "192.168.88.82", log)
if mfli.online:
	log.info("Zurich Instruments MFLI >ONLINE<")
else:
	log.critical("Failed to connect to Zurich Instruments MFLI!")
	exit()

mfli.set_50ohm(False)
mfli.set_range(10)
mfli.set_output_ac_ampl(0)
mfli.set_output_ac_enable(False)
mfli.set_differential_enable(True)
mfli.set_offset(0)
mfli.set_output_enable(True)

# Prepare DMM to measure voltage
dmm.set_measurement(DigitalMultimeterCtg1.MEAS_VOLT_DC, DigitalMultimeterCtg1.RANGE_AUTO)

current_set_res = 243 #152 #TODO: Autocal this
current_meas_res = 152 #152 #TODO: Autocal this
last_set_voltage = 0

# -------------------------
# YOUR INSTRUMENT FUNCTIONS
# -------------------------
# Replace these stubs with your real imports, e.g.:
# from my_instrument_module import read_current, read_voltage, set_current, read_temperature

def read_current() -> float:
	v_read = np.abs(dmm.trigger_and_read())
	return v_read/current_meas_res

def read_voltage() -> float:
	global last_set_voltage
	return last_set_voltage

def set_current(i_amps: float) -> None:
	global last_set_voltage
	Voffset = current_set_res * i_amps
	mfli.set_offset(Voffset)
	last_set_voltage = Voffset
	mfli.set_output_enable(True)

def read_temperature() -> float:
	return tempctrl.get_temp()


@dataclass
class State:
	set_current_A: float = 0.0
	set_voltage_V: float = float("nan")  # "set voltage" interpreted as last read voltage after setting current


class BiasControlGUI(QWidget):
	def __init__(self, poll_ms: int = 250):
		super().__init__()
		self.setWindowTitle("Bias Current Control")

		self.state = State()
		self.poll_ms = int(poll_ms)

		# ---- UI ----
		root = QVBoxLayout(self)

		# Setpoint group
		gb_set = QGroupBox("Setpoint")
		set_layout = QGridLayout(gb_set)

		self.sb_set_current = QDoubleSpinBox()
		self.sb_set_current.setDecimals(9)
		self.sb_set_current.setRange(0, 10.0)   # adjust range to your compliance
		self.sb_set_current.setSingleStep(1e-6)
		self.sb_set_current.setSuffix(" mA")
		self.sb_set_current.setKeyboardTracking(False)

		self.btn_apply = QPushButton("Set current")
		self.btn_apply.clicked.connect(self.on_apply_current)

		self.btn_enable_poll = QPushButton("Start polling")
		self.btn_enable_poll.setCheckable(True)
		self.btn_enable_poll.clicked.connect(self.on_toggle_poll)

		set_layout.addWidget(QLabel("Current setpoint:"), 0, 0)
		set_layout.addWidget(self.sb_set_current, 0, 1)
		set_layout.addWidget(self.btn_apply, 0, 2)
		set_layout.addWidget(self.btn_enable_poll, 1, 2)

		root.addWidget(gb_set)

		# Readback group
		gb_read = QGroupBox("Readback")
		read_layout = QGridLayout(gb_read)

		self.lbl_set_current = QLabel("—")
		self.lbl_set_voltage = QLabel("—")
		self.lbl_meas_current = QLabel("—")
		self.lbl_meas_temp = QLabel("—")

		# Make readbacks selectable for copy/paste
		for w in (self.lbl_set_current, self.lbl_set_voltage, self.lbl_meas_current, self.lbl_meas_temp):
			w.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

		read_layout.addWidget(QLabel("Set current:"), 0, 0)
		read_layout.addWidget(self.lbl_set_current, 0, 1)

		read_layout.addWidget(QLabel("Set voltage:"), 1, 0)
		read_layout.addWidget(self.lbl_set_voltage, 1, 1)

		read_layout.addWidget(QLabel("Measured current:"), 2, 0)
		read_layout.addWidget(self.lbl_meas_current, 2, 1)

		read_layout.addWidget(QLabel("Measured temperature:"), 3, 0)
		read_layout.addWidget(self.lbl_meas_temp, 3, 1)

		root.addWidget(gb_read)

		# Status line
		status_row = QHBoxLayout()
		self.lbl_status = QLabel("Idle.")
		status_row.addWidget(self.lbl_status)
		status_row.addStretch(1)
		root.addLayout(status_row)

		# ---- Poll timer ----
		self.timer = QTimer(self)
		self.timer.setInterval(self.poll_ms)
		self.timer.timeout.connect(self.poll_once)

		# initial display
		self.refresh_labels()

	def set_status(self, msg: str):
		self.lbl_status.setText(msg)

	def safe_call(self, fn, *, name: str):
		"""Call an instrument function; on error, show status and return (False, None)."""
		try:
			return True, fn()
		except Exception as e:
			self.set_status(f"{name} failed: {e}")
			return False, None

	def on_apply_current(self):
		target_mA = float(self.sb_set_current.value())
		target = target_mA/1e3

		ok, _ = self.safe_call(lambda: set_current(target), name="set_current")
		if not ok:
			# optional: popup once; for a lab GUI I usually avoid modal popups for transient glitches
			return

		self.state.set_current_A = target

		# Interpret "set voltage" as "the last measured voltage after applying current"
		ok_v, v = self.safe_call(read_voltage, name="read_voltage (after set)")
		if ok_v:
			self.state.set_voltage_V = float(v)

		self.refresh_labels()
		self.set_status("Current applied.")

	def on_toggle_poll(self, checked: bool):
		if checked:
			self.btn_enable_poll.setText("Stop polling")
			self.timer.start()
			self.set_status(f"Polling every {self.poll_ms} ms.")
		else:
			self.btn_enable_poll.setText("Start polling")
			self.timer.stop()
			self.set_status("Polling stopped.")

	def poll_once(self):
		ok_i, i = self.safe_call(read_current, name="read_current")
		ok_t, t = self.safe_call(read_temperature, name="read_temperature")

		if ok_i:
			self.lbl_meas_current.setText(self.fmt(i*1e3, "mA"))
		if ok_t:
			self.lbl_meas_temp.setText(self.fmt(t, "°K"))

		# If you want live voltage readback too, uncomment:
		# ok_v, v = self.safe_call(read_voltage, name="read_voltage")
		# if ok_v:
		#     self.lbl_set_voltage.setText(self.fmt(v, "V"))

	def refresh_labels(self):
		self.lbl_set_current.setText(self.fmt(self.state.set_current_A, "A"))
		if self.state.set_voltage_V == self.state.set_voltage_V:  # not NaN
			self.lbl_set_voltage.setText(self.fmt(self.state.set_voltage_V, "V"))
		else:
			self.lbl_set_voltage.setText("—")

	@staticmethod
	def fmt(x: float, unit: str) -> str:
		# engineering-ish display without being fancy
		try:
			return f"{float(x):.9g} {unit}"
		except Exception:
			return f"— {unit}"


def main():
	app = QApplication(sys.argv)
	w = BiasControlGUI(poll_ms=250)
	w.resize(520, 220)
	w.show()
	sys.exit(app.exec())


if __name__ == "__main__":
	main()


