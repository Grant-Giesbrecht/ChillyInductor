"""
Pickle Figure Viewer
====================
Drag and drop Python pickle files containing matplotlib figures.
Each file gets a top-level tab; each figure within a file gets a nested tab.

Requirements:
	pip install PyQt5 matplotlib
"""

import sys
import pickle
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Qt5Agg")

from matplotlib.backends.backend_qt5agg import (
	FigureCanvasQTAgg as FigureCanvas,
	NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure

from PyQt5.QtCore import Qt, QMimeData
from PyQt5.QtGui import QDragEnterEvent, QDropEvent, QColor, QPalette, QFont, QIcon
from PyQt5.QtWidgets import (
	QApplication,
	QMainWindow,
	QTabWidget,
	QWidget,
	QVBoxLayout,
	QLabel,
	QSizePolicy,
	QScrollArea,
	QMessageBox,
	QTabBar,
	QStyleFactory,
)


# ── Colour palette ────────────────────────────────────────────────────────────
BG        = "#1a1a2e"   # deep navy
SURFACE   = "#16213e"   # card surface
ACCENT    = "#0f3460"   # tab bar bg
HIGHLIGHT = "#e94560"   # active-tab accent / drop indicator
TEXT      = "#eaeaea"
SUBTEXT   = "#8888aa"
BORDER    = "#2a2a4a"


STYLESHEET = f"""
QMainWindow, QWidget {{
	background-color: {BG};
	color: {TEXT};
	font-family: "Menlo", "Consolas", "DejaVu Sans Mono", monospace;
}}

/* ── Outer (file) tab bar ── */
QTabWidget#fileTab::pane {{
	border: 1px solid {BORDER};
	border-top: 2px solid {HIGHLIGHT};
	background: {SURFACE};
}}
QTabWidget#fileTab > QTabBar::tab {{
	background: {ACCENT};
	color: {SUBTEXT};
	padding: 8px 20px;
	margin-right: 2px;
	border-top-left-radius: 4px;
	border-top-right-radius: 4px;
	font-size: 12px;
	min-width: 120px;
}}
QTabWidget#fileTab > QTabBar::tab:selected {{
	background: {SURFACE};
	color: {TEXT};
	border-top: 2px solid {HIGHLIGHT};
	font-weight: bold;
}}
QTabWidget#fileTab > QTabBar::tab:hover:!selected {{
	background: #1e2a50;
	color: {TEXT};
}}

/* ── Inner (figure) tab bar ── */
QTabWidget#figTab::pane {{
	border: 1px solid {BORDER};
	border-top: 2px solid #4a90d9;
	background: {BG};
}}
QTabWidget#figTab > QTabBar::tab {{
	background: #0d1b33;
	color: {SUBTEXT};
	padding: 6px 16px;
	margin-right: 2px;
	border-top-left-radius: 3px;
	border-top-right-radius: 3px;
	font-size: 11px;
	min-width: 90px;
}}
QTabWidget#figTab > QTabBar::tab:selected {{
	background: {BG};
	color: {TEXT};
	border-top: 2px solid #4a90d9;
}}
QTabWidget#figTab > QTabBar::tab:hover:!selected {{
	background: #162038;
	color: {TEXT};
}}

/* ── Navigation toolbar ── */
QToolBar {{
	background: {SURFACE};
	border: none;
	spacing: 2px;
	padding: 2px 4px;
}}
QToolButton {{
	background: transparent;
	color: {TEXT};
	border: none;
	border-radius: 3px;
	padding: 3px;
}}
QToolButton:hover {{
	background: {ACCENT};
}}

/* ── Drop-zone label ── */
QLabel#dropLabel {{
	color: {SUBTEXT};
	font-size: 15px;
	border: 2px dashed {BORDER};
	border-radius: 12px;
	padding: 40px;
	background: transparent;
}}

/* ── Scrollbar ── */
QScrollBar:vertical {{
	background: {SURFACE};
	width: 8px;
	border-radius: 4px;
}}
QScrollBar::handle:vertical {{
	background: {ACCENT};
	border-radius: 4px;
	min-height: 24px;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}

/* ── Status bar ── */
QStatusBar {{
	background: {ACCENT};
	color: {SUBTEXT};
	font-size: 11px;
}}
"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_figures_from_pickle(path: str) -> list[Figure]:
	"""Return a list of matplotlib Figure objects from a pickle file."""
	with open(path, "rb") as fh:
		obj = pickle.load(fh)

	# Normalise to list
	if isinstance(obj, Figure):
		return [obj]
	elif isinstance(obj, (list, tuple)):
		figs = [o for o in obj if isinstance(o, Figure)]
		if figs:
			return figs
	elif isinstance(obj, dict):
		figs = [v for v in obj.values() if isinstance(v, Figure)]
		if figs:
			return figs

	raise ValueError(
		f"Pickle contains {type(obj).__name__!r}, which has no matplotlib Figure objects."
	)


def make_figure_tab(fig: Figure) -> QWidget:
	"""Wrap a Figure in a QWidget with a navigation toolbar."""
	widget = QWidget()
	layout = QVBoxLayout(widget)
	layout.setContentsMargins(0, 0, 0, 0)
	layout.setSpacing(0)

	canvas = FigureCanvas(fig)
	canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

	# Style the figure background to match the app
	fig.patch.set_facecolor(BG)
	for ax in fig.get_axes():
		ax.set_facecolor("#111128")
		ax.tick_params(colors=SUBTEXT)
		ax.xaxis.label.set_color(TEXT)
		ax.yaxis.label.set_color(TEXT)
		ax.title.set_color(TEXT)
		for spine in ax.spines.values():
			spine.set_edgecolor(BORDER)

	canvas.draw()

	toolbar = NavigationToolbar(canvas, widget)
	layout.addWidget(toolbar)
	layout.addWidget(canvas)
	return widget


def make_file_tab(figures: list[Figure], filepath: str) -> QWidget:
	"""Create the per-file tab containing a nested figure tab widget."""
	outer = QWidget()
	layout = QVBoxLayout(outer)
	layout.setContentsMargins(4, 4, 4, 4)
	layout.setSpacing(0)

	if len(figures) == 1:
		# Skip inner tabs if only one figure
		layout.addWidget(make_figure_tab(figures[0]))
	else:
		fig_tabs = QTabWidget()
		fig_tabs.setObjectName("figTab")
		fig_tabs.setTabPosition(QTabWidget.North)
		for i, fig in enumerate(figures):
			title = getattr(fig, "canvas", None)
			title = (fig.get_label() or f"Figure {i + 1}")
			fig_tabs.addTab(make_figure_tab(fig), title)
		layout.addWidget(fig_tabs)

	return outer


# ── Drop-zone placeholder ─────────────────────────────────────────────────────

class DropZone(QLabel):
	def __init__(self):
		super().__init__(
			"Drop  .pkl / .pickle  files here\n\n"
			"Each file → file tab\n"
			"Each figure → figure tab"
		)
		self.setObjectName("dropLabel")
		self.setAlignment(Qt.AlignCenter)
		self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)


# ── Main window ───────────────────────────────────────────────────────────────

class PickleViewer(QMainWindow):
	def __init__(self):
		super().__init__()
		self.setWindowTitle("Pickle Figure Viewer")
		self.resize(1100, 760)
		self.setAcceptDrops(True)

		# Central stack: show drop-zone until first file loaded
		self._container = QWidget()
		self._container_layout = QVBoxLayout(self._container)
		self._container_layout.setContentsMargins(0, 0, 0, 0)

		self._drop_zone = DropZone()
		self._file_tabs = QTabWidget()
		self._file_tabs.setObjectName("fileTab")
		self._file_tabs.setTabPosition(QTabWidget.North)
		self._file_tabs.setTabsClosable(True)
		self._file_tabs.tabCloseRequested.connect(self._close_tab)

		self._container_layout.addWidget(self._drop_zone)
		self._container_layout.addWidget(self._file_tabs)
		self._file_tabs.hide()

		self.setCentralWidget(self._container)
		self.statusBar().showMessage("Ready — drag pickle files onto the window.")

	# ── Drag / drop ────────────────────────────────────────────────────────

	def dragEnterEvent(self, event: QDragEnterEvent):
		if event.mimeData().hasUrls():
			urls = event.mimeData().urls()
			# if any(u.toLocalFile().endswith((".pklfig", ".pickle")) for u in urls):
			event.acceptProposedAction()
			self._drop_zone.setStyleSheet(
				f"QLabel#dropLabel {{ border-color: {HIGHLIGHT}; color: {TEXT}; }}"
			)

	def dragLeaveEvent(self, event):
		self._drop_zone.setStyleSheet("")

	def dropEvent(self, event: QDropEvent):
		self._drop_zone.setStyleSheet("")
		paths = [
			u.toLocalFile()
			for u in event.mimeData().urls()
			if u.toLocalFile().endswith((".pklfig", ".pickle"))
		]
		for path in paths:
			self._load_file(path)

	# ── File loading ───────────────────────────────────────────────────────

	def _load_file(self, path: str):
		name = Path(path).name
		self.statusBar().showMessage(f"Loading {name} …")
		QApplication.processEvents()

		try:
			figures = load_figures_from_pickle(path)
		except Exception as exc:
			QMessageBox.critical(
				self,
				"Load Error",
				f"<b>{name}</b><br><br>{exc}<br><br>"
				f"<pre>{traceback.format_exc()}</pre>",
			)
			self.statusBar().showMessage("Error loading file.")
			return

		tab_widget = make_file_tab(figures, path)
		idx = self._file_tabs.addTab(tab_widget, name)
		self._file_tabs.setTabToolTip(idx, path)
		self._file_tabs.setCurrentIndex(idx)

		if self._file_tabs.count() == 1:
			self._drop_zone.hide()
			self._file_tabs.show()

		n = len(figures)
		self.statusBar().showMessage(
			f"Loaded '{name}'  —  {n} figure{'s' if n != 1 else ''}"
		)

	def _close_tab(self, index: int):
		self._file_tabs.removeTab(index)
		if self._file_tabs.count() == 0:
			self._file_tabs.hide()
			self._drop_zone.show()
			self.statusBar().showMessage("Ready — drag pickle files onto the window.")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
	app = QApplication(sys.argv)
	app.setStyle(QStyleFactory.create("Fusion"))
	app.setStyleSheet(STYLESHEET)

	viewer = PickleViewer()
	viewer.show()
	sys.exit(app.exec_())


if __name__ == "__main__":
	main()