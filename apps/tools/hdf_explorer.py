#!/usr/bin/env python3
"""
PyQt6 HDF5 Viewer/Editor (with Lock + Undo)

Additions:
- Lock toggle (ON by default): prevents accidental edits
- Undo buffer: revert last successful write (dataset cell / attribute add/modify/delete)
- Extra error catching around IO and UI callbacks
"""

from __future__ import annotations

import sys
import traceback
from dataclasses import dataclass
from typing import Any, Optional, Tuple, List, Dict, Union

import numpy as np
import h5py

from PyQt6.QtCore import (
	Qt, QAbstractTableModel, QModelIndex, QVariant, QObject, pyqtSignal
)
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
	QApplication, QMainWindow, QFileDialog, QMessageBox,
	QTreeWidget, QTreeWidgetItem, QSplitter, QWidget, QVBoxLayout,
	QTableView, QHeaderView, QLabel, QFormLayout, QLineEdit,
	QHBoxLayout, QPushButton, QAbstractItemView, QDialog, QDialogButtonBox,
	QToolButton
)


# -------------------------
# Global error helper
# -------------------------

def show_error(parent: Optional[QWidget], title: str, msg: str, exc: Optional[BaseException] = None):
	detail = ""
	if exc is not None:
		detail = traceback.format_exc()
	box = QMessageBox(parent)
	box.setIcon(QMessageBox.Icon.Critical)
	box.setWindowTitle(title)
	box.setText(msg)
	if detail:
		box.setDetailedText(detail)
	box.exec()


# -------------------------
# Helpers: type conversion
# -------------------------

def _is_bytes_dtype(dt: np.dtype) -> bool:
	return dt.kind == "S"

def _is_unicode_dtype(dt: np.dtype) -> bool:
	return dt.kind == "U"

def _is_numeric_dtype(dt: np.dtype) -> bool:
	return dt.kind in ("i", "u", "f", "c")

def _parse_scalar_text(text: str) -> Any:
	t = text.strip()
	if t.lower() in ("true", "false"):
		return t.lower() == "true"
	try:
		if t.startswith(("0x", "0X")):
			return int(t, 16)
		return int(t)
	except Exception:
		pass
	try:
		return float(t)
	except Exception:
		pass
	return t

def _coerce_to_dtype(value: Any, dt: np.dtype) -> Any:
	if isinstance(value, str):
		parsed = _parse_scalar_text(value)
	else:
		parsed = value

	if _is_numeric_dtype(dt):
		try:
			return np.array(parsed, dtype=dt).item()
		except Exception as e:
			raise ValueError(f"Cannot convert {parsed!r} to dtype {dt}") from e

	if dt.kind == "b":
		if isinstance(parsed, bool):
			return bool(parsed)
		if isinstance(parsed, (int, float, np.integer, np.floating)):
			return bool(parsed)
		if isinstance(parsed, str):
			s = parsed.strip().lower()
			if s in ("true", "1", "yes", "y", "t"):
				return True
			if s in ("false", "0", "no", "n", "f"):
				return False
		raise ValueError(f"Cannot convert {parsed!r} to bool")

	if _is_bytes_dtype(dt):
		b = parsed
		if isinstance(b, str):
			b = b.encode("utf-8")
		elif isinstance(b, (bytes, bytearray)):
			b = bytes(b)
		else:
			b = str(b).encode("utf-8")
		maxlen = dt.itemsize
		if len(b) > maxlen:
			raise ValueError(f"String too long for dtype {dt} (max {maxlen} bytes)")
		return np.array(b, dtype=dt).tobytes()

	if _is_unicode_dtype(dt):
		s = parsed if isinstance(parsed, str) else str(parsed)
		maxlen = dt.itemsize // 4
		if len(s) > maxlen:
			raise ValueError(f"String too long for dtype {dt} (max {maxlen} chars)")
		return np.array(s, dtype=dt).item()

	raise ValueError(f"Editing for dtype {dt} not supported")

def _format_attr_value(v: Any) -> str:
	if isinstance(v, np.ndarray):
		if v.ndim == 0:
			return str(v.item())
		return np.array2string(v, threshold=20, edgeitems=3)
	if isinstance(v, (bytes, bytearray, np.bytes_)):
		try:
			return bytes(v).decode("utf-8")
		except Exception:
			return repr(v)
	return str(v)


# -------------------------
# Undo system
# -------------------------

@dataclass
class UndoAction:
	kind: str  # "dataset_set" | "attr_set" | "attr_add" | "attr_del"
	path: str
	# dataset fields
	index: Optional[Tuple[int, ...]] = None
	old_value: Any = None
	new_value: Any = None
	# attr fields
	attr_key: Optional[str] = None
	existed_before: Optional[bool] = None  # for attr_set: whether it existed
	# for deletes, store deleted value
	deleted_value: Any = None


class H5PagedDatasetTableModel(QAbstractTableModel):
	"""
	Page/window viewer for large datasets.
	Only loads a slice into cache (chunk) and serves table values from cache.
	Supports scalar/1D/2D only (same as before), but without full dataset load.
	"""

	def __init__(self, dref: DatasetRef, undo: UndoManager, parent=None,
				 row0: int = 0, col0: int = 0, view_rows: int = 500, view_cols: int = 200):
		super().__init__(parent)
		self.dref = dref
		self.undo = undo
		self._locked = True

		self.row0 = row0
		self.col0 = col0
		self.view_rows = view_rows
		self.view_cols = view_cols

		self._cache = None
		self._cache_shape = (0, 0)
		self._dtype = None

		self.reload_chunk()

	def set_locked(self, locked: bool):
		self._locked = locked
		self.layoutChanged.emit()

	def set_window(self, row0: int, col0: int, view_rows: int | None = None, view_cols: int | None = None):
		self.row0 = max(0, int(row0))
		self.col0 = max(0, int(col0))
		if view_rows is not None:
			self.view_rows = max(1, int(view_rows))
		if view_cols is not None:
			self.view_cols = max(1, int(view_cols))
		self.reload_chunk()

	def reload_chunk(self):
		ds = self.dref.ds
		self.beginResetModel()
		self._dtype = ds.dtype
		self._cache = None
		self._cache_shape = (0, 0)

		try:
			if ds.ndim == 0:
				self._cache = np.array([[ds[()]]])
				self._cache_shape = (1, 1)

			elif ds.ndim == 1:
				r0 = min(self.row0, max(0, ds.shape[0] - 1))
				r1 = min(ds.shape[0], r0 + self.view_rows)
				chunk = ds[r0:r1]
				self._cache = np.array(chunk).reshape(-1, 1)
				self._cache_shape = (self._cache.shape[0], 1)

			elif ds.ndim == 2:
				r0 = min(self.row0, max(0, ds.shape[0] - 1))
				c0 = min(self.col0, max(0, ds.shape[1] - 1))
				r1 = min(ds.shape[0], r0 + self.view_rows)
				c1 = min(ds.shape[1], c0 + self.view_cols)
				chunk = ds[r0:r1, c0:c1]
				self._cache = np.array(chunk)
				self._cache_shape = self._cache.shape

			else:
				self._cache = None
				self._cache_shape = (0, 0)

		except Exception:
			self._cache = None
			self._cache_shape = (0, 0)

		self.endResetModel()

	def rowCount(self, parent=QModelIndex()) -> int:
		if parent.isValid():
			return 0
		return self._cache_shape[0]

	def columnCount(self, parent=QModelIndex()) -> int:
		if parent.isValid():
			return 0
		return self._cache_shape[1]

	def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole):
		if role not in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole):
			return QVariant()
		if not index.isValid() or self._cache is None:
			return QVariant()

		try:
			val = self._cache[index.row(), index.column()]
		except Exception:
			return QVariant()

		if isinstance(val, (bytes, bytearray, np.bytes_)):
			try:
				return bytes(val).decode("utf-8")
			except Exception:
				return repr(val)
		if isinstance(val, np.generic):
			return str(val.item())
		return str(val)

	def flags(self, index: QModelIndex):
		if not index.isValid():
			return Qt.ItemFlag.NoItemFlags
		base = Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled
		if self._locked:
			return base
		return base | Qt.ItemFlag.ItemIsEditable

	def _global_index(self, r: int, c: int) -> Tuple[int, ...]:
		ds = self.dref.ds
		if ds.ndim == 0:
			return tuple()
		if ds.ndim == 1:
			return (self.row0 + r,)
		return (self.row0 + r, self.col0 + c)

	def setData(self, index: QModelIndex, value: Any, role: int = Qt.ItemDataRole.EditRole) -> bool:
		if role != Qt.ItemDataRole.EditRole or not index.isValid() or self._cache is None:
			return False
		if self._locked:
			return False

		ds = self.dref.ds
		dt = ds.dtype

		try:
			coerced = _coerce_to_dtype(str(value), dt)
		except Exception as e:
			QMessageBox.critical(None, "Type conversion failed", str(e))
			return False

		gidx = self._global_index(index.row(), index.column())

		# read old for undo
		try:
			old = ds[()] if ds.ndim == 0 else ds[gidx]
		except Exception as e:
			QMessageBox.critical(None, "Read failed", f"Could not read old value:\n{e}")
			return False

		# write
		try:
			if ds.ndim == 0:
				ds[()] = coerced
				self._cache[0, 0] = np.array(coerced, dtype=dt)
			else:
				ds[gidx] = coerced
				self._cache[index.row(), index.column()] = np.array(coerced, dtype=dt)
		except Exception as e:
			QMessageBox.critical(None, "Write failed", f"Could not write:\n{e}")
			return False

		self.undo.push(UndoAction(kind="dataset_set", path=ds.name, index=gidx, old_value=old, new_value=coerced))
		self.dataChanged.emit(index, index, [Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole])
		return True


class UndoManager(QObject):
	changed = pyqtSignal()

	def __init__(self, limit: int = 500):
		super().__init__()
		self._stack: List[UndoAction] = []
		self._limit = limit

	def push(self, action: UndoAction):
		self._stack.append(action)
		if len(self._stack) > self._limit:
			self._stack.pop(0)
		self.changed.emit()

	def can_undo(self) -> bool:
		return len(self._stack) > 0

	def clear(self):
		self._stack.clear()
		self.changed.emit()

	def pop(self) -> Optional[UndoAction]:
		if not self._stack:
			return None
		act = self._stack.pop()
		self.changed.emit()
		return act

	def peek_summary(self) -> str:
		if not self._stack:
			return "Undo"
		a = self._stack[-1]
		if a.kind == "dataset_set":
			return f"Undo dataset edit {a.path}{a.index}"
		if a.kind == "attr_set":
			return f"Undo attr set {a.path}@{a.attr_key}"
		if a.kind == "attr_add":
			return f"Undo attr add {a.path}@{a.attr_key}"
		if a.kind == "attr_del":
			return f"Undo attr delete {a.path}@{a.attr_key}"
		return "Undo"


# -------------------------
# Dataset table model
# -------------------------

@dataclass
class DatasetRef:
	file: h5py.File
	path: str

	@property
	def ds(self) -> h5py.Dataset:
		return self.file[self.path]  # type: ignore[index]


class H5DatasetTableModel(QAbstractTableModel):
	"""
	Supports scalar, 1D, 2D datasets.
	Locking is enforced via set_locked().
	"""

	def __init__(self, dref: DatasetRef, undo: UndoManager, parent=None):
		super().__init__(parent)
		self.dref = dref
		self.undo = undo
		self._cache = None
		self._dtype: Optional[np.dtype] = None
		self._locked = True

		self.reload()

	def set_locked(self, locked: bool):
		self._locked = locked
		# Update view flags
		self.layoutChanged.emit()

	def reload(self):
		try:
			ds = self.dref.ds
			self.beginResetModel()
			self._dtype = ds.dtype
			self._cache = np.array(ds[()])
			self.endResetModel()
		except Exception:
			self.beginResetModel()
			self._cache = None
			self._dtype = None
			self.endResetModel()

	def _view_dims(self) -> Tuple[int, int]:
		if self._cache is None:
			return (0, 0)
		arr = self._cache
		if arr.ndim == 0:
			return (1, 1)
		if arr.ndim == 1:
			return (arr.shape[0], 1)
		if arr.ndim == 2:
			return (arr.shape[0], arr.shape[1])
		return (0, 0)

	def rowCount(self, parent=QModelIndex()) -> int:
		if parent.isValid():
			return 0
		return self._view_dims()[0]

	def columnCount(self, parent=QModelIndex()) -> int:
		if parent.isValid():
			return 0
		return self._view_dims()[1]

	def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole):
		if not index.isValid() or self._cache is None:
			return QVariant()

		arr = self._cache
		r, c = index.row(), index.column()

		try:
			if arr.ndim == 0:
				val = arr.item()
			elif arr.ndim == 1:
				val = arr[r]
			else:
				val = arr[r, c]
		except Exception:
			return QVariant()

		if role in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole):
			if isinstance(val, (bytes, bytearray, np.bytes_)):
				try:
					return bytes(val).decode("utf-8")
				except Exception:
					return repr(val)
			if isinstance(val, np.generic):
				return str(val.item())
			return str(val)

		return QVariant()

	def flags(self, index: QModelIndex):
		if not index.isValid():
			return Qt.ItemFlag.NoItemFlags
		base = Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled
		if self._locked:
			return base
		return base | Qt.ItemFlag.ItemIsEditable

	def _to_h5_index(self, r: int, c: int) -> Tuple[int, ...]:
		ds = self.dref.ds
		if ds.ndim == 0:
			return tuple()
		if ds.ndim == 1:
			return (r,)
		return (r, c)

	def setData(self, index: QModelIndex, value: Any, role: int = Qt.ItemDataRole.EditRole) -> bool:
		if role != Qt.ItemDataRole.EditRole or not index.isValid() or self._cache is None:
			return False
		if self._locked:
			return False

		ds = self.dref.ds
		dt = ds.dtype
		text = str(value)

		try:
			coerced = _coerce_to_dtype(text, dt)
		except Exception as e:
			QMessageBox.critical(None, "Type conversion failed", str(e))
			return False

		r, c = index.row(), index.column()
		h5_index = self._to_h5_index(r, c)

		# Capture old value for undo
		try:
			if ds.ndim == 0:
				old = ds[()]
			elif ds.ndim == 1:
				old = ds[r]
			else:
				old = ds[r, c]
		except Exception as e:
			QMessageBox.critical(None, "Read failed", f"Could not read old value:\n{e}")
			return False

		try:
			if ds.ndim == 0:
				ds[()] = coerced
				self._cache[...] = np.array(coerced, dtype=dt)
			elif ds.ndim == 1:
				ds[r] = coerced
				self._cache[r] = np.array(coerced, dtype=dt)
			elif ds.ndim == 2:
				ds[r, c] = coerced
				self._cache[r, c] = np.array(coerced, dtype=dt)
			else:
				QMessageBox.warning(None, "Not supported", "Editing >2D datasets not supported.")
				return False
		except Exception as e:
			QMessageBox.critical(None, "Write failed", f"Could not write to dataset:\n{e}")
			return False

		# Record undo action AFTER success
		self.undo.push(UndoAction(
			kind="dataset_set",
			path=ds.name,
			index=h5_index,
			old_value=old,
			new_value=coerced
		))

		self.dataChanged.emit(index, index, [Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole])
		return True

	def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.ItemDataRole.DisplayRole):
		if role != Qt.ItemDataRole.DisplayRole:
			return QVariant()
		if orientation == Qt.Orientation.Horizontal:
			if self._cache is not None and self._cache.ndim == 1:
				return "value"
			return str(section)
		return str(section)


# -------------------------
# Attribute editor widget
# -------------------------

class AddAttributeDialog(QDialog):
	def __init__(self, parent=None):
		super().__init__(parent)
		self.setWindowTitle("Add Attribute")
		layout = QVBoxLayout(self)

		form = QFormLayout()
		self.key = QLineEdit()
		self.val = QLineEdit()
		form.addRow("Key", self.key)
		form.addRow("Value", self.val)
		layout.addLayout(form)

		buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
								  QDialogButtonBox.StandardButton.Cancel)
		buttons.accepted.connect(self.accept)
		buttons.rejected.connect(self.reject)
		layout.addWidget(buttons)

	def get_value(self) -> Tuple[str, str]:
		return self.key.text().strip(), self.val.text()


class DeleteAttributeDialog(QDialog):
	def __init__(self, keys: list[str], parent=None):
		super().__init__(parent)
		self.setWindowTitle("Delete Attribute")
		layout = QVBoxLayout(self)
		self.label = QLabel("Enter attribute key to delete:")
		layout.addWidget(self.label)
		self.key = QLineEdit()
		if keys:
			self.key.setPlaceholderText(f"e.g. {keys[0]}")
		layout.addWidget(self.key)

		buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
								  QDialogButtonBox.StandardButton.Cancel)
		buttons.accepted.connect(self.accept)
		buttons.rejected.connect(self.reject)
		layout.addWidget(buttons)

	def get_key(self) -> str:
		return self.key.text().strip()


class AttributeEditor(QWidget):
	"""
	Lock-aware attribute editor.
	Every successful mutation is recorded into UndoManager.
	"""

	def __init__(self, undo: UndoManager, parent=None):
		super().__init__(parent)

		self.undo = undo
		self._file: Optional[h5py.File] = None
		self._path: Optional[str] = None
		self._locked = True

		layout = QVBoxLayout(self)

		self.title = QLabel("Attributes")
		self.title.setStyleSheet("font-weight: 600;")
		layout.addWidget(self.title)

		self.form = QFormLayout()
		layout.addLayout(self.form)

		btn_row = QHBoxLayout()
		self.btn_add = QPushButton("Add")
		self.btn_del = QPushButton("Delete")
		self.btn_reload = QPushButton("Reload")
		btn_row.addWidget(self.btn_add)
		btn_row.addWidget(self.btn_del)
		btn_row.addWidget(self.btn_reload)
		btn_row.addStretch(1)
		layout.addLayout(btn_row)

		self.btn_add.clicked.connect(self._add_attr)
		self.btn_del.clicked.connect(self._delete_attr)
		self.btn_reload.clicked.connect(self.reload)

		self._editors: Dict[str, QLineEdit] = {}

		self._apply_lock_state()

	def set_locked(self, locked: bool):
		self._locked = locked
		self._apply_lock_state()

	def _apply_lock_state(self):
		self.btn_add.setEnabled(not self._locked)
		self.btn_del.setEnabled(not self._locked)
		# editors get enabled/disabled on reload since they’re created there
		for ed in self._editors.values():
			ed.setReadOnly(self._locked)

	def set_target(self, h5file: Optional[h5py.File], path: Optional[str]):
		self._file = h5file
		self._path = path
		self.reload()

	def _clear_form(self):
		while self.form.rowCount():
			self.form.removeRow(0)
		self._editors.clear()

	def reload(self):
		self._clear_form()

		if self._file is None or self._path is None:
			self.title.setText("Attributes (no selection)")
			return

		try:
			obj = self._file[self._path]  # type: ignore[index]
		except Exception as e:
			self.title.setText("Attributes (error)")
			show_error(self, "Selection error", "Could not open selected HDF5 object.", e)
			return

		self.title.setText(f"Attributes: {self._path}")

		try:
			keys = list(obj.attrs.keys())
		except Exception as e:
			show_error(self, "Attribute read failed", "Could not list attributes.", e)
			return

		for k in keys:
			try:
				v = obj.attrs.get(k)
			except Exception:
				v = "<unreadable>"
			le = QLineEdit(_format_attr_value(v))
			le.setReadOnly(self._locked)
			le.editingFinished.connect(self._make_attr_commit_fn(k, le))
			self.form.addRow(k, le)
			self._editors[k] = le

		self._apply_lock_state()

	def _make_attr_commit_fn(self, key: str, editor: QLineEdit):
		def _commit():
			if self._locked:
				return
			if self._file is None or self._path is None:
				return
			try:
				obj = self._file[self._path]  # type: ignore[index]
			except Exception as e:
				show_error(self, "Selection error", "Could not open selected object.", e)
				return

			try:
				existed = key in obj.attrs
				old = obj.attrs.get(key) if existed else None
			except Exception as e:
				show_error(self, "Attribute read failed", f"Could not read attribute '{key}'.", e)
				return

			new_text = editor.text()

			try:
				# preserve type as best as possible
				if isinstance(old, np.ndarray):
					parsed = _parse_scalar_text(new_text)
					obj.attrs.modify(key, parsed)
					new_val = parsed
				elif isinstance(old, np.generic):
					coerced = _coerce_to_dtype(new_text, old.dtype)
					obj.attrs.modify(key, coerced)
					new_val = coerced
				elif isinstance(old, (bytes, bytearray, np.bytes_)):
					b = new_text.encode("utf-8")
					obj.attrs.modify(key, b)
					new_val = b
				elif isinstance(old, str):
					obj.attrs.modify(key, new_text)
					new_val = new_text
				elif isinstance(old, (bool, int, float)):
					parsed = _parse_scalar_text(new_text)
					obj.attrs.modify(key, parsed)
					new_val = parsed
				else:
					parsed = _parse_scalar_text(new_text)
					obj.attrs.modify(key, parsed)
					new_val = parsed
			except Exception as e:
				show_error(self, "Attribute update failed", f"Failed to update '{key}'.", e)
				# revert view
				editor.setText(_format_attr_value(old))
				return

			# record undo AFTER success
			self.undo.push(UndoAction(
				kind="attr_set",
				path=obj.name,
				attr_key=key,
				existed_before=existed,
				old_value=old,
				new_value=new_val
			))

		return _commit

	def _add_attr(self):
		if self._locked:
			return
		if self._file is None or self._path is None:
			return
		try:
			obj = self._file[self._path]  # type: ignore[index]
		except Exception as e:
			show_error(self, "Selection error", "Could not open selected object.", e)
			return

		dlg = AddAttributeDialog(self)
		if dlg.exec() != QDialog.DialogCode.Accepted:
			return
		key, val = dlg.get_value()
		if not key:
			QMessageBox.warning(self, "Invalid key", "Attribute key cannot be empty.")
			return

		try:
			if key in obj.attrs:
				QMessageBox.warning(self, "Already exists", f"Attribute '{key}' already exists.")
				return
		except Exception as e:
			show_error(self, "Attribute check failed", "Could not check existing attributes.", e)
			return

		try:
			parsed = _parse_scalar_text(val)
			obj.attrs[key] = parsed
		except Exception as e:
			show_error(self, "Add attribute failed", f"Could not add '{key}'.", e)
			return

		self.undo.push(UndoAction(
			kind="attr_add",
			path=obj.name,
			attr_key=key,
			new_value=parsed
		))

		self.reload()

	def _delete_attr(self):
		if self._locked:
			return
		if self._file is None or self._path is None:
			return
		try:
			obj = self._file[self._path]  # type: ignore[index]
		except Exception as e:
			show_error(self, "Selection error", "Could not open selected object.", e)
			return

		try:
			keys = list(obj.attrs.keys())
		except Exception as e:
			show_error(self, "Attribute read failed", "Could not list attributes.", e)
			return

		if not keys:
			return

		dlg = DeleteAttributeDialog(keys, self)
		if dlg.exec() != QDialog.DialogCode.Accepted:
			return
		key = dlg.get_key()
		if not key:
			return

		try:
			if key not in obj.attrs:
				QMessageBox.warning(self, "Not found", f"No attribute named '{key}'.")
				return
			old = obj.attrs.get(key)
			del obj.attrs[key]
		except Exception as e:
			show_error(self, "Delete attribute failed", f"Could not delete '{key}'.", e)
			return

		self.undo.push(UndoAction(
			kind="attr_del",
			path=obj.name,
			attr_key=key,
			deleted_value=old
		))

		self.reload()


# -------------------------
# Main window
# -------------------------

class HDF5EditorMainWindow(QMainWindow):
	def __init__(self):
		super().__init__()
		self.setWindowTitle("HDF5 Viewer/Editor (PyQt6)")
		self.resize(1200, 720)

		self._h5: Optional[h5py.File] = None
		self._filename: Optional[str] = None

		self.undo = UndoManager(limit=1000)
		self.undo.changed.connect(self._update_undo_button)

		# lock state
		self._locked = True  # ON by default

		# Actions
		self.act_open = QAction("Open…", self)
		self.act_close = QAction("Close", self)
		self.act_exit = QAction("Exit", self)

		self.act_open.triggered.connect(self.open_file)
		self.act_close.triggered.connect(self.close_file)
		self.act_exit.triggered.connect(self.close)

		menu = self.menuBar().addMenu("File")
		menu.addAction(self.act_open)
		menu.addAction(self.act_close)
		menu.addSeparator()
		menu.addAction(self.act_exit)

		# Layout
		splitter = QSplitter()
		splitter.setOrientation(Qt.Orientation.Horizontal)

		# Left: tree
		self.tree = QTreeWidget()
		self.tree.setHeaderLabels(["HDF5 Path", "Type", "Shape", "DType"])
		self.tree.itemSelectionChanged.connect(self._on_tree_selection)
		splitter.addWidget(self.tree)

		# Right: details
		right = QWidget()
		right_layout = QVBoxLayout(right)

		# Top controls row: lock + undo
		controls = QHBoxLayout()

		self.btn_lock = QToolButton()
		self.btn_lock.setCheckable(True)
		self.btn_lock.setChecked(True)  # locked by default
		self.btn_lock.clicked.connect(self._toggle_lock)
		controls.addWidget(self.btn_lock)

		self.btn_undo = QPushButton("Undo")
		self.btn_undo.clicked.connect(self.undo_last)
		controls.addWidget(self.btn_undo)

		controls.addStretch(1)
		right_layout.addLayout(controls)

		self.info = QLabel("Open an HDF5 file to begin.")
		right_layout.addWidget(self.info)
		
		# Window controls (paged viewing)
		winrow = QHBoxLayout()
		self.row0_edit = QLineEdit("0")
		self.col0_edit = QLineEdit("0")
		self.btn_jump = QPushButton("Load window")
		winrow.addWidget(QLabel("Row start:"))
		winrow.addWidget(self.row0_edit)
		winrow.addWidget(QLabel("Col start:"))
		winrow.addWidget(self.col0_edit)
		winrow.addWidget(self.btn_jump)
		winrow.addStretch(1)
		right_layout.addLayout(winrow)

		self.btn_jump.clicked.connect(self._jump_window)
		
		self.dataset_view = QTableView()
		self.dataset_view.setEditTriggers(
			QAbstractItemView.EditTrigger.DoubleClicked |
			QAbstractItemView.EditTrigger.SelectedClicked |
			QAbstractItemView.EditTrigger.EditKeyPressed
		)
		self.dataset_view.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
		self.dataset_view.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Fixed)
		self.dataset_view.setMinimumHeight(260)
		right_layout.addWidget(self.dataset_view)

		self.attr_editor = AttributeEditor(self.undo)
		right_layout.addWidget(self.attr_editor)

		splitter.addWidget(right)
		splitter.setStretchFactor(0, 2)
		splitter.setStretchFactor(1, 3)

		self.setCentralWidget(splitter)

		self._current_dataset_model: Optional[H5DatasetTableModel] = None

		self._apply_lock_ui()
		self._update_undo_button()
		
		self.setAcceptDrops(True)

	# ---- Lock handling ----

	def _toggle_lock(self):
		# Checked == locked
		self._locked = bool(self.btn_lock.isChecked())
		self._apply_lock_ui()

	def _apply_lock_ui(self):
		if self._locked:
			self.btn_lock.setText("Locked (click to unlock)")
		else:
			self.btn_lock.setText("Unlocked (click to lock)")

		# Dataset model
		if self._current_dataset_model is not None:
			self._current_dataset_model.set_locked(self._locked)

		# Attributes
		self.attr_editor.set_locked(self._locked)

		# Disable undo while locked? You asked lock to prevent editing;
		# undo is also a write, so it should be blocked unless unlocked.
		self.btn_undo.setEnabled((not self._locked) and self.undo.can_undo())
		self.btn_undo.setText(self.undo.peek_summary() if (not self._locked) else "Undo (unlock to enable)")

	def _jump_window(self):
		if self._current_dataset_model is None:
			return
		try:
			r0 = int(self.row0_edit.text().strip() or "0")
			c0 = int(self.col0_edit.text().strip() or "0")
		except Exception:
			QMessageBox.warning(self, "Invalid input", "Row/col start must be integers.")
			return

		if hasattr(self._current_dataset_model, "set_window"):
			try:
				self._current_dataset_model.set_window(r0, c0)
				self.dataset_view.resizeColumnsToContents()
			except Exception as e:
				show_error(self, "Window load failed", "Could not load dataset window.", e)


	def _update_undo_button(self):
		if self._locked:
			self.btn_undo.setEnabled(False)
			self.btn_undo.setText("Undo (unlock to enable)")
		else:
			self.btn_undo.setEnabled(self.undo.can_undo())
			self.btn_undo.setText(self.undo.peek_summary())

	#----- drag and drop ----
	
	def dragEnterEvent(self, event):
		try:
			md = event.mimeData()
			if md and md.hasUrls():
				# Accept if any url looks like a local HDF5-ish file
				for url in md.urls():
					if not url.isLocalFile():
						continue
					p = url.toLocalFile().lower()
					if p.endswith((".h5", ".hdf5", ".hdf", ".he5")):
						event.acceptProposedAction()
						return
		except Exception:
			pass
		event.ignore()


	def dropEvent(self, event):
		try:
			md = event.mimeData()
			if not (md and md.hasUrls()):
				event.ignore()
				return

			# Pick the first valid local file
			for url in md.urls():
				if not url.isLocalFile():
					continue
				fn = url.toLocalFile()
				low = fn.lower()
				if low.endswith((".h5", ".hdf5", ".hdf", ".he5")):
					try:
						self._open_h5(fn)  # reuse your existing open logic
					except Exception as e:
						show_error(self, "Open failed", f"Could not open dropped file:\n{fn}", e)
					event.acceptProposedAction()
					return

		except Exception as e:
			show_error(self, "Drop failed", "Unexpected error handling dropped file.", e)

		event.ignore()


	# ---- File operations ----

	def open_file(self):
		fn, _ = QFileDialog.getOpenFileName(
			self, "Open HDF5 file", "", "HDF5 Files (*.h5 *.hdf5 *.hdf *.he5);;All Files (*)"
		)
		if not fn:
			return
		try:
			self._open_h5(fn)
		except Exception as e:
			show_error(self, "Open failed", f"Could not open file:\n{fn}", e)

	def _open_h5(self, filename: str):
		self.close_file()

		# Open read/write
		self._h5 = h5py.File(filename, "r+")
		self._filename = filename
		self.setWindowTitle(f"HDF5 Viewer/Editor - {filename}")
		self.info.setText(f"Opened: {filename}")
		self.undo.clear()
		self._populate_tree()

	def close_file(self):
		# UI reset
		self.tree.blockSignals(True)
		self.tree.clear()
		self.tree.blockSignals(False)

		self.dataset_view.setModel(None)
		self._current_dataset_model = None
		self.attr_editor.set_target(None, None)
		self.info.setText("Open an HDF5 file to begin.")
		self.undo.clear()

		if self._h5 is not None:
			try:
				self._h5.flush()
			except Exception:
				pass
			try:
				self._h5.close()
			except Exception:
				pass
		self._h5 = None
		self._filename = None
		self.setWindowTitle("HDF5 Viewer/Editor (PyQt6)")

		self._apply_lock_ui()

	# ---- Tree population ----

	def _populate_tree(self):
		self.tree.clear()
		if self._h5 is None:
			return

		try:
			root_item = QTreeWidgetItem(["/", "Group", "", ""])
			root_item.setData(0, Qt.ItemDataRole.UserRole, "/")
			self.tree.addTopLevelItem(root_item)

			self._populate_children(root_item, self._h5["/"])
			root_item.setExpanded(True)

			self.tree.resizeColumnToContents(0)
			self.tree.resizeColumnToContents(1)
		except Exception as e:
			show_error(self, "Tree build failed", "Could not populate HDF5 tree.", e)

	def _populate_children(self, parent_item: QTreeWidgetItem, group: h5py.Group):
		try:
			names = sorted(group.keys())
		except Exception as e:
			raise RuntimeError(f"Could not list group keys for {group.name}") from e

		for name in names:
			try:
				obj = group[name]
				path = obj.name
			except Exception:
				# Keep going even if one entry is problematic
				it = QTreeWidgetItem([f"{group.name}/{name}", "Unreadable", "", ""])
				parent_item.addChild(it)
				continue

			if isinstance(obj, h5py.Group):
				it = QTreeWidgetItem([path, "Group", "", ""])
				it.setData(0, Qt.ItemDataRole.UserRole, path)
				parent_item.addChild(it)
				# recurse
				try:
					self._populate_children(it, obj)
				except Exception:
					# don't crash on a bad subgroup
					pass

			elif isinstance(obj, h5py.Dataset):
				try:
					shape = str(obj.shape)
					dtype = str(obj.dtype)
				except Exception:
					shape = "?"
					dtype = "?"
				it = QTreeWidgetItem([path, "Dataset", shape, dtype])
				it.setData(0, Qt.ItemDataRole.UserRole, path)
				parent_item.addChild(it)

			else:
				it = QTreeWidgetItem([path, type(obj).__name__, "", ""])
				it.setData(0, Qt.ItemDataRole.UserRole, path)
				parent_item.addChild(it)

	# ---- Selection handling ----

	def _on_tree_selection(self):
		if self._h5 is None:
			return

		items = self.tree.selectedItems()
		if not items:
			return

		item = items[0]
		path = item.data(0, Qt.ItemDataRole.UserRole)
		if not path:
			return

		# Open object safely
		try:
			obj = self._h5[path]  # type: ignore[index]
		except Exception as e:
			show_error(self, "Selection error", f"Could not open selected HDF5 object:\n{path}", e)
			return

		# Attributes (lock-aware)
		try:
			self.attr_editor.set_target(self._h5, path)
			self.attr_editor.set_locked(self._locked)
		except Exception as e:
			show_error(self, "Attribute view error", "Could not update attribute editor.", e)

		# Clear dataset view by default; we’ll set it below if it’s a dataset
		self.dataset_view.setModel(None)
		self._current_dataset_model = None

		# Non-dataset selection
		if not isinstance(obj, h5py.Dataset):
			try:
				if isinstance(obj, h5py.Group):
					self.info.setText(f"Selected: {path} (Group)")
				else:
					self.info.setText(f"Selected: {path} ({type(obj).__name__})")
			except Exception:
				self.info.setText(f"Selected: {path}")
			self._apply_lock_ui()
			return

		# Dataset selection
		ds = obj
		try:
			self.info.setText(f"Selected dataset: {path} | shape={ds.shape} dtype={ds.dtype}")
		except Exception:
			self.info.setText(f"Selected dataset: {path}")

		# Only support 0/1/2-D table view
		try:
			if ds.ndim not in (0, 1, 2):
				QMessageBox.information(
					self,
					"Dataset view limited",
					f"This dataset has ndim={ds.ndim}. The table editor supports only 0/1/2-D.\n\n"
					f"{ds.name}\nshape={ds.shape}\ndtype={ds.dtype}"
				)
				self._apply_lock_ui()
				return
		except Exception as e:
			show_error(self, "Dataset metadata error", "Could not read dataset dimensionality.", e)
			self._apply_lock_ui()
			return

		# Large dataset guard thresholds (tweak to taste)
		MAX_PREVIEW_CELLS = 200_000
		MAX_PREVIEW_BYTES = 64 * 1024 * 1024  # 64 MiB

		# Estimate dataset size
		try:
			n_cells = int(ds.size)
			# dtype.itemsize works for fixed-size types; for vlen/object, this will be misleading, but still ok as a guard.
			itemsize = int(getattr(ds.dtype, "itemsize", 0) or 0)
			n_bytes = n_cells * itemsize
		except Exception:
			n_cells = 0
			n_bytes = 0

		# Default behavior: use paged model (fast/safe)
		use_paged = True

		# Ask user if big (or if itemsize unknown but cells huge)
		try:
			is_big = (n_cells > MAX_PREVIEW_CELLS) or (n_bytes > MAX_PREVIEW_BYTES) or (itemsize == 0 and n_cells > MAX_PREVIEW_CELLS)
		except Exception:
			is_big = True

		if is_big:
			mb = (n_bytes / (1024 * 1024)) if n_bytes else 0.0
			size_line = f"Approx size: {mb:.1f} MB" if n_bytes else "Approx size: unknown"
			msg = (
				"Large dataset detected:\n\n"
				f"Path: {ds.name}\n"
				f"Shape: {ds.shape}\n"
				f"DType: {ds.dtype}\n"
				f"Elements: {n_cells:,}\n"
				f"{size_line}\n\n"
				"Loading the entire dataset may freeze the GUI.\n"
				"How do you want to proceed?"
			)

			box = QMessageBox(self)
			box.setIcon(QMessageBox.Icon.Warning)
			box.setWindowTitle("Large dataset")
			box.setText(msg)

			preview_btn = box.addButton("Preview window", QMessageBox.ButtonRole.AcceptRole)
			loadall_btn = box.addButton("Load all anyway", QMessageBox.ButtonRole.DestructiveRole)
			cancel_btn = box.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)

			box.exec()
			clicked = box.clickedButton()

			if clicked == cancel_btn:
				# leave cleared view
				self._apply_lock_ui()
				return
			elif clicked == loadall_btn:
				use_paged = False
			else:
				use_paged = True

		# Build the model
		try:
			dref = DatasetRef(self._h5, ds.name)

			if use_paged:
				# Use your paged model defaults; these should match your Step 3 controls
				model = H5PagedDatasetTableModel(
					dref,
					self.undo,
					view_rows=500,
					view_cols=200,
				)
				# Optionally sync row0/col0 controls to 0 on new selection:
				try:
					if hasattr(self, "row0_edit"):
						self.row0_edit.setText("0")
					if hasattr(self, "col0_edit"):
						self.col0_edit.setText("0")
				except Exception:
					pass
			else:
				# user asked for full-load model (can freeze on huge datasets)
				model = H5DatasetTableModel(dref, self.undo)

			model.set_locked(self._locked)
			self._current_dataset_model = model
			self.dataset_view.setModel(model)
			self.dataset_view.resizeColumnsToContents()

		except Exception as e:
			show_error(self, "Dataset view failed", "Could not display dataset.", e)
			self.dataset_view.setModel(None)
			self._current_dataset_model = None

		self._apply_lock_ui()


	# ---- Undo ----

	def undo_last(self):
		if self._locked:
			QMessageBox.information(self, "Locked", "Unlock to undo changes.")
			return
		if self._h5 is None:
			return

		act = self.undo.pop()
		if act is None:
			return

		try:
			obj = self._h5[act.path]  # type: ignore[index]
		except Exception as e:
			show_error(self, "Undo failed", f"Could not open object for undo:\n{act.path}", e)
			return

		try:
			if act.kind == "dataset_set":
				if not isinstance(obj, h5py.Dataset):
					raise TypeError("Undo target is not a dataset.")
				ds = obj
				# act.index == () for scalar
				if act.index is None or act.index == tuple():
					ds[()] = act.old_value
				else:
					ds[act.index] = act.old_value

				# refresh view if current dataset is this one
				if self._current_dataset_model is not None and self._current_dataset_model.dref.path == act.path:
					self._current_dataset_model.reload()

			elif act.kind == "attr_set":
				if act.attr_key is None:
					raise ValueError("Missing attr_key")
				key = act.attr_key
				if act.existed_before:
					obj.attrs.modify(key, act.old_value)
				else:
					# didn't exist before => remove it
					if key in obj.attrs:
						del obj.attrs[key]

				# refresh attrs if current selection is this object
				self._refresh_if_selected(act.path)

			elif act.kind == "attr_add":
				if act.attr_key is None:
					raise ValueError("Missing attr_key")
				key = act.attr_key
				# added => undo by deleting
				if key in obj.attrs:
					del obj.attrs[key]
				self._refresh_if_selected(act.path)

			elif act.kind == "attr_del":
				if act.attr_key is None:
					raise ValueError("Missing attr_key")
				key = act.attr_key
				# deleted => undo by restoring
				obj.attrs[key] = act.deleted_value
				self._refresh_if_selected(act.path)

			else:
				raise ValueError(f"Unknown undo action kind: {act.kind}")

		except Exception as e:
			show_error(self, "Undo failed", "Undo operation failed.", e)

		self._apply_lock_ui()

	def _refresh_if_selected(self, path: str):
		# If the currently selected item is path, reload attr editor
		items = self.tree.selectedItems()
		if not items:
			return
		cur_path = items[0].data(0, Qt.ItemDataRole.UserRole)
		if cur_path == path:
			self.attr_editor.reload()

	def closeEvent(self, event):
		self.close_file()
		super().closeEvent(event)


def main():
	app = QApplication(sys.argv)
	win = HDF5EditorMainWindow()
	win.show()
	sys.exit(app.exec())


if __name__ == "__main__":
	main()


# #!/usr/bin/env python3
# """
# PyQt6 HDF5 Viewer/Editor
# 
# Features:
# - Tree view of groups/datasets, expand/collapse
# - Attribute viewer/editor for any object
# - Dataset table viewer/editor for 0/1/2-D datasets (editable)
# - Basic type conversion on attribute/dataset edit where possible
# 
# Notes / limitations (by design, to stay robust):
# - For datasets: edits are in-place only (no resizing, no dtype change).
# - Supports 0/1/2-D datasets. Higher dims are shown as a brief summary.
# - For object/reference/compound/variable-length exotic dtypes: view may be limited.
# """
# 
# from __future__ import annotations
# 
# import sys
# import traceback
# from dataclasses import dataclass
# from typing import Any, Optional, Tuple
# 
# import numpy as np
# import h5py
# 
# from PyQt6.QtCore import (
#     Qt, QAbstractTableModel, QModelIndex, QVariant
# )
# from PyQt6.QtGui import QAction
# from PyQt6.QtWidgets import (
#     QApplication, QMainWindow, QFileDialog, QMessageBox,
#     QTreeWidget, QTreeWidgetItem, QSplitter, QWidget, QVBoxLayout,
#     QTableView, QHeaderView, QLabel, QFormLayout, QLineEdit,
#     QHBoxLayout, QPushButton, QAbstractItemView, QDialog, QDialogButtonBox
# )
# 
# 
# # -------------------------
# # Helpers: type conversion
# # -------------------------
# 
# def _is_bytes_dtype(dt: np.dtype) -> bool:
#     # fixed-length bytes string: '|S...' or 'S...'
#     return dt.kind == "S"
# 
# def _is_unicode_dtype(dt: np.dtype) -> bool:
#     # fixed-length unicode: 'U...'
#     return dt.kind == "U"
# 
# def _is_numeric_dtype(dt: np.dtype) -> bool:
#     return dt.kind in ("i", "u", "f", "c")  # int, uint, float, complex
# 
# def _parse_scalar_text(text: str) -> Any:
#     """
#     Parse a scalar from user input. Tries:
#     - int
#     - float
#     - bool literals
#     - fallback to string
#     """
#     t = text.strip()
#     if t.lower() in ("true", "false"):
#         return t.lower() == "true"
#     # try int
#     try:
#         if t.startswith("0x") or t.startswith("0X"):
#             return int(t, 16)
#         return int(t)
#     except Exception:
#         pass
#     # try float
#     try:
#         return float(t)
#     except Exception:
#         pass
#     return t
# 
# def _coerce_to_dtype(value: Any, dt: np.dtype) -> Any:
#     """
#     Attempt to coerce python value into dt. Raises ValueError if impossible.
#     Handles:
#     - numeric -> numeric
#     - str -> numeric where parseable
#     - bytes/str -> fixed-length strings
#     - bool -> numeric/bool
#     """
#     if isinstance(value, str):
#         # attempt parse if numeric/bool
#         parsed = _parse_scalar_text(value)
#     else:
#         parsed = value
# 
#     if _is_numeric_dtype(dt):
#         try:
#             return np.array(parsed, dtype=dt).item()
#         except Exception as e:
#             raise ValueError(f"Cannot convert {parsed!r} to dtype {dt}") from e
# 
#     if dt.kind == "b":  # boolean
#         if isinstance(parsed, bool):
#             return bool(parsed)
#         if isinstance(parsed, (int, float, np.integer, np.floating)):
#             return bool(parsed)
#         if isinstance(parsed, str):
#             if parsed.strip().lower() in ("true", "1", "yes", "y", "t"):
#                 return True
#             if parsed.strip().lower() in ("false", "0", "no", "n", "f"):
#                 return False
#         raise ValueError(f"Cannot convert {parsed!r} to bool")
# 
#     if _is_bytes_dtype(dt):
#         # fixed-size bytes: need to encode and fit
#         b = parsed
#         if isinstance(b, str):
#             b = b.encode("utf-8")
#         elif isinstance(b, (bytes, bytearray)):
#             b = bytes(b)
#         else:
#             b = str(b).encode("utf-8")
#         maxlen = dt.itemsize
#         if len(b) > maxlen:
#             raise ValueError(f"String too long for dtype {dt} (max {maxlen} bytes)")
#         # pad or leave? h5py stores fixed-len bytes as-is; numpy will pad with nulls.
#         return np.array(b, dtype=dt).tobytes()
# 
#     if _is_unicode_dtype(dt):
#         s = parsed if isinstance(parsed, str) else str(parsed)
#         maxlen = dt.itemsize // 4  # rough; numpy U uses 4 bytes/char
#         if len(s) > maxlen:
#             raise ValueError(f"String too long for dtype {dt} (max {maxlen} chars)")
#         return np.array(s, dtype=dt).item()
# 
#     # object / compound / other:
#     raise ValueError(f"Editing for dtype {dt} not supported")
# 
# 
# def _format_attr_value(v: Any) -> str:
#     """
#     Display attributes in a single-line, editable form.
#     If it's a numpy scalar/array, show a reasonable representation.
#     """
#     if isinstance(v, (np.ndarray,)):
#         if v.ndim == 0:
#             return str(v.item())
#         # Keep it compact
#         return np.array2string(v, threshold=20, edgeitems=3)
#     if isinstance(v, (bytes, bytearray)):
#         # show as utf-8 if possible
#         try:
#             return bytes(v).decode("utf-8")
#         except Exception:
#             return repr(v)
#     return str(v)
# 
# 
# # -------------------------
# # Dataset table model
# # -------------------------
# 
# @dataclass
# class DatasetRef:
#     file: h5py.File
#     path: str
# 
#     @property
#     def ds(self) -> h5py.Dataset:
#         return self.file[self.path]  # type: ignore[index]
# 
# 
# class H5DatasetTableModel(QAbstractTableModel):
#     """
#     Simple table model over a dataset. Supports:
#     - scalar: shown as 1x1
#     - 1D: shown as Nx1
#     - 2D: shown as NxM
#     """
# 
#     def __init__(self, dref: DatasetRef, parent=None):
#         super().__init__(parent)
#         self.dref = dref
#         self._cache = None  # loaded numpy array for display/edit
#         self._shape = ()
#         self._dtype = None
# 
#         self.reload()
# 
#     def reload(self):
#         ds = self.dref.ds
#         self.beginResetModel()
#         self._shape = tuple(ds.shape)
#         self._dtype = ds.dtype
#         # Load data for view/edit; keep it simple and local.
#         try:
#             arr = ds[()]  # scalar or array
#             self._cache = np.array(arr)
#         except Exception:
#             self._cache = None
#         self.endResetModel()
# 
#     def dataset_summary(self) -> str:
#         ds = self.dref.ds
#         return f"path={self.dref.path}, shape={ds.shape}, dtype={ds.dtype}"
# 
#     def _view_dims(self) -> Tuple[int, int]:
#         if self._cache is None:
#             return (0, 0)
#         arr = self._cache
#         if arr.ndim == 0:
#             return (1, 1)
#         if arr.ndim == 1:
#             return (arr.shape[0], 1)
#         if arr.ndim == 2:
#             return (arr.shape[0], arr.shape[1])
#         return (0, 0)
# 
#     def rowCount(self, parent=QModelIndex()) -> int:
#         if parent.isValid():
#             return 0
#         return self._view_dims()[0]
# 
#     def columnCount(self, parent=QModelIndex()) -> int:
#         if parent.isValid():
#             return 0
#         return self._view_dims()[1]
# 
#     def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole):
#         if not index.isValid() or self._cache is None:
#             return QVariant()
#         arr = self._cache
# 
#         r, c = index.row(), index.column()
#         if arr.ndim == 0:
#             val = arr.item()
#         elif arr.ndim == 1:
#             val = arr[r]
#         else:  # 2D
#             val = arr[r, c]
# 
#         if role in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole):
#             if isinstance(val, (bytes, bytearray, np.bytes_)):
#                 try:
#                     return bytes(val).decode("utf-8")
#                 except Exception:
#                     return repr(val)
#             if isinstance(val, (np.generic,)):
#                 # numpy scalar -> python scalar for cleaner display
#                 return str(val.item())
#             return str(val)
# 
#         return QVariant()
# 
#     def flags(self, index: QModelIndex):
#         if not index.isValid():
#             return Qt.ItemFlag.NoItemFlags
#         return (Qt.ItemFlag.ItemIsSelectable |
#                 Qt.ItemFlag.ItemIsEnabled |
#                 Qt.ItemFlag.ItemIsEditable)
# 
#     def setData(self, index: QModelIndex, value: Any, role: int = Qt.ItemDataRole.EditRole) -> bool:
#         if role != Qt.ItemDataRole.EditRole or not index.isValid() or self._cache is None:
#             return False
# 
#         ds = self.dref.ds
#         dt = ds.dtype
# 
#         text = str(value)
# 
#         try:
#             coerced = _coerce_to_dtype(text, dt)
#         except Exception as e:
#             QMessageBox.critical(None, "Type conversion failed", str(e))
#             return False
# 
#         r, c = index.row(), index.column()
#         try:
#             # Write into dataset (in-place)
#             if ds.ndim == 0:
#                 ds[()] = coerced
#                 self._cache[...] = np.array(coerced, dtype=dt)
#             elif ds.ndim == 1:
#                 ds[r] = coerced
#                 self._cache[r] = np.array(coerced, dtype=dt)
#             elif ds.ndim == 2:
#                 ds[r, c] = coerced
#                 self._cache[r, c] = np.array(coerced, dtype=dt)
#             else:
#                 QMessageBox.warning(None, "Not supported", "Editing >2D datasets not supported.")
#                 return False
#         except Exception as e:
#             QMessageBox.critical(None, "Write failed", f"Could not write to dataset:\n{e}")
#             return False
# 
#         self.dataChanged.emit(index, index, [Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole])
#         return True
# 
#     def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.ItemDataRole.DisplayRole):
#         if role != Qt.ItemDataRole.DisplayRole:
#             return QVariant()
#         if orientation == Qt.Orientation.Horizontal:
#             # For 1D datasets, show a single column named "value"
#             if self._cache is not None and self._cache.ndim == 1:
#                 return "value"
#             return str(section)
#         return str(section)
# 
# 
# # -------------------------
# # Attribute editor widget
# # -------------------------
# 
# class AttributeEditor(QWidget):
#     """
#     Shows attributes of an object, allows editing and adding/removing.
#     """
# 
#     def __init__(self, parent=None):
#         super().__init__(parent)
# 
#         self._file: Optional[h5py.File] = None
#         self._path: Optional[str] = None
# 
#         layout = QVBoxLayout(self)
# 
#         self.title = QLabel("Attributes")
#         self.title.setStyleSheet("font-weight: 600;")
#         layout.addWidget(self.title)
# 
#         self.form = QFormLayout()
#         layout.addLayout(self.form)
# 
#         btn_row = QHBoxLayout()
#         self.btn_add = QPushButton("Add")
#         self.btn_del = QPushButton("Delete")
#         self.btn_reload = QPushButton("Reload")
#         btn_row.addWidget(self.btn_add)
#         btn_row.addWidget(self.btn_del)
#         btn_row.addWidget(self.btn_reload)
#         btn_row.addStretch(1)
#         layout.addLayout(btn_row)
# 
#         self.btn_add.clicked.connect(self._add_attr)
#         self.btn_del.clicked.connect(self._delete_attr)
#         self.btn_reload.clicked.connect(self.reload)
# 
#         self._editors: dict[str, QLineEdit] = {}
# 
#     def set_target(self, h5file: Optional[h5py.File], path: Optional[str]):
#         self._file = h5file
#         self._path = path
#         self.reload()
# 
#     def _clear_form(self):
#         while self.form.rowCount():
#             self.form.removeRow(0)
#         self._editors.clear()
# 
#     def reload(self):
#         self._clear_form()
# 
#         if self._file is None or self._path is None:
#             self.title.setText("Attributes (no selection)")
#             return
# 
#         obj = self._file[self._path]  # type: ignore[index]
#         self.title.setText(f"Attributes: {self._path}")
# 
#         # List all attributes
#         for k in obj.attrs.keys():
#             v = obj.attrs.get(k)
#             le = QLineEdit(_format_attr_value(v))
#             le.editingFinished.connect(self._make_attr_commit_fn(k, le))
#             self.form.addRow(k, le)
#             self._editors[k] = le
# 
#     def _make_attr_commit_fn(self, key: str, editor: QLineEdit):
#         def _commit():
#             if self._file is None or self._path is None:
#                 return
#             obj = self._file[self._path]  # type: ignore[index]
#             old = obj.attrs.get(key)
#             new_text = editor.text()
# 
#             # Try to preserve type if possible
#             try:
#                 if isinstance(old, np.ndarray):
#                     # Try parse as scalar -> broadcast not supported; keep as text if can't parse.
#                     parsed = _parse_scalar_text(new_text)
#                     obj.attrs.modify(key, parsed)
#                 elif isinstance(old, (np.generic,)):
#                     coerced = _coerce_to_dtype(new_text, old.dtype)
#                     obj.attrs.modify(key, coerced)
#                 elif isinstance(old, (bytes, bytearray)):
#                     obj.attrs.modify(key, new_text.encode("utf-8"))
#                 elif isinstance(old, (str,)):
#                     obj.attrs.modify(key, new_text)
#                 elif isinstance(old, (bool, int, float)):
#                     obj.attrs.modify(key, _parse_scalar_text(new_text))
#                 else:
#                     # unknown type, store as string
#                     obj.attrs.modify(key, new_text)
#             except Exception as e:
#                 QMessageBox.critical(self, "Attribute update failed", f"{key}: {e}")
#                 # revert view
#                 editor.setText(_format_attr_value(old))
#         return _commit
# 
#     def _add_attr(self):
#         if self._file is None or self._path is None:
#             return
#         dlg = AddAttributeDialog(self)
#         if dlg.exec() != QDialog.DialogCode.Accepted:
#             return
#         key, val = dlg.get_value()
#         obj = self._file[self._path]  # type: ignore[index]
#         try:
#             # Store parsed scalar if it looks numeric/bool; else string
#             parsed = _parse_scalar_text(val)
#             obj.attrs[key] = parsed
#         except Exception as e:
#             QMessageBox.critical(self, "Add attribute failed", str(e))
#             return
#         self.reload()
# 
#     def _delete_attr(self):
#         if self._file is None or self._path is None:
#             return
#         obj = self._file[self._path]  # type: ignore[index]
#         keys = list(obj.attrs.keys())
#         if not keys:
#             return
#         dlg = DeleteAttributeDialog(keys, self)
#         if dlg.exec() != QDialog.DialogCode.Accepted:
#             return
#         key = dlg.get_key()
#         try:
#             del obj.attrs[key]
#         except Exception as e:
#             QMessageBox.critical(self, "Delete attribute failed", str(e))
#             return
#         self.reload()
# 
# 
# class AddAttributeDialog(QDialog):
#     def __init__(self, parent=None):
#         super().__init__(parent)
#         self.setWindowTitle("Add Attribute")
#         layout = QVBoxLayout(self)
# 
#         form = QFormLayout()
#         self.key = QLineEdit()
#         self.val = QLineEdit()
#         form.addRow("Key", self.key)
#         form.addRow("Value", self.val)
#         layout.addLayout(form)
# 
#         buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
#                                   QDialogButtonBox.StandardButton.Cancel)
#         buttons.accepted.connect(self.accept)
#         buttons.rejected.connect(self.reject)
#         layout.addWidget(buttons)
# 
#     def get_value(self) -> Tuple[str, str]:
#         return self.key.text().strip(), self.val.text()
# 
# 
# class DeleteAttributeDialog(QDialog):
#     def __init__(self, keys: list[str], parent=None):
#         super().__init__(parent)
#         self.setWindowTitle("Delete Attribute")
#         layout = QVBoxLayout(self)
#         self.label = QLabel("Enter attribute key to delete:")
#         layout.addWidget(self.label)
#         self.key = QLineEdit()
#         if keys:
#             self.key.setPlaceholderText(f"e.g. {keys[0]}")
#         layout.addWidget(self.key)
# 
#         buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
#                                   QDialogButtonBox.StandardButton.Cancel)
#         buttons.accepted.connect(self.accept)
#         buttons.rejected.connect(self.reject)
#         layout.addWidget(buttons)
# 
#     def get_key(self) -> str:
#         return self.key.text().strip()
# 
# 
# # -------------------------
# # Main window
# # -------------------------
# 
# class HDF5EditorMainWindow(QMainWindow):
#     def __init__(self):
#         super().__init__()
#         self.setWindowTitle("HDF5 Viewer/Editor (PyQt6)")
#         self.resize(1200, 700)
# 
#         self._h5: Optional[h5py.File] = None
#         self._filename: Optional[str] = None
# 
#         # Actions
#         self.act_open = QAction("Open…", self)
#         self.act_close = QAction("Close", self)
#         self.act_exit = QAction("Exit", self)
# 
#         self.act_open.triggered.connect(self.open_file)
#         self.act_close.triggered.connect(self.close_file)
#         self.act_exit.triggered.connect(self.close)
# 
#         menu = self.menuBar().addMenu("File")
#         menu.addAction(self.act_open)
#         menu.addAction(self.act_close)
#         menu.addSeparator()
#         menu.addAction(self.act_exit)
# 
#         # UI layout
#         splitter = QSplitter()
#         splitter.setOrientation(Qt.Orientation.Horizontal)
# 
#         # Left: tree
#         self.tree = QTreeWidget()
#         self.tree.setHeaderLabels(["HDF5 Path", "Type", "Shape", "DType"])
#         self.tree.itemSelectionChanged.connect(self._on_tree_selection)
#         splitter.addWidget(self.tree)
# 
#         # Right: details (dataset + attrs)
#         right = QWidget()
#         right_layout = QVBoxLayout(right)
# 
#         self.info = QLabel("Open an HDF5 file to begin.")
#         right_layout.addWidget(self.info)
# 
#         self.dataset_view = QTableView()
#         self.dataset_view.setEditTriggers(QAbstractItemView.EditTrigger.DoubleClicked |
#                                           QAbstractItemView.EditTrigger.SelectedClicked |
#                                           QAbstractItemView.EditTrigger.EditKeyPressed)
#         self.dataset_view.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
#         self.dataset_view.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Fixed)
#         self.dataset_view.setMinimumHeight(250)
#         right_layout.addWidget(self.dataset_view)
# 
#         self.attr_editor = AttributeEditor()
#         right_layout.addWidget(self.attr_editor)
# 
#         splitter.addWidget(right)
#         splitter.setStretchFactor(0, 2)
#         splitter.setStretchFactor(1, 3)
# 
#         self.setCentralWidget(splitter)
# 
#         self._current_dataset_model: Optional[H5DatasetTableModel] = None
# 
#     # ---- File operations ----
# 
#     def open_file(self):
#         fn, _ = QFileDialog.getOpenFileName(
#             self, "Open HDF5 file", "", "HDF5 Files (*.h5 *.hdf5 *.hdf *.he5);;All Files (*)"
#         )
#         if not fn:
#             return
#         try:
#             self._open_h5(fn)
#         except Exception as e:
#             QMessageBox.critical(self, "Open failed", f"{e}\n\n{traceback.format_exc()}")
# 
#     def _open_h5(self, filename: str):
#         self.close_file()
# 
#         # Open read/write; if user has permission issues, they'll see it.
#         self._h5 = h5py.File(filename, "r+")
#         self._filename = filename
#         self.setWindowTitle(f"HDF5 Viewer/Editor - {filename}")
#         self.info.setText(f"Opened: {filename}")
#         self._populate_tree()
# 
#     def close_file(self):
#         self.tree.clear()
#         self.dataset_view.setModel(None)
#         self._current_dataset_model = None
#         self.attr_editor.set_target(None, None)
#         self.info.setText("Open an HDF5 file to begin.")
# 
#         if self._h5 is not None:
#             try:
#                 self._h5.flush()
#                 self._h5.close()
#             except Exception:
#                 pass
#         self._h5 = None
#         self._filename = None
#         self.setWindowTitle("HDF5 Viewer/Editor (PyQt6)")
# 
#     # ---- Tree population ----
# 
#     def _populate_tree(self):
#         self.tree.clear()
#         if self._h5 is None:
#             return
# 
#         # Root item
#         root_item = QTreeWidgetItem(["/", "Group", "", ""])
#         root_item.setData(0, Qt.ItemDataRole.UserRole, "/")
#         self.tree.addTopLevelItem(root_item)
# 
#         self._populate_children(root_item, self._h5["/"])
#         root_item.setExpanded(True)
# 
#         self.tree.resizeColumnToContents(0)
#         self.tree.resizeColumnToContents(1)
# 
#     def _populate_children(self, parent_item: QTreeWidgetItem, group: h5py.Group):
#         # Add children sorted by name for stable browsing
#         for name in sorted(group.keys()):
#             obj = group[name]
#             path = obj.name
# 
#             if isinstance(obj, h5py.Group):
#                 it = QTreeWidgetItem([path, "Group", "", ""])
#                 it.setData(0, Qt.ItemDataRole.UserRole, path)
#                 parent_item.addChild(it)
#                 # Recursively populate
#                 self._populate_children(it, obj)
# 
#             elif isinstance(obj, h5py.Dataset):
#                 shape = str(obj.shape)
#                 dtype = str(obj.dtype)
#                 it = QTreeWidgetItem([path, "Dataset", shape, dtype])
#                 it.setData(0, Qt.ItemDataRole.UserRole, path)
#                 parent_item.addChild(it)
#             else:
#                 it = QTreeWidgetItem([path, type(obj).__name__, "", ""])
#                 it.setData(0, Qt.ItemDataRole.UserRole, path)
#                 parent_item.addChild(it)
# 
#     # ---- Selection handling ----
# 
#     def _on_tree_selection(self):
#         if self._h5 is None:
#             return
#         items = self.tree.selectedItems()
#         if not items:
#             return
# 
#         item = items[0]
#         path = item.data(0, Qt.ItemDataRole.UserRole)
#         if not path:
#             return
# 
#         obj = self._h5[path]  # type: ignore[index]
# 
#         # Attributes
#         self.attr_editor.set_target(self._h5, path)
# 
#         # Dataset view
#         if isinstance(obj, h5py.Dataset):
#             ds = obj
#             self.info.setText(f"Selected dataset: {path} | shape={ds.shape} dtype={ds.dtype}")
# 
#             if ds.ndim in (0, 1, 2):
#                 dref = DatasetRef(self._h5, path)
#                 model = H5DatasetTableModel(dref)
#                 self._current_dataset_model = model
#                 self.dataset_view.setModel(model)
#                 self.dataset_view.resizeColumnsToContents()
#             else:
#                 self.dataset_view.setModel(None)
#                 self._current_dataset_model = None
#                 QMessageBox.information(
#                     self,
#                     "Dataset view limited",
#                     f"This dataset has ndim={ds.ndim}. The table editor supports only 0/1/2-D.\n\n"
#                     f"{path}\nshape={ds.shape}\ndtype={ds.dtype}"
#                 )
#         else:
#             self.info.setText(f"Selected: {path} ({'Group' if isinstance(obj, h5py.Group) else type(obj).__name__})")
#             self.dataset_view.setModel(None)
#             self._current_dataset_model = None
# 
#     def closeEvent(self, event):
#         self.close_file()
#         super().closeEvent(event)
# 
# 
# def main():
#     app = QApplication(sys.argv)
#     win = HDF5EditorMainWindow()
#     win.show()
#     sys.exit(app.exec())
# 
# 
# if __name__ == "__main__":
#     main()
