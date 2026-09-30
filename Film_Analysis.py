# -*- coding: utf-8 -*-
"""
Film_Analysis.py

Layout
------
Left   : Settings panel (calibration, background, output, dosemap settings)
Right   top   : Embedded Film Review panel (dose map + ROI tools) — this is
                where FilmReviewWindow used to pop up as a separate dialog.
                It now lives permanently in the layout and its content is
                swapped each time a new film is loaded.
Right bottom  : Film Queue (left) + Results Log (right), side by side, small.
"""

import os
from pathlib import Path

import numpy as np
import cv2
import pandas as pd

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.patches import Rectangle, Circle

from PyQt5.QtWidgets import (
    QWidget, QLabel, QPushButton, QListWidget,
    QFileDialog, QMessageBox, QHBoxLayout, QVBoxLayout, QFrame,
    QRadioButton, QSizePolicy, QDoubleSpinBox,
)

from utils import (
    BG_MAIN, BG_SURFACE, PRIMARY,TEXT_MUTED, WARNING, ERROR,
    heading_label, section_label, make_entry, make_log, log_write, hseparator,
)



# ═════════════════════════════════════════════════════════════════════════════
# Pure logic — no UI
# ═════════════════════════════════════════════════════════════════════════════

class FilmAnalyser:
    """Load, calibrate and convert a radiochromic film TIFF scan to a dose map."""

    @staticmethod
    def load_calibration(path: str):
        with open(path) as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            if line.strip().startswith("a ="):
                break
        a = float(lines[i    ].split("=")[1])
        b = float(lines[i + 1].split("=")[1])
        c = float(lines[i + 2].split("=")[1])
        return a, b, c

    @staticmethod
    def OD_to_dose(od, a, b, c):
        x = 10 ** (-od)
        return (c * x - a) / (b - x)

    @staticmethod
    def load_tiff(path: str) -> np.ndarray:
        data = np.frombuffer(Path(path).read_bytes(), dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_UNCHANGED)

    @classmethod
    def process(cls, image_path: str, bg_path: str, calib_abc: tuple) -> np.ndarray:
        a, b, c = calib_abc
        img = cls.load_tiff(image_path)[:, :, 2].astype(np.float32)
        bg  = cls.load_tiff(bg_path   )[:, :, 2].astype(np.float32)
        img = 65535 - img
        bg  = 65535 - bg
        h, w   = bg.shape
        bg_roi = bg[h // 2 - 50 : h // 2 + 50, w // 2 - 50 : w // 2 + 50]
        I0     = np.mean(bg_roi)
        img    = np.clip(img, 1e3, None)
        OD     = np.log10(I0 / img)
        OD     = np.clip(OD, -0.45, 0)
        return cls.OD_to_dose(OD, a, b, c)


# ═════════════════════════════════════════════════════════════════════════════
# Embedded Film Review panel (was FilmReviewWindow / QDialog — now a QWidget
# that lives permanently in FilmTab's layout and swaps content per film)
# ═════════════════════════════════════════════════════════════════════════════

class FilmReviewPanel(QWidget):
    """
    Embedded matplotlib figure for reviewing one film at a time.

    Usage from FilmTab:
        self.review_panel.load_film(dose, film_name, output_dir,
                                     doselim=..., roi_size_cm=...)
        # connect once:
        self.review_panel.saved.connect(self._on_film_saved)
    """

    PIXELS_PER_CM = 300 / 2.54

    from PyQt5.QtCore import pyqtSignal
    saved = pyqtSignal(bool, str, object)   # skipped, film_name, stats|None

    def __init__(self, parent=None):
        super().__init__(parent)

        self.dose        = None
        self.film_name    = None
        self.output_dir   = None
        self.doselim      = 15.0
        self.roi_size_px  = int(1.0 * self.PIXELS_PER_CM)

        self.center_x    = None
        self.center_y    = None
        self.roi_shape   = "Rectangle"
        self.roi_patch   = None
        self._last_stats = None
        self._has_film   = False

        self._build_ui()
        self._connect_events()
        self._show_placeholder()

    # ── UI ────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 8, 10, 8)
        root.setSpacing(6)

        # ── Top bar: title + dose limit spinbox + buttons ─────────────────────
        top = QHBoxLayout()

        self.title_lbl = QLabel("📽  No film loaded")
        self.title_lbl.setStyleSheet(
            f"color:{BG_SURFACE}; font-size:13pt; font-weight:bold;")
        top.addWidget(self.title_lbl)
        top.addStretch()

        top.addWidget(QLabel("Colour max (Gy):"))
        self.doselim_spin = QDoubleSpinBox()
        self.doselim_spin.setRange(0.1, 200.0)
        self.doselim_spin.setSingleStep(1.0)
        self.doselim_spin.setDecimals(1)
        self.doselim_spin.setValue(self.doselim)
        self.doselim_spin.setFixedWidth(80)
        self.doselim_spin.setToolTip(
            "Upper limit of the dose colour scale.\n"
            "Change this at any time — the map updates immediately."
        )
        self.doselim_spin.valueChanged.connect(self._update_doselim)
        top.addWidget(self.doselim_spin)

        self.skip_btn = QPushButton("Skip")
        self.skip_btn.setEnabled(False)
        self.skip_btn.clicked.connect(self._skip)
        top.addWidget(self.skip_btn)

        self.save_btn = QPushButton("💾  SAVE && NEXT")
        self.save_btn.setObjectName("success")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self._save_and_advance)
        top.addWidget(self.save_btn)

        root.addLayout(top)

        # ── ROI shape + profile y-limit ───────────────────────────────────────
        ctrl_row = QHBoxLayout()

        ctrl_row.addWidget(QLabel("ROI shape:"))
        self.roi_rect = QRadioButton("Rectangle")
        self.roi_circ = QRadioButton("Circle")
        self.roi_rect.setChecked(True)
        self.roi_rect.toggled.connect(self._shape_changed)
        ctrl_row.addWidget(self.roi_rect)
        ctrl_row.addWidget(self.roi_circ)
        ctrl_row.addSpacing(20)

        ctrl_row.addWidget(QLabel("Profile y-max (Gy):"))
        self.ylim_spin = QDoubleSpinBox()
        self.ylim_spin.setRange(0.1, 200.0)
        self.ylim_spin.setSingleStep(1.0)
        self.ylim_spin.setDecimals(1)
        self.ylim_spin.setValue(self.doselim)
        self.ylim_spin.setFixedWidth(80)
        self.ylim_spin.valueChanged.connect(self._update_doselim)
        ctrl_row.addWidget(self.ylim_spin)

        ctrl_row.addStretch()
        root.addLayout(ctrl_row)

        # ── Stats bar ─────────────────────────────────────────────────────────
        self.stats_lbl = QLabel("Load films and click Start Analysis")
        self.stats_lbl.setStyleSheet(
            f"background:{BG_MAIN}; color:{PRIMARY}; padding:4px 10px;")
        root.addWidget(self.stats_lbl)

        # ── Matplotlib figure ─────────────────────────────────────────────────
        self.fig = Figure(figsize=(11, 6.5))
        self.fig.patch.set_facecolor(BG_MAIN)
        gs = self.fig.add_gridspec(2, 2, width_ratios=[3, 1],
                                   hspace=0.35, wspace=0.25)
        self.ax_img   = self.fig.add_subplot(gs[:, 0])
        self.ax_profh = self.fig.add_subplot(gs[0, 1])
        self.ax_profv = self.fig.add_subplot(gs[1, 1])

        for ax in (self.ax_img, self.ax_profh, self.ax_profv):
            ax.set_facecolor("#ffffff")
            ax.tick_params(colors=PRIMARY, labelsize=8)
            for sp in ax.spines.values():
                sp.set_color(TEXT_MUTED)

        self.im = self.ax_img.imshow(
            np.zeros((10, 10)), cmap="rainbow", origin="upper",
            vmin=0, vmax=self.doselim,
        )
        self.cbar = self.fig.colorbar(self.im, ax=self.ax_img)
        self.cbar.set_label("Dose (Gy)", color=PRIMARY)
        self.cbar.ax.yaxis.set_tick_params(color=PRIMARY)
        plt.setp(self.cbar.ax.yaxis.get_ticklabels(), color=PRIMARY)

        self.cross_h = self.ax_img.axhline(0, color=TEXT_MUTED,   lw=0.8, zorder=10)
        self.cross_v = self.ax_img.axvline(0, color="#ff7043", lw=0.8, zorder=10)
        self.ax_img.set_title("Load a film to begin", color=PRIMARY, fontsize=10)

        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        root.addWidget(self.canvas)

    def _show_placeholder(self):
        self.skip_btn.setEnabled(False)
        self.save_btn.setEnabled(False)

    # ── matplotlib event connections ──────────────────────────────────────────
    def _connect_events(self):
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_move)
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)

    def _on_move(self, event):
        if not self._has_film or event.inaxes != self.ax_img:
            return
        self.cross_h.set_ydata([event.ydata, event.ydata])
        self.cross_v.set_xdata([event.xdata, event.xdata])
        self.fig.canvas.draw_idle()

    def _on_click(self, event):
        if not self._has_film or event.inaxes != self.ax_img:
            return
        self.center_x = int(event.xdata)
        self.center_y = int(event.ydata)
        self._redraw_roi()
        self._update_profiles()
        self._update_stats()
        self.save_btn.setEnabled(True)

    def _shape_changed(self):
        self.roi_shape = "Rectangle" if self.roi_rect.isChecked() else "Circle"
        if self._has_film and self.center_x is not None:
            self._redraw_roi()
            self._update_profiles()
            self._update_stats()

    # ── live dose-limit update ────────────────────────────────────────────────
    def _update_doselim(self, value: float):
        self.doselim = value
        self.im.set_clim(vmin=0, vmax=value)
        self.ylim_spin.blockSignals(True)
        self.ylim_spin.setValue(value)
        self.ylim_spin.blockSignals(False)
        for ax in (self.ax_profh, self.ax_profv):
            ax.set_ylim(0, value)
    
        self.fig.canvas.draw_idle()

    # ── drawing helpers ───────────────────────────────────────────────────────
    def _redraw_roi(self):
        if self.roi_patch:
            self.roi_patch.remove()
            self.roi_patch = None
        cx, cy, r = self.center_x, self.center_y, self.roi_size_px
        if self.roi_shape == "Rectangle":
            self.roi_patch = Rectangle(
                (cx - r // 2, cy - r // 2), r, r,
                edgecolor="#ffffff", facecolor="none", lw=2, zorder=10
            )
        else:
            self.roi_patch = Circle(
                (cx, cy), r / 2,
                edgecolor=TEXT_MUTED, facecolor="none", lw=2, zorder=10
            )
        self.ax_img.add_patch(self.roi_patch)
        self.ax_img.set_title(f"ROI centre: ({cx}, {cy})", color=PRIMARY, fontsize=9)
        self.fig.canvas.draw_idle()

    def _compute_roi_mask(self):
        yy, xx = np.ogrid[:self.dose.shape[0], :self.dose.shape[1]]
        cx, cy, r = self.center_x, self.center_y, self.roi_size_px
        if self.roi_shape == "Rectangle":
            return ((xx >= cx - r // 2) & (xx <= cx + r // 2) &
                    (yy >= cy - r // 2) & (yy <= cy + r // 2))
        return (xx - cx) ** 2 + (yy - cy) ** 2 <= (r / 2) ** 2

    def _update_profiles(self):
        ax1, ax2 = self.ax_profh, self.ax_profv
        ax1.clear(); ax2.clear()

        ppcm  = self.PIXELS_PER_CM
        x     = np.arange(self.dose.shape[1])
        y     = np.arange(self.dose.shape[0])
        ylim  = self.ylim_spin.value()

        ax1.plot((x - self.center_x) / ppcm, self.dose[self.center_y, :],
                 color="#6eaee6", lw=1)
        ax2.plot((y - self.center_y) / ppcm, self.dose[:, self.center_x],
                 color="#ff7043", lw=1)

        half = (self.roi_size_px / 2) / ppcm
        for ax in (ax1, ax2):
            ax.axvline(-half, color=PRIMARY, ls="--", alpha=0.5, lw=0.8)
            ax.axvline(+half, color=PRIMARY, ls="--", alpha=0.5, lw=0.8)
            ax.set_ylim(0, ylim)
            ax.set_facecolor("#ffffff")
            ax.tick_params(colors=PRIMARY, labelsize=7)
            for sp in ax.spines.values():
                sp.set_color(TEXT_MUTED)
            ax.set_ylabel("Dose (Gy)", color=PRIMARY, fontsize=8)
            ax.grid(True, alpha=0.2, color=TEXT_MUTED)

        ax1.set_title("Horizontal profile", color=PRIMARY, fontsize=8)
        ax2.set_title("Vertical profile",   color=PRIMARY, fontsize=8)
        ax2.set_xlabel("Distance from centre (cm)", color=PRIMARY, fontsize=8)
        self.fig.canvas.draw_idle()

    def _update_stats(self):
        mask   = self._compute_roi_mask()
        vals   = self.dose[mask]
        mean_d = np.mean(vals)
        min_d  = np.min(vals)
        max_d  = np.max(vals)
        self.stats_lbl.setText(
            f"Mean: {mean_d:.3f} Gy   Min: {min_d:.3f} Gy   Max: {max_d:.3f} Gy"
        )
        self._last_stats = dict(mean=mean_d, min=min_d, max=max_d)

    # ── Public API — called by FilmTab ──────────────────────────────────────

    def load_film(self, dose: np.ndarray, film_name: str, output_dir: str,
                  doselim: float = 15.0, roi_size_cm: float = 1.0):
        """Load a new film into the panel, resetting ROI state."""
        self.dose        = dose
        self.film_name    = film_name
        self.output_dir   = output_dir
        self.doselim      = doselim
        self.roi_size_px  = int(roi_size_cm * self.PIXELS_PER_CM)

        self.center_x    = None
        self.center_y    = None
        self._last_stats = None
        self._has_film   = True
        if self.roi_patch:
            self.roi_patch.remove()
            self.roi_patch = None

        self.title_lbl.setText(f"📽  {film_name}")
        self.doselim_spin.blockSignals(True)
        self.doselim_spin.setValue(doselim)
        self.doselim_spin.blockSignals(False)
        self.ylim_spin.blockSignals(True)
        self.ylim_spin.setValue(doselim)
        self.ylim_spin.blockSignals(False)

        self.im.set_data(dose)
        self.im.set_data(dose)
        self.im.set_extent((-0.5, dose.shape[1] - 0.5, dose.shape[0] - 0.5, -0.5))
        self.ax_img.set_xlim(-0.5, dose.shape[1] - 0.5)
        self.ax_img.set_ylim(dose.shape[0] - 0.5, -0.5)
        self.im.set_clim(vmin=0, vmax=doselim)
        self.ax_img.set_title("Click to place ROI", color=PRIMARY, fontsize=10)
        self.ax_profh.clear()
        self.ax_profv.clear()
        for ax in (self.ax_profh, self.ax_profv):
            ax.set_facecolor("#ffffff")
        self.fig.canvas.draw_idle()

        self.stats_lbl.setText("Click on the dose map to place ROI")
        self.skip_btn.setEnabled(True)
        self.save_btn.setEnabled(False)

    def clear(self):
        """Reset to the empty / no-film state."""
        self._has_film = False
        self.dose = None
        self.title_lbl.setText("📽  No film loaded")
        self.stats_lbl.setText("Load films and click Start Analysis")
        self.im.set_data(np.zeros((10, 10)))
        self.ax_img.set_title("Load a film to begin", color=PRIMARY, fontsize=10)
        self.ax_profh.clear()
        self.ax_profv.clear()
        for ax in (self.ax_profh, self.ax_profv):
            ax.set_facecolor("#ffffff")
        self.fig.canvas.draw_idle()
        self._show_placeholder()

    # ── save / skip — emit signal, FilmTab advances to next film ────────────
    def _save_and_advance(self):
        if self._last_stats is None or not self._has_film:
            return
        s = self._last_stats

        fig_path = os.path.join(self.output_dir, f"{self.film_name}_dosemap.png")
        self.fig.savefig(fig_path, dpi=150, bbox_inches="tight",
                         facecolor=self.fig.get_facecolor())

        txt_path = os.path.join(self.output_dir, f"{self.film_name}_ROI_stats.txt")
        with open(txt_path, "w") as f:
            f.write(f"Film        : {self.film_name}\n")
            f.write(f"ROI shape   : {self.roi_shape}\n")
            f.write(f"ROI centre  : ({self.center_x}, {self.center_y}) px\n")
            f.write(f"Mean dose   : {s['mean']:.4f} Gy\n")
            f.write(f"Min dose    : {s['min']:.4f} Gy\n")
            f.write(f"Max dose    : {s['max']:.4f} Gy\n")

        name = self.film_name
        self.saved.emit(False, name, s)

    def _skip(self):
        if not self._has_film:
            return
        name = self.film_name
        self.saved.emit(True, name, None)


# ═════════════════════════════════════════════════════════════════════════════
# Film Tab — the QWidget registered in the main QTabWidget
# ═════════════════════════════════════════════════════════════════════════════

class FilmTab(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self._calib_abc   = None
        self._bg_path     = None
        self._output_dir  = None
        self._film_queue  = []
        self._results     = []
        self._current_idx = 0
        self._build()

    # ── layout ────────────────────────────────────────────────────────────────
    def _build(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        # ── Left config panel ────────────────────────────────────────────────
        left = QFrame()
        left.setFixedWidth(350)
        left.setObjectName("settingsPanel")
        lv = QVBoxLayout(left)
        lv.setContentsMargins(12, 12, 12, 12)
        lv.setSpacing(4)

        lv.addWidget(heading_label("SETTINGS"))
        lv.addWidget(hseparator())

        # Calibration
        lv.addWidget(section_label("Calibration file (.txt)"))
        self._calib_lbl = QLabel("No calibration loaded")
        self._calib_lbl.setStyleSheet(f"color:{PRIMARY}; font-size:9pt;")
        self._calib_lbl.setWordWrap(True)
        lv.addWidget(self._calib_lbl)
        btn_cal = QPushButton("Browse…")
        btn_cal.setObjectName("accent")
        btn_cal.clicked.connect(self._load_calib)
        lv.addWidget(btn_cal)
        lv.addSpacing(25)

        # Background
        lv.addWidget(section_label("Background scan (.tif)"))
        self._bg_lbl = QLabel("No background loaded")
        self._bg_lbl.setStyleSheet(f"color:{PRIMARY}; font-size:9pt;")
        self._bg_lbl.setWordWrap(True)
        lv.addWidget(self._bg_lbl)
        btn_bg = QPushButton("Browse…")
        btn_bg.setObjectName("accent")
        btn_bg.clicked.connect(self._load_bg)
        lv.addWidget(btn_bg)
        lv.addSpacing(25)

        # Output folder
        lv.addWidget(section_label("Output folder"))
        self._outdir_lbl = QLabel("Same folder as each film")
        self._outdir_lbl.setStyleSheet(f"color:{PRIMARY}; font-size:9pt;")
        self._outdir_lbl.setWordWrap(True)
        lv.addWidget(self._outdir_lbl)
        btn_out = QPushButton("Browse…")
        btn_out.setObjectName("accent")
        btn_out.clicked.connect(self._pick_output)
        lv.addWidget(btn_out)

        lv.addSpacing(15)
        lv.addWidget(hseparator())
        lv.addSpacing(15)

        lv.addWidget(section_label("Dosemap settings"))

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Dose limit (Gy):"))
        self.doselim_entry = make_entry("10", width=100)
        self.doselim_entry.setToolTip(
            "Default colour-scale maximum passed to each review panel.\n"
            "You can adjust it per-film inside the panel."
        )
        row1.addWidget(self.doselim_entry)
        row1.addStretch()
        lv.addLayout(row1)
        lv.addSpacing(15)
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("ROI size (cm):"))
        self.roi_entry = make_entry("1.0", width=100)
        row2.addWidget(self.roi_entry)
        row2.addStretch()
        lv.addLayout(row2)

        lv.addStretch()
        root.addWidget(left)

        # ── Right column ─────────────────────────────────────────────────────
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.setSpacing(6)

        # Top: header row with batch action buttons
        hdr = QHBoxLayout()
        hdr.addWidget(heading_label("FILM REVIEW"))
        hdr.addStretch()

        btn_add = QPushButton("+ Add Films")
        btn_add.setObjectName("accent")
        btn_add.clicked.connect(self._add_films)
        btn_clr = QPushButton("Clear")
        btn_clr.clicked.connect(self._clear_queue)
        self.start_btn = QPushButton("▶  Start Analysis")
        self.start_btn.setObjectName("success")
        self.start_btn.clicked.connect(self._start_analysis)

        for b in (btn_add, btn_clr, self.start_btn):
            hdr.addWidget(b)
        rv.addLayout(hdr)

        # Top-right main area: embedded review panel (replaces the old dialog)
        self.review_panel = FilmReviewPanel()
        self.review_panel.saved.connect(self._on_film_saved)
        rv.addWidget(self.review_panel, stretch=1)

        # Bottom row: Film Queue (left, small) + Results Log (right, small)
        bottom_row = QHBoxLayout()
        bottom_row.setSpacing(10)
        
        log_box = QVBoxLayout()
        log_box.addWidget(heading_label("RESULTS LOG"))
        self.log = make_log(height=150)
        log_box.addWidget(self.log)
        bottom_row.addLayout(log_box, stretch=1)
        
        queue_box = QVBoxLayout()
        queue_box.addWidget(heading_label("FILM QUEUE"))
        self.film_list = QListWidget()
        self.film_list.setFixedHeight(150)
        queue_box.addWidget(self.film_list)
        bottom_row.addLayout(queue_box, stretch=1)

        rv.addLayout(bottom_row)

        root.addWidget(right, stretch=1)

    # ── file helpers ──────────────────────────────────────────────────────────
    def _load_calib(self):
        p, _ = QFileDialog.getOpenFileName(
            self, "Select calibration file", "",
            "Text files (*.txt);;All files (*.*)")
        if not p:
            return
        try:
            self._calib_abc = FilmAnalyser.load_calibration(p)
            self._calib_lbl.setText(Path(p).name)
            log_write(self.log, f"✓ Calibration loaded: {Path(p).name}")
        except Exception as e:
            QMessageBox.critical(self, "Calibration error", str(e))

    def _load_bg(self):
        p, _ = QFileDialog.getOpenFileName(
            self, "Select background scan", "",
            "TIFF files (*.tif *.tiff);;All files (*.*)")
        if not p:
            return
        self._bg_path = p
        self._bg_lbl.setText(Path(p).name)
        log_write(self.log, f"✓ Background loaded: {Path(p).name}")

    def _pick_output(self):
        d = QFileDialog.getExistingDirectory(self, "Select output folder")
        if d:
            self._output_dir = d
            self._outdir_lbl.setText(d)

    def _add_films(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select film scans", "",
            "TIFF files (*.tif *.tiff);;All files (*.*)")
        for p in paths:
            if p not in self._film_queue:
                self._film_queue.append(p)
                self.film_list.addItem(Path(p).name)

    def _clear_queue(self):
        self._film_queue.clear()
        self.film_list.clear()
        self.review_panel.clear()

    # ── analysis flow ─────────────────────────────────────────────────────────
    def _start_analysis(self):
        if not self._calib_abc:
            QMessageBox.warning(self, "Missing calibration",
                                "Please load a calibration file first.")
            return
        if not self._bg_path:
            QMessageBox.warning(self, "Missing background",
                                "Please load a background scan first.")
            return
        if not self._film_queue:
            QMessageBox.warning(self, "No films",
                                "Add at least one film to the queue.")
            return
        self._current_idx = 0
        self._results     = []
        self.start_btn.setEnabled(False)
        self._open_next_film()

    def _open_next_film(self):
        if self._current_idx >= len(self._film_queue):
            self._finish()
            return

        path   = self._film_queue[self._current_idx]
        name   = Path(path).stem
        outdir = self._output_dir or str(Path(path).parent)

        self.film_list.clearSelection()
        item = self.film_list.item(self._current_idx)
        if item:
            item.setSelected(True)
            self.film_list.setCurrentItem(item)

        try:
            doselim  = float(self.doselim_entry.text())
            roi_size = float(self.roi_entry.text())
            dose     = FilmAnalyser.process(path, self._bg_path, self._calib_abc)
        except Exception as e:
            log_write(self.log, f"✗ {name}: {e}")
            self._current_idx += 1
            self._open_next_film()
            return

        # Load directly into the embedded panel — no dialog, no exec()
        self.review_panel.load_film(
            dose, name, outdir, doselim=doselim, roi_size_cm=roi_size,
        )

    def _on_film_saved(self, skipped: bool, film_name: str, stats):
        if skipped:
            log_write(self.log, f"→ {film_name}: skipped")
        else:
            log_write(self.log,
                      f"✓ {film_name}: mean {stats['mean']:.3f} Gy  "
                      f"min {stats['min']:.3f} Gy  max {stats['max']:.3f} Gy")
            self._results.append({"film": film_name, **stats})

        self._current_idx += 1
        self._open_next_film()

    def _finish(self):
        self.start_btn.setEnabled(True)
        self.review_panel.clear()
        log_write(self.log, f"── Batch complete ({len(self._results)} saved) ──")

        if self._results:
            out_root = (self._output_dir
                        or str(Path(self._film_queue[0]).parent))
            csv_path = os.path.join(out_root, "film_batch_summary.csv")
            pd.DataFrame(self._results).to_csv(csv_path, index=False)
            log_write(self.log, f"📊 Summary saved: {Path(csv_path).name}")

        QMessageBox.information(
            self, "Done",
            f"Batch complete!\n{len(self._results)} films saved.",
        )