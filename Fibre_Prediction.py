"""

Prediction_Tab.py

FLASH Dose Predictor — QWidget tab, drop-in for the main QTabWidget.

Usage in main.py:
    from Prediction_Tab import PredictionTab
    tabs.addTab(PredictionTab(), "Dose Predictions")

Workflow
--------
1. Define calibration data in the CALIBRATION section, or load a .txt
   warmup file as before.
2. Select warmup .txt files (exported by the Fibre Analysis tab).
3. Set session parameters (target SSD, durations).
4. Click "Predict" → table + plot update live.
5. Save results to .txt.

Units
-----
  AUC       : V·ms
  Dose      : Gy
  Dose rate : Gy/s
  SSD       : cm
  Duration  : ms
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("QtAgg")

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from utils import (
    PRIMARY,
    heading_label,
    hseparator,
    log_write,
    make_log,
    section_label,
)

# ─────────────────────────────────────────────────────────────────────────────
# Change if different, taken from reproducibility measurements
# ─────────────────────────────────────────────────────────────────────────────
#Bottom tube (CONV 1.8mA)
CONV_FLASH_RATIO = 0.0019

#Two tubes
# CONV_FLASH_RATIO = 0.002


# ─────────────────────────────────────────────────────────────────────────────
# Physics helpers
# ─────────────────────────────────────────────────────────────────────────────

def parse_analysis_txt(filepath: str) -> dict:
    results: dict = {}
    pattern = re.compile(r"^(.+?)\s*:\s*([-\d.]+)\s*$")
    with open(filepath, "r", encoding="cp1252", errors="replace") as fh:
        for line in fh:
            m = pattern.match(line.strip())
            if m:
                try:
                    results[m.group(1).strip()] = float(m.group(2))
                except ValueError:
                    pass
    return results


def fit_calibration(cal: list[dict]):
    """Linear fit with free intercept. Returns (slope, intercept, r2)."""
    aucs  = np.array([c["AUC_Vms"]     for c in cal])
    doses = np.array([c["film_dose_Gy"] for c in cal])
    slope, intercept = np.polyfit(aucs, doses, 1)
    ss_res = np.sum((doses - (slope * aucs + intercept)) ** 2)
    ss_tot = np.sum((doses - doses.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return float(slope), float(intercept), float(r2)


def ssd_scale(dose: float, ref: float, tgt: float) -> float:
    return dose * (ref / tgt) ** 2


# ─────────────────────────────────────────────────────────────────────────────
# Calibration input table
# ─────────────────────────────────────────────────────────────────────────────

_CAL_DEFAULTS = [
    (50,  20.205, 3.128),
    (100, 50.084, 6.850),
    (200, 85.382, 12.201),
    (250, 107.142, 14.856),
]

_TABLE_STYLE = """
    QTableWidget {
        background: #ffffff;
        color: #000000;
        gridline-color: #000000;
        border: 1px solid #000000;
    }
    QHeaderView::section {
        background: #f0f0f0;
        color: #000000;
        padding: 4px;
        border: 1px solid #000000;
        font-size: 8pt;
        font-weight: bold;
    }
    QTableWidget::item:alternate { background: #f8f8f8; }
    QTableWidget::item:selected  { background: #cce5ff; color: #000000; }
"""


class CalibTable(QTableWidget):
    HEADERS = ["Duration (ms)", "AUC (V·ms)", "Film dose (Gy)"]  # noqa: RUF012

    def __init__(self, parent=None):
        super().__init__(0, 3, parent)
        self.setHorizontalHeaderLabels(self.HEADERS)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setAlternatingRowColors(True)
        self.setStyleSheet(_TABLE_STYLE)
        for row in _CAL_DEFAULTS:
            self._append(*row)

    def _append(self, dur=100, auc=0.0, dose=0.0):
        r = self.rowCount()
        self.insertRow(r)
        for col, val in enumerate((dur, auc, dose)):
            item = QTableWidgetItem(f"{val}")
            item.setTextAlignment(Qt.AlignCenter)
            self.setItem(r, col, item)

    def add_row(self):    self._append(100, 0.0, 0.0)

    def remove_selected(self):
        for r in sorted({i.row() for i in self.selectedItems()}, reverse=True):
            self.removeRow(r)

    def get_data(self) -> list[dict]:
        out = []
        for r in range(self.rowCount()):
            try:
                dur  = float(self.item(r, 0).text())
                auc  = float(self.item(r, 1).text())
                dose = float(self.item(r, 2).text())
                if auc > 0 and dose > 0:
                    out.append({"duration_ms": dur, "AUC_Vms": auc,
                                "film_dose_Gy": dose})
            except (AttributeError, ValueError):
                pass
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Results table
# ─────────────────────────────────────────────────────────────────────────────

class ResultsTable(QTableWidget):

    COL_DUR       = 0
    COL_DOSE_REF  = 1
    COL_CONV_REF  = 2
    COL_DOSE_TGT  = 3
    COL_CONV_TGT  = 4
    COL_DR_REF    = 5
    COL_DR_TGT    = 6
    COL_CONV_DR   = 7

    _BASE_HEADERS = [  # noqa: RUF012
        "Duration (ms)",
        "Dose @ REF (Gy)",
        "CONV time @ REF\n(s)",
        "Dose @ TGT (Gy)",
        "CONV time @ TGT\n(s)",
        "FLASH DR @ REF\n(Gy/s)",
        "FLASH DR @ TGT\n(Gy/s)",
        "CONV DR @ TGT\n(Gy/s)",
    ]

    def __init__(self, parent=None):
        super().__init__(0, 8, parent)   # ← 8 not 10
        self.setHorizontalHeaderLabels(self._BASE_HEADERS)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setAlternatingRowColors(True)
        self.setStyleSheet(_TABLE_STYLE)

    def populate(self, rows: list[dict], ref_ssd: float, tgt_ssd: float):
        hdrs = list(self._BASE_HEADERS)
        hdrs[self.COL_DOSE_REF] = f"Dose @ {ref_ssd:.1f} cm (Gy)"
        hdrs[self.COL_CONV_REF] = f"CONV time\n@ {ref_ssd:.1f} cm (s)"
        hdrs[self.COL_DOSE_TGT] = f"Dose @ {tgt_ssd:.1f} cm (Gy)"
        hdrs[self.COL_CONV_TGT] = f"CONV time\n@ {tgt_ssd:.1f} cm (s)"
        hdrs[self.COL_DR_REF]   = f"FLASH DR\n@ {ref_ssd:.1f} cm (Gy/s)"
        hdrs[self.COL_DR_TGT]   = f"FLASH DR\n@ {tgt_ssd:.1f} cm (Gy/s)"
        hdrs[self.COL_CONV_DR]  = f"CONV DR\n@ {tgt_ssd:.1f} cm (Gy/s)"
        self.setHorizontalHeaderLabels(hdrs)
        self.setRowCount(0)

        for r_idx, r in enumerate(rows):
            self.insertRow(r_idx)
            vals = [
                f"{r['duration_ms']}",
                f"{r['dose_ref']:.3f}",
                f"{r['conv_time_ref']:.1f}",
                f"{r['dose_tgt']:.3f}",
                f"{r['conv_time_tgt']:.1f}",
                f"{r['dr_ref']:.1f}",
                f"{r['dr_tgt']:.1f}",
                f"{r['conv_dr_tgt']:.4f}",
            ]
            for c_idx, val in enumerate(vals):
                item = QTableWidgetItem(val)
                item.setTextAlignment(Qt.AlignCenter)
                self.setItem(r_idx, c_idx, item)

# ─────────────────────────────────────────────────────────────────────────────
# Main tab
# ─────────────────────────────────────────────────────────────────────────────

class PredictionTab(QWidget):
    """Drop-in tab: tabs.addTab(PredictionTab(), "Dose Predictions")"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._warmup_paths: list[str] = []
        self._results: list[dict]     = []
        self._fig = None
        self._ax_pred = None
        self._init_widgets()
        self._init_layout()

    # ── Widget creation ───────────────────────────────────────────────────────

    def _init_widgets(self):
        self.left_frame = QFrame()
        self.left_frame.setFixedWidth(350)
        self.left_frame.setObjectName("settingsPanel")

        self.ref_ssd_spin = QDoubleSpinBox()
        self.ref_ssd_spin.setRange(1.0, 100.0)
        self.ref_ssd_spin.setSingleStep(0.1)
        self.ref_ssd_spin.setDecimals(1)
        self.ref_ssd_spin.setValue(7.0)

        self.tgt_ssd_spin = QDoubleSpinBox()
        self.tgt_ssd_spin.setRange(1.0, 100.0)
        self.tgt_ssd_spin.setSingleStep(0.1)
        self.tgt_ssd_spin.setDecimals(1)
        self.tgt_ssd_spin.setValue(7.5)

        self.dur_edit = QLineEdit("32, 40, 50, 63, 80, 100, 125, 160, 200, 250")

        # CONV ratio
        self.conv_ratio_spin = QDoubleSpinBox()
        self.conv_ratio_spin.setRange(0.0001, 1.0)
        self.conv_ratio_spin.setSingleStep(0.0001)
        self.conv_ratio_spin.setDecimals(4)
        self.conv_ratio_spin.setValue(CONV_FLASH_RATIO)
        self.conv_ratio_spin.setToolTip(
            "CONV dose rate = this ratio × FLASH dose rate.\n"
            "Default: 0.002")

        self.warmup_lbl = QLabel("No warmup files loaded")
        self.warmup_lbl.setStyleSheet(f"color:{PRIMARY}; font-size:9pt;")
        self.warmup_lbl.setWordWrap(True)

        self.btn_warmup = QPushButton("Browse warmup .txt files…")
        self.btn_warmup.setObjectName("accent")
        self.btn_warmup.clicked.connect(self._load_warmup)

        self.predict_btn = QPushButton("▶  Predict")
        self.predict_btn.setObjectName("success")
        self.predict_btn.clicked.connect(self._predict)

        self.save_btn = QPushButton("💾  Save results…")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self._save_results)

        self.log = make_log(height=160)

        self.cal_table = CalibTable()
        self.cal_table.setMinimumHeight(160)

        self.btn_add_row = QPushButton("+ Row")
        self.btn_add_row.clicked.connect(self.cal_table.add_row)
        self.btn_del_row = QPushButton("− Row")
        self.btn_del_row.clicked.connect(self.cal_table.remove_selected)

        self.stats_lbl = QLabel("—")
        self.stats_lbl.setStyleSheet("""background: #f5f5f5;color: #1a1a1a;padding: 8px;border: 1px solid #d0d0d0;font-size: 9pt;""")

        self.res_table = ResultsTable()
        self.res_table.setMinimumHeight(180)

        self._fig = Figure(figsize=(12, 4))  # Wider for two plots
        self._fig.patch.set_facecolor("white")
        
        # Create two subplots side by side
        self._ax_cal = self._fig.add_subplot(121)  # Left: calibration
        self._ax_pred = self._fig.add_subplot(122)  # Right: predictions
        
        # Style calibration plot
        self._ax_cal.set_facecolor("white")
        self._ax_cal.tick_params(colors="black", labelsize=8)
        for sp in self._ax_cal.spines.values():
            sp.set_color("black")
        
        # Style predictions plot
        self._ax_pred.set_facecolor("white")
        self._ax_pred.tick_params(colors="black", labelsize=8)
        for sp in self._ax_pred.spines.values():
            sp.set_color("black")
        
        self._canvas = FigureCanvas(self._fig)
        self._canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._canvas.setMaximumHeight(320)

    # ── Layout ────────────────────────────────────────────────────────────────

    def _init_layout(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)
    
        # ── LEFT PANEL ────────────────────────────────────────────────────
        lv = QVBoxLayout(self.left_frame)
        lv.setContentsMargins(12, 12, 12, 12)
        lv.setSpacing(8)
    
        lv.addWidget(heading_label("SESSION PARAMETERS"))
        lv.addWidget(hseparator())
        lv.addSpacing(20)
    
        lv.addWidget(section_label("Reference SSD (cm)"))
        lv.addWidget(self.ref_ssd_spin)
        lv.addSpacing(20)
        
        lv.addWidget(section_label("Target SSD (cm)"))
        lv.addWidget(self.tgt_ssd_spin)
        lv.addSpacing(20)
        
        lv.addWidget(section_label("Target durations (ms, comma-separated)"))
        lv.addWidget(self.dur_edit)
        lv.addSpacing(20)
    
        lv.addWidget(hseparator())
        lv.addSpacing(20)
        lv.addWidget(heading_label("CONV SETTINGS"))
        lv.addWidget(section_label("CONV / FLASH dose-rate ratio"))
        lv.addWidget(self.conv_ratio_spin)
        lv.addSpacing(20)
    
        lv.addWidget(hseparator())
        lv.addSpacing(20)
        lv.addWidget(heading_label("WARMUP FILES"))
        lv.addWidget(self.warmup_lbl)
        lv.addSpacing(20)
        
        lv.addWidget(hseparator())
        lv.addSpacing(20)
        lv.addWidget(heading_label("LOG"))
        lv.addWidget(self.log)
        lv.addStretch()
    
        root.addWidget(self.left_frame)
    
        # ── RIGHT PANEL ───────────────────────────────────────────────────
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.setSpacing(8)
        
        # ── TOP ROW: HEADER WITH BUTTONS ─────────────────────────────
        hdr = QHBoxLayout()
        hdr.addWidget(heading_label("CALIBRATION & PREDICTIONS"))
        hdr.addStretch()
        
        hdr.addWidget(self.btn_warmup)
        hdr.addWidget(self.predict_btn)
        hdr.addWidget(self.save_btn)
        rv.addLayout(hdr)
        
        # ── TOP: calibration table + plot ────────────────────────────
        top = QHBoxLayout()
        
        # LEFT: calibration table
        left_top = QVBoxLayout()
        cal_hdr = QHBoxLayout()
        cal_hdr.addWidget(heading_label("CALIBRATION DATA"))
        cal_hdr.addStretch()
        cal_hdr.addWidget(self.btn_add_row)
        cal_hdr.addWidget(self.btn_del_row)
        
        left_top.addLayout(cal_hdr)
        left_top.addWidget(self.cal_table)
        
        left_widget = QWidget()
        left_widget.setLayout(left_top)
        
        # RIGHT: single plot
        right_top = QVBoxLayout()
        right_top.addWidget(heading_label("CALIBRATION CURVE & REFERENCE"))
        right_top.addWidget(self._canvas)
        
        right_widget = QWidget()
        right_widget.setLayout(right_top)
        
        top.addWidget(left_widget, 1)
        top.addWidget(right_widget, 2)
        
        # ── FULL WIDTH STATS BAR ─────────────────────────────────────
        rv.addLayout(top, 3)
        rv.addWidget(self.stats_lbl)
        
        # ── PREDICTIONS ──────────────────────────────────────────────
        rv.addWidget(heading_label("PREDICTIONS"))
        rv.addWidget(self.res_table, 3)
        
        root.addWidget(right)

    # ── File loading ──────────────────────────────────────────────────────────

    def _load_warmup(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select warmup analysis .txt files", "",
            "Text files (*.txt);;All files (*.*)")
        if not paths:
            return
        self._warmup_paths = sorted(paths)
        names  = [Path(p).name for p in paths[:3]]
        suffix = "…" if len(paths) > 3 else ""
        self.warmup_lbl.setText(
            f"{len(paths)} file(s): {', '.join(names)}{suffix}")
        log_write(self.log, f"✓ Loaded {len(paths)} warmup file(s)")

    # ── Prediction ────────────────────────────────────────────────────────────

    def _predict(self):
        # 1. Calibration
        cal = self.cal_table.get_data()
        if len(cal) < 2:
            QMessageBox.warning(self, "Calibration",
                                "Enter at least 2 calibration rows with "
                                "non-zero AUC and dose.")
            return

        slope, intercept, r2 = fit_calibration(cal)
        ref_ssd    = self.ref_ssd_spin.value()
        tgt_ssd    = self.tgt_ssd_spin.value()
        isl        = (ref_ssd / tgt_ssd) ** 2
        conv_ratio = self.conv_ratio_spin.value()

        log_write(self.log,
                  f"Cal fit: slope={slope:.4f}  intercept={intercept:.4f}"
                  f"  R²={r2:.5f}")

        # 2. Warmup
        if not self._warmup_paths:
            QMessageBox.warning(self, "Warmup files",
                                "Load at least one warmup .txt file first.")
            return

        warmup_aucs, warmup_durs = [], []
        for fp in self._warmup_paths:
            parsed  = parse_analysis_txt(fp)
            auc_key = next((k for k in parsed if "area" in k.lower()), None)
            dur_key = next((k for k in parsed if "Duration Tube 1" in k), None)
            if auc_key is None:
                log_write(self.log, f"⚠ No AUC in {Path(fp).name} — skipped")
                continue
            warmup_aucs.append(parsed[auc_key])
            warmup_durs.append(parsed.get(dur_key, 100.0))

        if not warmup_aucs:
            QMessageBox.warning(self, "Warmup", "No valid AUC values found.")
            return

        w_aucs   = np.array(warmup_aucs)
        w_durs   = np.array(warmup_durs)
        mean_auc = w_aucs.mean()
        std_auc  = w_aucs.std(ddof=1) if len(w_aucs) > 1 else 0.0
        mean_dur = w_durs.mean()

        cal_auc_per_ms   = np.mean([c["AUC_Vms"] / c["duration_ms"] for c in cal])
        today_auc_per_ms = mean_auc / mean_dur
        scaling          = today_auc_per_ms / cal_auc_per_ms

        warmup_dose_ref = slope * mean_auc + intercept
        warmup_dose_tgt = ssd_scale(warmup_dose_ref, ref_ssd, tgt_ssd)
        dr_ref          = warmup_dose_ref / (mean_dur / 1000)
        dr_tgt          = warmup_dose_tgt / (mean_dur / 1000)
        conv_dr_tgt     = conv_ratio * dr_tgt
        conv_dr_ref     = conv_ratio * dr_ref

        log_write(self.log,
                  f"Warmup: mean AUC={mean_auc:.3f} ±{std_auc:.3f}  "
                  f"dur={mean_dur:.0f} ms  scaling={scaling:.4f}")
        log_write(self.log,
                  f"FLASH DR: {dr_ref:.1f} Gy/s @ {ref_ssd:.1f} cm  |  "
                  f"{dr_tgt:.1f} Gy/s @ {tgt_ssd:.1f} cm")
        log_write(self.log,
                  f"CONV DR: {conv_dr_ref:.4f} Gy/s @ {ref_ssd:.1f} cm  "
                  f"(ratio={conv_ratio})")

        self.stats_lbl.setText(
            f"slope={slope:.4f}   intercept={intercept:.4f}   R²={r2:.4f}   │   "
            f"scaling={scaling:.4f}   │   "
            f"FLASH DR @ {tgt_ssd:.1f} cm: {dr_tgt:.1f} Gy/s   │   "
            f"CONV DR @ {tgt_ssd:.1f} cm: {conv_dr_tgt:.4f} Gy/s"
        )

        # 3. Per-duration predictions
        try:
            durations = [int(x.strip())
                         for x in self.dur_edit.text().split(",")
                         if x.strip()]
        except ValueError:
            QMessageBox.warning(self, "Durations",
                                "Use comma-separated integers, e.g. 50, 100, 200")
            return

        self._results = []
        for dur in durations:
            pred_auc = today_auc_per_ms * dur
            dose_ref = slope * pred_auc + intercept
            dose_tgt = ssd_scale(dose_ref, ref_ssd, tgt_ssd)

            if std_auc > 0:
                hi      = slope * (today_auc_per_ms + std_auc / mean_dur) * dur + intercept
                lo      = slope * (today_auc_per_ms - std_auc / mean_dur) * dur + intercept
                sig_ref = (hi - lo) / 2.0
                sig_tgt = sig_ref * isl
            else:
                sig_ref = sig_tgt = 0.0

            flash_dr_ref  = dose_ref / (dur / 1000.0)
            flash_dr_tgt  = dose_tgt / (dur / 1000.0)
            conv_dr       = conv_ratio * flash_dr_tgt
            # Time in CONV to deliver the same dose as this FLASH pulse
            conv_time     = dose_tgt / conv_dr if conv_dr > 0 else float("inf")
            conv_time_ref = dose_ref / (conv_ratio * flash_dr_ref) if conv_ratio * flash_dr_ref > 0 else float("inf")

            self._results.append({
                "duration_ms":  dur,
                "pred_auc":     pred_auc,
                "dose_ref":     dose_ref,
                "sigma_ref":    sig_ref,
                "dose_tgt":     dose_tgt,
                "sigma_tgt":    sig_tgt,
                "dr_ref":       flash_dr_ref,
                "dr_tgt":       flash_dr_tgt,
                "conv_dr_tgt":  conv_dr,
                "conv_time_ref": conv_time_ref,
                "conv_time_tgt": conv_time,
            })

        self.res_table.populate(self._results, ref_ssd, tgt_ssd)
        self._draw_plots(cal, slope, intercept, r2,
                         ref_ssd, tgt_ssd, scaling, dr_ref, dr_tgt,
                         conv_ratio)
        self.save_btn.setEnabled(True)
        log_write(self.log,
                  f"✓ Predictions ready for {len(durations)} durations")

    # ── Plots ─────────────────────────────────────────────────────────────────

    def _draw_plots(self, cal, slope, intercept, r2,
                ref_ssd, tgt_ssd, scaling, dr_ref, dr_tgt,
                conv_ratio):

        # ── LEFT PLOT: Calibration curve only ─────────────────────────────
        ax_cal = self._ax_cal
        ax_cal.clear()
        ax_cal.set_facecolor("white")
    
        # Calibration data
        cal_aucs  = np.array([c["AUC_Vms"] for c in cal])
        cal_doses = np.array([c["film_dose_Gy"] for c in cal])
    
        ax_cal.scatter(cal_aucs, cal_doses,
                       color="black", s=40, label="Calibration data")
    
        # Calibration fit
        auc_fit  = np.linspace(0, cal_aucs.max() * 1.2, 200)
        dose_fit = slope * auc_fit + intercept
    
        ax_cal.plot(auc_fit, dose_fit,
                    color="#1f77b4", lw=2,
                    label=f"Fit (R²={r2:.4f})")
    
        ax_cal.set_xlabel("AUC (V·ms)", fontsize=9)
        ax_cal.set_ylabel(f"Dose @ {ref_ssd:.1f} cm (Gy)", fontsize=9)
        ax_cal.set_title("Calibration Curve", fontsize=10, fontweight="bold")
        ax_cal.grid(True, alpha=0.2, color="black")
        ax_cal.legend(fontsize=8)
        
        # ── RIGHT PLOT: Predictions only ──────────────────────────────────
        ax_pred = self._ax_pred
        ax_pred.clear()
        ax_pred.set_facecolor("white")
    
        # FLASH predictions
        if self._results:
            pred_aucs = np.array([r["pred_auc"] for r in self._results])
            pred_dose_ref = np.array([r["dose_ref"] for r in self._results])
        
            ax_pred.plot(pred_aucs, pred_dose_ref,
                        linestyle="--",
                        color="#2ca02c",
                        lw=2,
                        marker="o",
                        markersize=5,
                        label=f"FLASH @ {ref_ssd:.1f} cm")
    
        ax_pred.set_xlabel("AUC (V·ms)", fontsize=9)
        ax_pred.set_ylabel(f"Dose @ {ref_ssd:.1f} cm (Gy)", fontsize=9)
        ax_pred.set_title("Predictions", fontsize=10, fontweight="bold")
        ax_pred.grid(True, alpha=0.2, color="black")
        ax_pred.legend(fontsize=8)
        
        self._fig.tight_layout()
        self._canvas.draw()

    # ── Save ──────────────────────────────────────────────────────────────────

    def _save_results(self):
        if not self._results:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save predictions", "flash_dose_predictions.txt",
            "Text files (*.txt);;All files (*.*)")
        if not path:
            return

        cal = self.cal_table.get_data()
        slope, intercept, r2 = fit_calibration(cal)
        ref_ssd    = self.ref_ssd_spin.value()
        tgt_ssd    = self.tgt_ssd_spin.value()
        conv_ratio = self.conv_ratio_spin.value()

        with open(path, "w", encoding="utf-8") as fh:
            fh.write("FLASH Dose Predictions\n")
            fh.write("=" * 100 + "\n\n")
            fh.write(f"Calibration slope     : {slope:.5f}\n")
            fh.write(f"Calibration intercept : {intercept:.5f}\n")
            fh.write(f"R²                    : {r2:.5f}\n")
            fh.write(f"Reference SSD         : {ref_ssd:.1f} cm\n")
            fh.write(f"Target SSD            : {tgt_ssd:.1f} cm\n")
            fh.write(f"CONV/FLASH DR ratio   : {conv_ratio}\n\n")
            fh.write(
                f"{'Dur(ms)':>8}  {'Pred AUC':>10}  "
                f"{'Dose@ref':>10}  {'σ ref':>7}  "
                f"{'Dose@tgt':>10}  {'σ tgt':>7}  "
                f"{'FLASH DR@ref':>12}  {'FLASH DR@tgt':>12}  "
                f"{'CONV DR@tgt':>12}  {'CONV time(s)':>12}\n"
            )
            fh.write("-" * 115 + "\n")
            fh.writelines(f"{r['duration_ms']:>8}  {r['pred_auc']:>10.3f}  "
                    f"{r['dose_ref']:>10.3f}  {r['sigma_ref']:>7.3f}  "
                    f"{r['dose_tgt']:>10.3f}  {r['sigma_tgt']:>7.3f}  "
                    f"{r['dr_ref']:>12.1f}  {r['dr_tgt']:>12.1f}  "
                    f"{r['conv_dr_tgt']:>12.4f}  {r['conv_time_tgt']:>12.1f}\n" for r in self._results)

        log_write(self.log, f"✓ Saved: {Path(path).name}")