"""
DoseRate_Analysis.py

New tab for analysing predicted FLASH dose-rate vs SSD using a saved
dose-prediction .txt file.

Usage:
    from DoseRate_Analysis import DoseRateAnalysisTab
    tabs.addTab(DoseRateAnalysisTab(), "Dose-Rate Explorer")
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
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
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

# ---------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------

def _parse_reference_ssd(text: str) -> float | None:
    m = re.search(r"Reference SSD\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*cm", text, flags=re.IGNORECASE)
    if not m:
        return None
    return float(m.group(1))


def _parse_prediction_file(path: str) -> tuple[float, list[dict]]:
    """
    Parse the saved dose-prediction .txt file.
    Returns:
        ref_ssd_cm, rows = [{"duration_ms": ..., "dose_ref_Gy": ..., "flash_dr_Gy_s": ...}, ...]
    """
    rows: list[dict] = []
    ref_ssd = None

    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        text = fh.read()

    ref_ssd = _parse_reference_ssd(text)
    if ref_ssd is None:
        raise ValueError("Could not find a reference SSD in the file.")

    # Parse rows with pattern:
    # Dur(ms)  Pred AUC   Dose   σ   FLASH DR   CONV DR   CONV time
    lines = text.splitlines()
    for line in lines:
        s = line.strip()
        if not s:
            continue
        if s.startswith(("Dur(ms)", "FLASH Dose Predictions")):
            continue
        if re.match(r"^[-=]+$", s):
            continue

        parts = s.split()
        if len(parts) < 7:
            continue

        try:
            duration_ms = float(parts[0])
            dose_gy = float(parts[2])
            flash_dr_gy_s = float(parts[4])
        except ValueError:
            continue

        rows.append(
            {
                "duration_ms": duration_ms,
                "dose_ref_Gy": dose_gy,
                "flash_dr_Gy_s": flash_dr_gy_s,
            }
        )

    if not rows:
        raise ValueError("No usable prediction rows found in the selected file.")

    return ref_ssd, rows


# ---------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------

class CandidateTable(QTableWidget):
    def __init__(self, parent=None):
        super().__init__(0, 6, parent)
        self.setHorizontalHeaderLabels(
            ["Δ target (Gy)", "Dose (Gy)", "Duration (ms)", "Dose rate (Gy/s)", "SSD (cm)", "Extenders (cm)"]
        )
        self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setAlternatingRowColors(True)
        self.setStyleSheet(
            """
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
        )

    def populate(self, rows: list[dict], ref_ssd: float):
        self.setRowCount(0)

        if not rows:
            return

        max_dr = max(r["dose_rate_Gy_s"] for r in rows)
        min_delta = min(r["delta_target_Gy"] for r in rows)

        for idx, r in enumerate(rows):
            self.insertRow(idx)
            extenders = r["ssd_cm"] - ref_ssd

            values = [
                f"{r['delta_target_Gy']:.3f}",
                f"{r['dose_Gy']:.3f}",
                f"{r['duration_ms']:.0f}",
                f"{r['dose_rate_Gy_s']:.2f}",
                f"{r['ssd_cm']:.2f}",
                f"{extenders:.2f}",
            ]

            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setTextAlignment(Qt.AlignCenter)

                # Highlight every max dose-rate cell in column 3
                if col == 3 and np.isclose(r["dose_rate_Gy_s"], max_dr, rtol=1e-9, atol=1e-9):
                    item.setBackground(QColor("#ecc797"))

                # Highlight every minimum delta cell in column 0
                elif col == 0 and np.isclose(r["delta_target_Gy"], min_delta, rtol=1e-9, atol=1e-9):
                    item.setBackground(QColor("#FFFF99"))

                self.setItem(idx, col, item)

# ---------------------------------------------------------------------
# Main tab
# ---------------------------------------------------------------------

class DoseRateAnalysisTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._ref_ssd = 7.0
        self._dose_rows: list[dict] = []
        self._candidate_rows: list[dict] = []
        self._fig = None
        self._ax = None
        self._ax_candidates = None
        self._init_widgets()
        self._init_layout()

    def _init_widgets(self):
        self.left_frame = QFrame()
        self.left_frame.setFixedWidth(360)
        self.left_frame.setObjectName("settingsPanel")

        self.file_label = QLabel("No file loaded")
        self.file_label.setStyleSheet(f"color:{PRIMARY}; font-size:9pt;")
        self.file_label.setWordWrap(True)

        self.file_btn = QPushButton("Load predictions .txt")
        self.file_btn.setObjectName("accent")
        self.file_btn.clicked.connect(self._load_file)

        self.target_dose_spin = QDoubleSpinBox()
        self.target_dose_spin.setRange(0.0, 100.0)
        self.target_dose_spin.setSingleStep(0.1)
        self.target_dose_spin.setDecimals(2)
        self.target_dose_spin.setValue(10.0)

        self.tolerance_spin = QDoubleSpinBox()
        self.tolerance_spin.setRange(0.1, 10.0)
        self.tolerance_spin.setSingleStep(0.1)
        self.tolerance_spin.setDecimals(2)
        self.tolerance_spin.setValue(1.5)

        self.analyse_btn = QPushButton("▶ Analyse")
        self.analyse_btn.clicked.connect(self._analyse)

        self.save_btn = QPushButton("💾 Save results…")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self._save_results)

        self.log = make_log(height=180)

        self.candidate_table = CandidateTable()
        self.candidate_table.setMinimumHeight(220)

        self._fig = Figure(figsize=(11, 5))
        self._fig.patch.set_facecolor("white")

        self._ax = self._fig.add_subplot(121)
        self._ax_candidates = self._fig.add_subplot(122)

        self._ax.set_facecolor("white")
        self._ax.tick_params(colors="black", labelsize=8)
        for sp in self._ax.spines.values():
            sp.set_color("black")

        self._ax_candidates.set_facecolor("white")
        self._ax_candidates.tick_params(colors="black", labelsize=8)
        for sp in self._ax_candidates.spines.values():
            sp.set_color("black")

        self._canvas = FigureCanvas(self._fig)
        self._canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._canvas.setMaximumHeight(350)

    def _init_layout(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        lv = QVBoxLayout(self.left_frame)
        lv.setContentsMargins(12, 12, 12, 12)
        lv.setSpacing(8)

        lv.addWidget(heading_label("INPUT"))
        lv.addWidget(hseparator())
        lv.addSpacing(15)

        lv.addWidget(section_label("Dose prediction file"))
        lv.addWidget(self.file_btn)
        lv.addWidget(self.file_label)
        lv.addSpacing(15)

        lv.addWidget(section_label("Target dose (Gy)"))
        lv.addWidget(self.target_dose_spin)
        lv.addSpacing(15)

        lv.addWidget(section_label("Allowed difference from target (Gy)"))
        lv.addWidget(self.tolerance_spin)
        lv.addSpacing(20)

        lv.addWidget(hseparator())
        lv.addSpacing(15)
        lv.addWidget(heading_label("LOG"))
        lv.addWidget(self.log)
        lv.addStretch()

        root.addWidget(self.left_frame)

        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.setSpacing(8)

        hdr = QHBoxLayout()
        hdr.addWidget(heading_label("DOSE-RATE EXPLORER"))
        hdr.addStretch()
        hdr.addWidget(self.analyse_btn)
        hdr.addWidget(self.save_btn)
        rv.addLayout(hdr)

        # Plot area
        rv.addWidget(self._canvas)

        # Candidate table
        rv.addWidget(heading_label("CANDIDATE EXPOSURE TIMES / SSDs"))
        rv.addWidget(self.candidate_table)

        root.addWidget(right)

    def _load_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select saved predictions file",
            "",
            "Text files (*.txt);;All files (*.*)",
        )
        if not path:
            return

        try:
            ref_ssd, rows = _parse_prediction_file(path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "File error", f"Could not read the file:\n{exc}")
            return

        self._ref_ssd = ref_ssd
        self._dose_rows = rows
        self.file_label.setText(f"Loaded: {Path(path).name} (reference SSD = {ref_ssd:.1f} cm)")
        log_write(self.log, f"✓ Loaded {Path(path).name}")
        log_write(self.log, f"Reference SSD detected: {ref_ssd:.1f} cm")
        log_write(self.log, f"Found {len(rows)} dose rows in the file")

    def _get_ssds(self) -> list[float]:
        """Generate SSDs as ref_ssd + offsets"""
        offsets = [0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
        return [self._ref_ssd + offset for offset in offsets]

    def _analyse(self):
        if not self._dose_rows:
            QMessageBox.warning(self, "No file loaded", "Load a predictions .txt file first.")
            return

        ssds = self._get_ssds()
        target = self.target_dose_spin.value()
        tol = self.tolerance_spin.value()

        candidate_rows = []
        for ssd in ssds:
            for row in self._dose_rows:
                duration_ms = row["duration_ms"]
                dose_ref = row["dose_ref_Gy"]

                # Inverse square scaling from the reference SSD to the analysis SSD
                dose_at_ssd = dose_ref * (self._ref_ssd / ssd) ** 2

                # Dose rate in Gy/s for the chosen duration
                dose_rate = dose_at_ssd / (duration_ms / 1000.0)

                delta = abs(dose_at_ssd - target)
                if delta <= tol:
                    candidate_rows.append(
                        {
                            "duration_ms": duration_ms,
                            "ssd_cm": ssd,
                            "dose_Gy": dose_at_ssd,
                            "dose_rate_Gy_s": dose_rate,
                            "delta_target_Gy": delta,
                        }
                    )

        if not candidate_rows:
            QMessageBox.information(
                self,
                "No candidates",
                f"No exposure combinations are within ±{tol:.2f} Gy of the target dose {target:.2f} Gy.",
            )
            self._candidate_rows = []
            self.candidate_table.populate([], self._ref_ssd)
            self._draw_plots([], target, tol)
            self.save_btn.setEnabled(False)
            return

        # Sort by dose closeness to target, then time, then SSD
        candidate_rows.sort(key=lambda r: (r["delta_target_Gy"], r["duration_ms"], r["ssd_cm"]))
        self._candidate_rows = candidate_rows

        self.candidate_table.populate(candidate_rows, self._ref_ssd)
        self._draw_plots(candidate_rows, target, tol)
        self.save_btn.setEnabled(True)
        log_write(self.log, f"✓ Found {len(candidate_rows)} matching combinations")
        log_write(self.log, f"Target dose = {target:.2f} Gy, tolerance = ±{tol:.2f} Gy")

    def _draw_plots(self, candidate_rows: list[dict], target: float, tol: float):
        self._ax.clear()
        self._ax_candidates.clear()

        # Left plot: dose as function of duration for each SSD
        self._ax.set_title("Dose vs exposure time", fontsize=10, fontweight="bold")
        self._ax.set_xlabel("Exposure time (ms)", fontsize=9)
        self._ax.set_ylabel("Dose (Gy)", fontsize=9)

        ssds = sorted({r["ssd_cm"] for r in candidate_rows})
        if ssds:
            for ssd in ssds:
                rows = [r for r in candidate_rows if r["ssd_cm"] == ssd]
                times = np.array([r["duration_ms"] for r in rows])
                doses = np.array([r["dose_Gy"] for r in rows])
                self._ax.plot(times, doses, marker="o", linewidth=2, label=f"SSD {ssd:.2f} cm")
            self._ax.legend(fontsize=8)
        else:
            self._ax.text(0.5, 0.5, "No candidates", ha="center", va="center", transform=self._ax.transAxes)

        # Target band
        self._ax.axhline(target, color="red", linestyle="--", linewidth=1.5, label="Target")
        self._ax.axhspan(target - tol, target + tol, color="gray", alpha=0.15)
        self._ax.grid(True, alpha=0.25)

        # Right plot: dose rate vs SSD for the candidate combinations
        self._ax_candidates.set_title("Dose-rate vs SSD", fontsize=10, fontweight="bold")
        self._ax_candidates.set_xlabel("SSD (cm)", fontsize=9)
        self._ax_candidates.set_ylabel("Dose rate (Gy/s)", fontsize=9)

        if candidate_rows:
            ssd_values = np.array(sorted({r["ssd_cm"] for r in candidate_rows}))
            # For each SSD, show the best candidate dose-rate at that SSD
            # (closest to target)
            best_by_ssd = []
            for ssd in ssd_values:
                rows = [r for r in candidate_rows if r["ssd_cm"] == ssd]
                best = min(rows, key=lambda r: r["delta_target_Gy"])
                best_by_ssd.append((ssd, best["dose_rate_Gy_s"], best["duration_ms"]))

            ssd_plot = np.array([x[0] for x in best_by_ssd])
            dr_plot = np.array([x[1] for x in best_by_ssd])
            self._ax_candidates.plot(ssd_plot, dr_plot, marker="o", color="#2ca02c", linewidth=2)
            for ssd, dr, duration in best_by_ssd:
                self._ax_candidates.annotate(
                    f"{duration:.0f} ms",
                    xy=(ssd, dr),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=8,
                    color="black",
                )

        else:
            self._ax_candidates.text(0.5, 0.5, "No candidates", ha="center", va="center", transform=self._ax_candidates.transAxes)

        self._ax_candidates.grid(True, alpha=0.25)

        self._fig.tight_layout()
        self._canvas.draw()

    def _save_results(self):
        if not self._candidate_rows:
            QMessageBox.warning(self, "No results", "Run analysis first.")
            return

        # Save plots
        plot_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save plots as PNG",
            "dose_rate_analysis.png",
            "PNG files (*.png);;All files (*.*)"
        )
        if not plot_path:
            return

        try:
            self._fig.savefig(plot_path, dpi=300, bbox_inches="tight")
            log_write(self.log, f"✓ Plots saved: {Path(plot_path).name}")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Save error", f"Could not save plots:\n{exc}")
            return

        # Save table
        table_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save table as TXT",
            "dose_rate_candidates.txt",
            "Text files (*.txt);;All files (*.*)"
        )
        if not table_path:
            return

        try:
            with open(table_path, "w", encoding="utf-8") as fh:
                fh.write("FLASH Dose-Rate Analysis Results\n")
                fh.write("=" * 90 + "\n\n")
                fh.write(f"Reference SSD         : {self._ref_ssd:.2f} cm\n")
                fh.write(f"Target dose           : {self.target_dose_spin.value():.2f} Gy\n")
                fh.write(f"Tolerance             : ±{self.tolerance_spin.value():.2f} Gy\n\n")
                
                fh.write(
                    f"{'Δ target(Gy)':>12}  "
                    f"{'Dose(Gy)':>10}  "
                    f"{'Duration(ms)':>12}  "
                    f"{'Dose rate(Gy/s)':>15}  "
                    f"{'SSD(cm)':>10}  "
                    f"{'Extenders(cm)':>13}\n"
                )
                fh.write("-" * 90 + "\n")
                
                for r in self._candidate_rows:
                    extenders = r['ssd_cm'] - self._ref_ssd
                    fh.write(
                        f"{r['delta_target_Gy']:>12.3f}  "
                        f"{r['dose_Gy']:>10.3f}  "
                        f"{r['duration_ms']:>12.0f}  "
                        f"{r['dose_rate_Gy_s']:>15.2f}  "
                        f"{r['ssd_cm']:>10.2f}  "
                        f"{extenders:>13.2f}\n"
                    )

            log_write(self.log, f"✓ Table saved: {Path(table_path).name}")
            QMessageBox.information(self, "Saved", "Results saved successfully!")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Save error", f"Could not save table:\n{exc}")