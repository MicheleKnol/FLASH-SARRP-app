"""
Film_Calibration.py

Calibration tab for radiochromic film analysis.
Loads film TIFF scans at known doses, allows interactive ROI selection,
and fits an OD (Optical Density) model to generate calibration parameters.
"""

import os
import re
from pathlib import Path

import cv2
import matplotlib
import numpy as np
from PyQt5.QtCore import QObject, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from scipy.optimize import curve_fit

matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from utils import (
    BG_MAIN,
    PRIMARY,
    heading_label,
    hseparator,
    log_write,
    make_log,
    section_label,
)

# ════════════════════════════════════════════════════════════════════[...]
# OD Model
# ════════════════════════════════════════════════════════════════════[...]

def od_model(dose, a, b, c):
    """Optical density model: OD = -log10((a + b*dose) / (c + dose))"""
    x = (a + b * dose) / (c + dose)
    x = np.clip(x, 1e-8, None)
    return -np.log10(x)


# ════════════════════════════════════════════════════════════════════[...]
# Interactive ROI Selection Dialog
# ════════════════════════════════════════════════════════════════════[...]

class ROISelectionDialog(QDialog):
    """Interactive dialog for approving/adjusting ROI for each film."""
    
    def __init__(self, dose, image_array, center_x, center_y, crop_size, mean_val, parent=None):
        super().__init__(parent)
        self.dose = dose
        self.image = image_array
        self.center_x = center_x
        self.center_y = center_y
        self.crop_size = crop_size
        self.mean_val = mean_val
        self.approved = False
        self.approved_dose = dose
        self.new_center_x = center_x
        self.new_center_y = center_y
        
        self.setWindowTitle(f"ROI Selection - {dose:.3f} Gy")
        self.setGeometry(100, 100, 1200, 800)
        self._build_ui()
        self._update_display()
    
    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Instructions
        instr = QLabel("Click on the film to center the ROI. Adjust dose if needed, then click Approve.")
        instr.setStyleSheet(f"color:{PRIMARY}; font-size:10pt; font-weight:bold;")
        layout.addWidget(instr)
        
        # Image display (clickable)
        self.image_label = QLabel()
        self.image_label.setMinimumSize(800, 600)
        self.image_label.setMaximumSize(1000, 700)
        self.image_label.setStyleSheet("background: black; border: 2px solid gray;")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.mousePressEvent = self._on_image_click
        layout.addWidget(self.image_label, stretch=1)
        
        # Stats row
        stats_layout = QHBoxLayout()
        stats_layout.addWidget(QLabel(f"Dose: {self.dose:.3f} Gy"))
        stats_layout.addWidget(QLabel(f"Mean greyscale: {self.mean_val:.2f}"))
        stats_layout.addStretch()
        layout.addLayout(stats_layout)
        
        # Controls for dose adjustment only
        ctrl_layout = QHBoxLayout()
        ctrl_layout.addWidget(QLabel("Approved Dose (Gy):"))
        self.dose_spin = QDoubleSpinBox()
        self.dose_spin.setRange(0.0, 100.0)
        self.dose_spin.setSingleStep(0.1)
        self.dose_spin.setDecimals(3)
        self.dose_spin.setValue(self.dose)
        self.dose_spin.setMaximumWidth(100)
        ctrl_layout.addWidget(self.dose_spin)
        ctrl_layout.addStretch()
        layout.addLayout(ctrl_layout)
        
        # Action buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        
        btn_approve = QPushButton("✓ Approve & Continue")
        btn_approve.setObjectName("success")
        btn_approve.clicked.connect(self._approve)
        btn_layout.addWidget(btn_approve)
        
        layout.addLayout(btn_layout)
    
    def _on_image_click(self, event):
        """Handle mouse click on image to set ROI center."""
        # Get the pixmap displayed in the label
        pixmap = self.image_label.pixmap()
        if pixmap is None:
            return
        
        # Get click position relative to the label
        label_x = event.pos().x()
        label_y = event.pos().y()
        
        # Scale from label coordinates to image coordinates
        label_w = self.image_label.width()
        label_h = self.image_label.height()
        pixmap_w = pixmap.width()
        pixmap_h = pixmap.height()
        
        # Calculate scaling factor
        scale = min(label_w / pixmap_w, label_h / pixmap_h)
        
        # Calculate offsets (for centered image)
        offset_x = (label_w - pixmap_w * scale) / 2
        offset_y = (label_h - pixmap_h * scale) / 2
        
        # Convert click position to image coordinates
        if label_x >= offset_x and label_y >= offset_y:
            img_x = int((label_x - offset_x) / scale)
            img_y = int((label_y - offset_y) / scale)
            
            # Clamp to image bounds
            img_x = max(0, min(img_x, self.image.shape[1] - 1))
            img_y = max(0, min(img_y, self.image.shape[0] - 1))
            
            self.new_center_x = img_x
            self.new_center_y = img_y
            self._update_display()
    
    def _update_display(self):
        """Show full image with crop box overlay."""
        # Normalize image for display
        img_display = cv2.normalize(self.image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        img_display = cv2.cvtColor(img_display, cv2.COLOR_GRAY2BGR)
        
        # Draw crop rectangle
        x1 = max(0, self.new_center_x - self.crop_size)
        y1 = max(0, self.new_center_y - self.crop_size)
        x2 = min(self.image.shape[1] - 1, self.new_center_x + self.crop_size)
        y2 = min(self.image.shape[0] - 1, self.new_center_y + self.crop_size)
        
        cv2.rectangle(img_display, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        # Draw center point
        cv2.circle(img_display, (self.new_center_x, self.new_center_y), 5, (0, 255, 0), -1)
        
        # Convert to QPixmap
        h, w, ch = img_display.shape
        bytes_per_line = ch * w
        qt_image = QImage(img_display.data, w, h, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(qt_image)
        
        # Scale to fit label (maintaining aspect ratio)
        self.image_label.width()
        label_h = self.image_label.height()
        scaled = pixmap.scaledToHeight(label_h, Qt.SmoothTransformation) if label_h > 0 else pixmap
        self.image_label.setPixmap(scaled)
    
    def _approve(self):
        """User approved this ROI."""
        self.approved = True
        self.approved_dose = self.dose_spin.value()
        self.accept()
    
    def resizeEvent(self, event):
        """Redraw when dialog is resized."""
        super().resizeEvent(event)
        self._update_display()


# ════════════════════════════════════════════════════════════════════[...]
# Calibration worker (simplified)
# ════════════════════════════════════════════════════════════════════[...]

class _CalibrationWorker(QObject):
    """Background worker for calibration processing."""
    progress = pyqtSignal(int)
    log_message = pyqtSignal(str)
    roi_needed = pyqtSignal(dict)  # Signal to request ROI approval from GUI thread
    calibration_done = pyqtSignal(dict)
    finished = pyqtSignal()

    def __init__(self, dose_to_file, crop_size, output_dir):
        super().__init__()
        self.dose_to_file = dose_to_file
        self.crop_size = crop_size
        self.output_dir = output_dir
        self.calib_meas = {}
        self.roi_results = {}  # Store ROI results from GUI

    def process_roi_result(self, dose, greyscale_value, approved_dose):
        """Store the approved ROI result."""
        self.roi_results[dose] = (approved_dose, greyscale_value)

    def run(self):
        """Process each film and request ROI approval from GUI thread."""
        total = len(self.dose_to_file)
        
        for idx, (dose, path) in enumerate(sorted(self.dose_to_file.items())):
            try:
                self.log_message.emit(f"Processing {Path(path).name}...")
                
                # Load image
                img_bytes = Path(path).read_bytes()
                img_array = np.frombuffer(img_bytes, dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
                
                # Extract red channel and invert
                red = img[:, :, 2].astype(np.float32)
                red_inverted = 65535 - red
                
                # Get center and extract crop
                center_y, center_x = red.shape[0] // 2, red.shape[1] // 2
                crop = red_inverted[
                    center_y - self.crop_size:center_y + self.crop_size,
                    center_x - self.crop_size:center_x + self.crop_size
                ]
                mean_val = np.mean(crop)
                
                # Request ROI approval from GUI thread
                self.roi_needed.emit({
                    "dose": dose,
                    "image": red_inverted,
                    "center_x": center_x,
                    "center_y": center_y,
                    "mean_val": mean_val,
                })
                
                # Wait for result (blocking call)
                while dose not in self.roi_results:
                    QThread.msleep(100)
                
                approved_dose, _ = self.roi_results[dose]
                self.calib_meas[approved_dose] = mean_val
                self.log_message.emit(f"✓ {approved_dose:.3f} Gy: greyscale = {mean_val:.2f}")
                
                self.progress.emit(int((idx + 1) / total * 100))
                
            except Exception as e:  # noqa: BLE001
                self.log_message.emit(f"✗ Error processing {dose} Gy: {e}")
                import traceback
                traceback.print_exc()

        # Fit OD model
        if len(self.calib_meas) < 2:
            self.log_message.emit("❌ Need at least 2 calibration points.")
            self.finished.emit()
            return

        try:
            doses = np.array(sorted(self.calib_meas.keys()))
            greyscale_values = np.array([self.calib_meas[d] for d in doses])
            I0 = greyscale_values[0]
            optical_density = np.log10(I0 / greyscale_values)

            # Fit model
            p0 = [1, 10, 1]
            popt, _ = curve_fit(od_model, doses, optical_density, p0=p0)

            # Calculate R²
            residuals = optical_density - od_model(doses, *popt)
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((optical_density - np.mean(optical_density))**2)
            r_squared = 1 - (ss_res / ss_tot)

            self.log_message.emit(f"✅ OD Model fitted: R² = {r_squared:.4f}")
            self.log_message.emit(f"   Parameters: a={popt[0]:.4f}, b={popt[1]:.4f}, c={popt[2]:.4f}")

            # Save calibration summary
            summary_path = os.path.join(self.output_dir, "calibration_summary.txt")
            with open(summary_path, "w") as f:
                f.write("Dose (Gy)\tGreyvalue\tOptical Density\n")
                for d in sorted(self.calib_meas):
                    gv = self.calib_meas[d]
                    od = np.log10(I0 / gv)
                    f.write(f"{d:.3f}\t{gv:.4f}\t{od:.6f}\n")
                
                f.write("\n\nOD Model Fit Parameters:\n")
                f.write(f"a = {popt[0]:.4f}\n")
                f.write(f"b = {popt[1]:.4f}\n")
                f.write(f"c = {popt[2]:.4f}\n")
                f.write(f"R² = {r_squared:.4f}\n")
            
            self.log_message.emit(f"📄 Saved: {Path(summary_path).name}")

            # Emit results
            self.calibration_done.emit({
                "a": popt[0],
                "b": popt[1],
                "c": popt[2],
                "r_squared": r_squared,
                "doses": doses,
                "greyscale_values": greyscale_values,
                "optical_density": optical_density,
                "I0": I0,
            })

        except Exception as e:  # noqa: BLE001
            self.log_message.emit(f"❌ Fitting error: {e}")

        self.finished.emit()


# ════════════════════════════════════════════════════════════════════[...]
# Calibration Tab
# ════════════════════════════════════════════════════════════════════[...]

class FilmCalibrationTab(QWidget):
    """Film calibration tab for OD model fitting."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._calib_folder = None
        self._output_dir = None
        self._dose_to_file = {}
        self._thread = None
        self._worker = None
        self._calib_params = None
        self._build()

    def _build(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        # ── Left settings panel ────────────────────────────────────────────
        left = QFrame()
        left.setFixedWidth(350)
        left.setObjectName("settingsPanel")
        lv = QVBoxLayout(left)
        lv.setContentsMargins(12, 12, 12, 12)
        lv.setSpacing(4)

        lv.addWidget(heading_label("CALIBRATION SETUP"))
        lv.addWidget(hseparator())
        lv.addSpacing(25)

        # Calibration folder selection
        lv.addWidget(section_label("Calibration folder"))
        self.folder_lbl = QLabel("No folder selected")
        self.folder_lbl.setStyleSheet(f"color:{PRIMARY}; font-size:9pt;")
        self.folder_lbl.setWordWrap(True)
        lv.addWidget(self.folder_lbl)
        lv.addSpacing(25)
        
        btn_folder = QPushButton("Browse…")
        btn_folder.setObjectName("accent")
        btn_folder.clicked.connect(self._select_folder)
        lv.addWidget(btn_folder)
        lv.addSpacing(25)

        # Output folder selection
        lv.addWidget(section_label("Output folder"))
        self.output_lbl = QLabel("Same as calibration folder")
        self.output_lbl.setStyleSheet(f"color:{PRIMARY}; font-size:9pt;")
        self.output_lbl.setWordWrap(True)
        lv.addWidget(self.output_lbl)
        lv.addSpacing(25)
        
        btn_output = QPushButton("Browse…")
        btn_output.setObjectName("accent")
        btn_output.clicked.connect(self._select_output)
        lv.addWidget(btn_output)
        lv.addSpacing(25)

        # Crop size
        lv.addWidget(section_label("ROI size (pixels)"))
        self.crop_spin = QSpinBox()
        self.crop_spin.setRange(10, 500)
        self.crop_spin.setValue(150)
        lv.addWidget(self.crop_spin)
        lv.addSpacing(25)

        lv.addWidget(hseparator())
        lv.addSpacing(25)
        lv.addWidget(heading_label("LOG"))
        self.log = make_log(height=200)
        lv.addWidget(self.log)
        
        lv.addStretch()
        root.addWidget(left)

        # ── Right panel ───────────────────────────────────────────────────
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.setSpacing(8)

        # Header with action buttons
        hdr = QHBoxLayout()
        hdr.addWidget(heading_label("CALIBRATION DATA & FIT"))
        hdr.addStretch()

        self.scan_btn = QPushButton("🔍 Scan files")
        self.scan_btn.setObjectName("accent")
        self.scan_btn.clicked.connect(self._scan_files)
        
        self.run_btn = QPushButton("▶  Run Calibration")
        self.run_btn.setObjectName("success")
        self.run_btn.setEnabled(False)
        self.run_btn.clicked.connect(self._run_calibration)

        self.save_plot_btn = QPushButton("💾 Save plot")
        self.save_plot_btn.setEnabled(False)
        self.save_plot_btn.clicked.connect(self._save_plot)

        hdr.addWidget(self.scan_btn)
        hdr.addWidget(self.run_btn)
        hdr.addWidget(self.save_plot_btn)
        rv.addLayout(hdr)

        # Status label
        self.status_lbl = QLabel("Ready. Select a calibration folder and scan for files.")
        self.status_lbl.setStyleSheet(f"background:{BG_MAIN}; color:{PRIMARY}; padding:6px 10px; font-size:9pt;")
        rv.addWidget(self.status_lbl)

        # Plot for calibration curve
        self._fig = Figure(figsize=(8, 5))
        self._fig.patch.set_facecolor("white")
        self._ax = self._fig.add_subplot(111)
        self._ax.set_facecolor("white")
        self._canvas = FigureCanvas(self._fig)
        self._canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        rv.addWidget(self._canvas)

        root.addWidget(right, stretch=1)

    def _select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select calibration folder")
        if folder:
            self._calib_folder = folder
            self.folder_lbl.setText(folder)
            self._output_dir = folder
            self.output_lbl.setText(folder)
            log_write(self.log, f"✓ Folder selected: {Path(folder).name}")

    def _select_output(self):
        folder = QFileDialog.getExistingDirectory(self, "Select output folder")
        if folder:
            self._output_dir = folder
            self.output_lbl.setText(folder)
            log_write(self.log, f"✓ Output folder: {Path(folder).name}")

    def _scan_files(self):
        """Scan folder for TIFF files and extract doses from filenames."""
        if not self._calib_folder:
            QMessageBox.warning(self, "No folder", "Select a calibration folder first.")
            return

        files = [f for f in os.listdir(self._calib_folder) if f.endswith((".tif", ".tiff"))]
        dose_pattern = re.compile(r"(\d+\.?\d*)Gy")
        self._dose_to_file = {}

        for f in files:
            match = dose_pattern.search(f)
            if match:
                dose = float(match.group(1))
                self._dose_to_file[dose] = os.path.join(self._calib_folder, f)
            else:
                log_write(self.log, f"⚠ Skipping (no dose in name): {f}", "warning")

        log_write(self.log, f"🔍 Found {len(self._dose_to_file)} calibration files")
        for dose in sorted(self._dose_to_file.keys()):
            log_write(self.log, f"   {dose:.3f} Gy")

        self.run_btn.setEnabled(len(self._dose_to_file) >= 2)

    def _run_calibration(self):
        """Run calibration in background thread."""
        if not self._dose_to_file:
            QMessageBox.warning(self, "No files", "Scan for files first.")
            return

        self.run_btn.setEnabled(False)
        self.status_lbl.setText("Running calibration...")

        self._worker = _CalibrationWorker(
            self._dose_to_file,
            self.crop_spin.value(),
            self._output_dir
        )
        self._thread = QThread()
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.log_message.connect(self._on_log)
        self._worker.roi_needed.connect(self._on_roi_needed)
        self._worker.calibration_done.connect(self._on_calibration_done)
        self._worker.finished.connect(self._thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)

        self._thread.start()

    def _on_roi_needed(self, data: dict):
        """Handle ROI approval request from worker."""
        dose = data["dose"]
        image = data["image"]
        center_x = data["center_x"]
        center_y = data["center_y"]
        mean_val = data["mean_val"]
        
        # Show dialog
        dialog = ROISelectionDialog(
            dose, image, center_x, center_y,
            self.crop_spin.value(), mean_val, self
        )
        result = dialog.exec_()
        
        if result == QDialog.Accepted and dialog.approved:
            # Send approval back to worker
            self._worker.process_roi_result(
                dose,
                mean_val,
                dialog.approved_dose
            )

    def _on_log(self, msg: str):
        """Receive log message from worker."""
        log_write(self.log, msg)

    def _on_calibration_done(self, params: dict):
        """Receive calibration results and plot."""
        self._calib_params = params
        self.run_btn.setEnabled(True)
        self.save_plot_btn.setEnabled(True)
        self.status_lbl.setText("✅ Calibration complete!")
        self._plot_calibration(params)

    def _plot_calibration(self, params):
        """Plot OD vs. Dose calibration curve."""
        ax = self._ax
        ax.clear()

        doses = params["doses"]
        od_values = params["optical_density"]
        params["I0"]

        # Plot measured points
        ax.scatter(doses, od_values, color="red", s=50, label="Measured", zorder=5)

        # Plot fit
        doses_fit = np.linspace(0, max(doses) + 5, 200)
        od_fit = od_model(doses_fit, params["a"], params["b"], params["c"])
        ax.plot(doses_fit, od_fit, color="blue", lw=2, label="OD Model Fit")

        ax.set_xlabel("Dose (Gy)", fontsize=11)
        ax.set_ylabel("Optical Density", fontsize=11)
        ax.set_title(
            f"Film Calibration Curve (R² = {params['r_squared']:.4f})",
            fontsize=12, fontweight="bold"
        )
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)

        self._fig.tight_layout()
        self._canvas.draw()

    def _save_plot(self):
        """Save calibration plot to file."""
        if not self._calib_params:
            QMessageBox.warning(self, "No data", "Run calibration first.")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save calibration plot", "calibration_curve.png",
            "PNG files (*.png);;PDF files (*.pdf);;All files (*.*)"
        )
        
        if file_path:
            try:
                self._fig.savefig(file_path, dpi=150, bbox_inches="tight")
                QMessageBox.information(self, "Saved", f"Plot saved to:\n{file_path}")
                log_write(self.log, f"✓ Plot saved: {Path(file_path).name}")
            except Exception as e:  # noqa: BLE001
                QMessageBox.critical(self, "Error", f"Failed to save plot:\n{e}")