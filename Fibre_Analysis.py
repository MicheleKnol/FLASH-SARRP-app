"""
Fibre_analysis.py
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtCore import QObject, QThread, Signal
from PyQt5.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, filtfilt, iirnotch

from utils import (
    BG_MAIN,
    BG_SURFACE,
    PRIMARY,
    TEXT_MUTED,
    heading_label,
    hseparator,
    log_write,
    make_entry,
    make_log,
    section_label,
)

# ═════════════════════════════════════════════════════════════════════════════
# Pulse-detection helpers
# ═════════════════════════════════════════════════════════════════════════════

def lowpass_filter(signal, time, cutoff_hz, order=4):
    """Zero-phase Butterworth low-pass filter."""
    dt  = np.median(np.diff(time))
    fs  = 1.0 / dt
    nyq = 0.5 * fs
    cutoff_hz = min(cutoff_hz, nyq * 0.99)
    b, a = butter(order, cutoff_hz / nyq, btype="low")
    return filtfilt(b, a, signal)


def notch_filter(signal, time, f0_hz=50.0, Q=30):
    """Zero-phase IIR notch filter (e.g. 50 Hz mains hum removal)."""
    dt = np.median(np.diff(time))
    fs = 1.0 / dt
    b, a = iirnotch(f0_hz, Q, fs)
    return filtfilt(b, a, signal)


def smooth_signal(signal, time, cutoff_hz=0.5, order=6,
                  remove_mains=False, mains_hz=50.0, mains_Q=30,
                  ma_window=5):
    """Moving-average pre-smooth → optional notch → double Butterworth LP."""
    y = signal.copy().astype(float)
    if ma_window > 1:
        y = uniform_filter1d(y, size=ma_window, mode="nearest")
    if remove_mains:
        y = notch_filter(y, time, f0_hz=mains_hz, Q=mains_Q)
    y = lowpass_filter(y, time, cutoff_hz=cutoff_hz, order=order)
    y = lowpass_filter(y, time, cutoff_hz=cutoff_hz, order=order)
    return y


def mad(x):
    """Median absolute deviation — robust noise estimator."""
    med = np.median(x)
    return np.median(np.abs(x - med))


def find_pulse_window(time, y, pre_end_ms=20.0, hi_frac=0.2, min_run_ms=0.5):
    """Locate the start and end of the main pulse."""
    pre_mask = time < pre_end_ms
    if not np.any(pre_mask):
        n10 = max(1, int(0.1 * len(y)))
        pre_mask = np.zeros_like(y, dtype=bool)
        pre_mask[:n10] = True

    base  = np.mean(y[pre_mask])
    noise = mad(y[pre_mask]) * 1.4826
    peak  = np.max(y)
    thr   = base + max(5 * max(noise, 1e-9), hi_frac * (peak - base))

    dt      = np.median(np.diff(time))
    min_run = max(1, round(min_run_ms / max(dt, 1e-9)))
    above   = y > thr

    start_idx, run = None, 0
    for i, v in enumerate(above):
        run = run + 1 if v else 0
        if run >= min_run:
            start_idx = i - min_run + 1
            break

    end_idx, run = None, 0
    for j in range(len(above) - 1, -1, -1):
        run = run + 1 if above[j] else 0
        if run >= min_run:
            end_idx = j + min_run - 1
            if end_idx >= len(above):
                end_idx = len(above) - 1
            break

    if start_idx is None:
        raise ValueError("Pulse start not found — try lowering hi_frac or min_run_ms.")
    if end_idx is None or end_idx <= start_idx:
        raise ValueError("Pulse end not found — try lowering hi_frac or min_run_ms.")

    return start_idx, end_idx, base


def clean_binary_runs(flags, min_run):
    """Merge short True/False runs (< min_run samples) into their neighbours."""
    flags = flags.astype(int)
    starts, lengths, values = [], [], []
    i, n = 0, len(flags)
    while i < n:
        j = i
        while j < n and flags[j] == flags[i]:
            j += 1
        starts.append(i); lengths.append(j - i); values.append(flags[i])
        i = j

    for k in range(1, len(starts) - 1):
        if lengths[k] < min_run:
            left, right = lengths[k - 1], lengths[k + 1]
            values[k] = values[k - 1] if left >= right else values[k + 1]

    out = np.empty(n, dtype=int)
    for s, L, v in zip(starts, lengths, values):
        out[s : s + L] = v
    return out.astype(bool)


def analyze_pulse(time, signal, cutoff_hz=2.0, filter_order=4,
                  remove_mains=False, mains_hz=50.0):
    """
    Smooth → detect pulse window → segment into low/high plateaus → metrics.

    Returns
    -------
    results   : dict of scalar metrics
    smoothed  : filtered voltage array (same length as inputs)
    seg_times : 8-tuple of boundary times for plotting
    """
    y = smooth_signal(signal, time, cutoff_hz=cutoff_hz, order=filter_order,
                      remove_mains=remove_mains, mains_hz=mains_hz)

    start_idx, end_idx, baseline_before = find_pulse_window(time, y)
    t_start = time[start_idx]
    t_end   = time[end_idx]

    after_mask     = np.arange(len(time)) > end_idx
    baseline_after = np.mean(y[after_mask]) if np.any(after_mask) else y[end_idx]

    tw = time[start_idx : end_idx + 1]
    yw = y    [start_idx : end_idx + 1]

    fixed_threshold = np.max(signal) * 0.75
    is_high = yw >= fixed_threshold
    dt = np.median(np.diff(tw))
    min_run_samples = max(1, round(0.5 / max(dt, 1e-9)))
    is_high = clean_binary_runs(is_high, min_run_samples)

    if not np.any(is_high):
        raise ValueError("No high plateau detected inside pulse window.")

    hi_start_rel = 0 if is_high[0] else int(np.argmax(is_high))
    hi_end_rel   = len(is_high) - 1 - int(np.argmax(is_high[::-1]))

    low1_start_rel = 0
    low1_end_rel   = max(0, hi_start_rel - 1)
    low2_start_rel = min(len(tw) - 1, hi_end_rel + 1)
    low2_end_rel   = len(tw) - 1

    t_low1_start, t_low1_end = tw[low1_start_rel], tw[low1_end_rel]
    t_high_start, t_high_end = tw[hi_start_rel],   tw[hi_end_rel]
    t_low2_start, t_low2_end = tw[low2_start_rel], tw[low2_end_rel]

    delay               = max(0.0, t_high_start - t_start)
    duration_both       = max(0.0, t_high_end - t_high_start)
    duration_tube2_only = max(0.0, t_end - t_high_end)
    total_duration      = max(0.0, t_end - t_start)
    duration_tube1      = delay + duration_both
    duration_tube2      = duration_both + duration_tube2_only

    high_mask   = (time >= t_high_start) & (time <= t_high_end)
    high_values = y[high_mask]

    amplitude = np.percentile(high_values, 99) - baseline_before
    area = (np.trapz(y[start_idx : end_idx + 1], time[start_idx : end_idx + 1])
            - baseline_before * total_duration)

    results = {
        "Mean baseline before (V)":         baseline_before,
        "Mean baseline after (V)":           baseline_after,
        "Total duration (ms)":               total_duration,
        "Delay (ms)":                        delay,
        "Duration Tube 1 (ms)":              duration_tube1,
        "Duration Tube 2 (ms)":              duration_tube2,
        "Area under curve (V·ms)":           area,
        "Amplitude (V)":                     amplitude
    }

    seg_times = (t_start, t_end,
                 t_low1_start, t_low1_end,
                 t_high_start, t_high_end,
                 t_low2_start, t_low2_end)

    return results, y, seg_times


# ═════════════════════════════════════════════════════════════════════════════
# File loading
# ═════════════════════════════════════════════════════════════════════════════

def load_fibre_file(file_path: str):
    """
    Load a pulse file and return (time_ms, voltage) arrays cropped to the
    analysis window.  Supports .csv, .dat, .txt.
    """
    ext = Path(file_path).suffix.lower()

    if ext == ".csv":
        df = pd.read_csv(file_path, skiprows=[1], sep=";")
        df.columns = (df.columns.str.strip()
                      .str.replace(" ", "_")
                      .str.replace(r"[()]", "", regex=True))
        df["Time"]      = pd.to_numeric(df["Time"],      errors="coerce")
        df["Channel_A"] = pd.to_numeric(df["Channel_A"], errors="coerce")
        df = df.dropna(subset=["Time", "Channel_A"])
        time   = df["Time"].values
        signal = df["Channel_A"].values
        mask   = (time >= -50) & (time <= 150)

    elif ext == ".dat":
        data   = np.fromfile(file_path, dtype=np.int16)
        signal = (data - 32767) * 0.000152941
        dt     = 1000 / 5000
        time   = np.arange(len(signal)) * dt
        mask   = (time >= 50) & (time <= 400)

    elif ext == ".txt":
        data = pd.read_csv(file_path, sep=r"\s+", header=None, engine="python")
        data = data.iloc[:, :-16].values.flatten().astype(np.float64)
        signal = (data - 32767) * 0.000152941
        dt     = 500 / 5000
        time   = np.arange(len(signal)) * dt
        mask   = (time >= 40) & (time <= 200)

    else:
        raise ValueError(f"Unsupported file type: {ext}")

    return time[mask], signal[mask]


# ═════════════════════════════════════════════════════════════════════════════
# Background worker (keeps the UI responsive during batch processing)
# ═════════════════════════════════════════════════════════════════════════════

class _FibreWorker(QObject):
    """Runs the batch in a QThread; emits signals back to the UI thread."""
    progress  = Signal(int)           # 0–100
    file_done = Signal(str, str)      # (log_line, colour)  colour = "" | "error"
    finished  = Signal(int, int)      # (succeeded, total)
    highlight = Signal(int)           # list index to highlight
    plot_ready = Signal(object, object, object, object, str) #Plot

    def __init__(self, file_queue, cutoff, order, gain_db, mains):
        super().__init__()
        self.file_queue = file_queue
        self.cutoff     = cutoff
        self.order      = order
        self.gain_db    = gain_db
        self.mains      = mains
        self._plot_history = []
        self._plot_index = -1

    def run(self):
        total   = len(self.file_queue)
        results = []
        gain    = 10 ** ((self.gain_db - 70) / 20.0)

        for idx, path in enumerate(self.file_queue):
            name = Path(path).stem
            self.highlight.emit(idx)

            try:
                time, signal = load_fibre_file(path)
                res, smoothed, seg = analyze_pulse(
                    time, signal,
                    cutoff_hz=self.cutoff, filter_order=self.order,
                    remove_mains=self.mains,
                )
                
                self.plot_ready.emit(time, signal, smoothed, seg, name)
                
                for k in ("Area under curve (V·ms)", "Amplitude (V)",
                          "Mean baseline before (V)", "Mean baseline after (V)"):
                    res[k] /= gain

                outdir   = str(Path(path).parent)
                basename = Path(path).stem

                # Text results
                txt_path = os.path.join(outdir, f"{basename}_analysis.txt")
                with open(txt_path, "w") as f:
                    f.write(f"Pulse Analysis — {Path(path).name}\n")
                    f.write(f"Cutoff : {self.cutoff} Hz | Order : {self.order} | "
                            f"Notch : {'on' if self.mains else 'off'} | "
                            f"Gain : {self.gain_db} dB\n\n")
                    for k, v in res.items():
                        f.write(f"{k:40s}: {v:.3f}\n")

                # Analysis plot
                (t_low1_s, t_low1_e,
                 t_high_s, t_high_e,
                 t_low2_s, t_low2_e) = seg

                fig = Figure(figsize=(9, 4))
                ax = fig.add_subplot(111)
                fig.patch.set_facecolor(BG_MAIN)
                ax.set_facecolor("#12151a")
                ax.tick_params(colors=PRIMARY)
                for sp in ax.spines.values():
                    sp.set_color(PRIMARY)

                ax.plot(time, signal,   alpha=0.25, color="white", label="Raw")
                ax.plot(time, smoothed, lw=1.5, color="#f44336",
                        label=f"LP {self.cutoff} Hz")
                ax.axvspan(t_low1_s, t_low1_e, alpha=0.20, color="#f44336",
                           label="One tube")
                ax.axvspan(t_high_s, t_high_e, alpha=0.20, color=TEXT_MUTED,
                           label="Both tubes")
                ax.axvspan(t_low2_s, t_low2_e, alpha=0.20, color="#f44336")

                ax.set_xlabel("Time (ms)", color=PRIMARY)
                ax.set_ylabel("Voltage (V)", color=PRIMARY)
                ax.set_title(Path(path).name, color=PRIMARY, fontsize=9)
                ax.legend(fontsize=8, facecolor=BG_SURFACE, labelcolor=PRIMARY)
                ax.grid(True, alpha=0.2, color=PRIMARY)
                fig.tight_layout()
                fig.savefig(os.path.join(outdir, f"{basename}_analysis.png"),
                            dpi=150, bbox_inches="tight",
                            facecolor=fig.get_facecolor())

                self.file_done.emit(
                    f"✓ {name}: dur {res['Total duration (ms)']:.1f} ms  "
                    f"amp {res['Amplitude (V)']:.3f} V", "")
                results.append({"file": name, **res})

            except Exception as e:  # noqa: BLE001
                self.file_done.emit(f"✗ {name}: {e}", "error")

            self.progress.emit(int((idx + 1) / total * 100))

        # Batch summary CSV
        if results:
            csv_path = os.path.join(
                str(Path(self.file_queue[0]).parent),
                "fibre_batch_summary.csv",
            )
            pd.DataFrame(results).to_csv(csv_path, index=False)
            self.file_done.emit("📊 Summary: fibre_batch_summary.csv", "")

        self.finished.emit(len(results), total)


# ═════════════════════════════════════════════════════════════════════════════
# Fibre Tab — the QWidget registered in the main QTabWidget
# ═════════════════════════════════════════════════════════════════════════════

class FibreTab(QWidget):
    """Drop this into any QTabWidget."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._file_queue = []
        self._thread     = None
        self._worker     = None
        self._plot_history = []
        self._plot_index = -1
        self._build()

    # ── layout ────────────────────────────────────────────────────────────────
    def _build(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        # ── Left settings panel ───────────────────────────────────────────────
        left = QFrame()
        left.setFixedWidth(270)
        left.setStyleSheet(f"background:{BG_SURFACE}; border-radius:4px;")
        lv = QVBoxLayout(left)
        lv.setContentsMargins(12, 12, 12, 12)
        lv.setSpacing(4)

        lv.addWidget(heading_label("SETTINGS"))
        lv.addWidget(hseparator())

        lv.addWidget(section_label("Butterworth low-pass cutoff (Hz)"))
        self.cutoff_entry = make_entry("1.0", width=80)
        lv.addWidget(self.cutoff_entry)

        lv.addWidget(section_label("Filter order"))
        self.order_entry = make_entry("4", width=80)
        lv.addWidget(self.order_entry)

        lv.addWidget(section_label("Gain (dB)  [reference: 70 dB]"))
        self.gain_entry = make_entry("70", width=80)
        lv.addWidget(self.gain_entry)

        self.mains_cb = QCheckBox("50 Hz notch filter")
        lv.addWidget(self.mains_cb)

        lv.addWidget(hseparator())
        note = QLabel("Output files are saved next\nto each source file.")
        note.setStyleSheet(f"color:{PRIMARY}; font-size:9pt;")
        lv.addWidget(note)

        lv.addStretch()
        root.addWidget(left)

        # ── Right queue / log panel ───────────────────────────────────────────
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)

        hdr = QHBoxLayout()
        hdr.addWidget(heading_label("FIBRE FILE QUEUE"))
        hdr.addStretch()

        btn_add = QPushButton("+ Add Files")
        btn_add.setObjectName("accent")
        btn_add.clicked.connect(self._add_files)
        btn_clr = QPushButton("Clear")
        btn_clr.clicked.connect(self._clear)
        self.run_btn = QPushButton("▶  Run All")
        self.run_btn.setObjectName("success")
        self.run_btn.clicked.connect(self._run_all)

        for b in (btn_add, btn_clr, self.run_btn):
            hdr.addWidget(b)
        rv.addLayout(hdr)

        self.file_list = QListWidget()
        rv.addWidget(self.file_list)

        rv.addWidget(heading_label("RESULTS LOG"))
        self.log = make_log(height=180)
        rv.addWidget(self.log)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedHeight(8)
        self.progress_bar.setTextVisible(False)
        rv.addWidget(self.progress_bar)

        # Embedded plot
        self.figure = Figure(figsize=(6, 3))
        self.canvas = FigureCanvas(self.figure)
        rv.addWidget(self.canvas)
        nav = QHBoxLayout()
        self.prev_btn = QPushButton("◀ Prev")
        self.next_btn = QPushButton("Next ▶")
        self.prev_btn.clicked.connect(self._prev_plot)
        self.next_btn.clicked.connect(self._next_plot)
        nav.addWidget(self.prev_btn)
        nav.addWidget(self.next_btn)
        rv.addLayout(nav)
        
        root.addWidget(right)

    # ── file helpers ──────────────────────────────────────────────────────────
    def _add_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select fibre pulse files", "",
            "Pulse files (*.dat *.csv *.txt);;All files (*.*)",
        )
        for p in paths:
            if p not in self._file_queue:
                self._file_queue.append(p)
                self.file_list.addItem(Path(p).name)

    def _clear(self):
        self._file_queue.clear()
        self.file_list.clear()

    # ── batch run ─────────────────────────────────────────────────────────────
    def _run_all(self):
        if not self._file_queue:
            QMessageBox.warning(self, "No files", "Add files to the queue first.")
            return

        try:
            cutoff  = float(self.cutoff_entry.text())
            order   = int(self.order_entry.text())
            gain_db = float(self.gain_entry.text())
        except ValueError as e:
            QMessageBox.critical(self, "Settings error", str(e))
            return

        mains = self.mains_cb.isChecked()

        self.run_btn.setEnabled(False)
        self.progress_bar.setValue(0)

        self._worker = _FibreWorker(
            list(self._file_queue), cutoff, order, gain_db, mains)
        self._thread = QThread()
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self.progress_bar.setValue)
        self._worker.highlight.connect(self._highlight_row)
        self._worker.file_done.connect(self._on_file_done)
        self._worker.finished.connect(self._on_finished)
        self._worker.finished.connect(self._thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        self._worker.plot_ready.connect(self._update_plot)

        self._thread.start()

    def _update_plot(self, time, signal, smoothed, seg, name):
        self._plot_history.append((time, signal, smoothed, seg, name))
        self._plot_index = len(self._plot_history) - 1
        self._render_plot()
        
    def _render_plot(self):
        if not self._plot_history:
            return
    
        time, signal, smoothed, seg, name = self._plot_history[self._plot_index]
    
        self.figure.clear()
        ax = self.figure.add_subplot(111)
    
        (t_low1_s, t_low1_e,
         t_high_s, t_high_e,
         t_low2_s, t_low2_e) = seg
    
        ax.plot(time, signal, alpha=0.25, color="black", label="Raw")
        ax.plot(time, smoothed, lw=1.5, color="red", label="Filtered")
    
        ax.axvspan(t_low1_s, t_low1_e, alpha=0.2, color="red", label="One tube")
        ax.axvspan(t_high_s, t_high_e, alpha=0.2, color="blue", label="Both tubes")
        ax.axvspan(t_low2_s, t_low2_e, alpha=0.2, color="red")
    
        ax.set_title(name)
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Voltage (V)")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
        self.canvas.draw()

    def _prev_plot(self):
        if self._plot_index > 0:
            self._plot_index -= 1
            self._render_plot()

    def _next_plot(self):
        if self._plot_index < len(self._plot_history) - 1:
            self._plot_index += 1
            self._render_plot()

    def _highlight_row(self, idx: int):
        self.file_list.clearSelection()
        item = self.file_list.item(idx)
        if item:
            item.setSelected(True)
            self.file_list.setCurrentItem(item)

    def _on_file_done(self, msg: str, colour: str):
        log_write(self.log, msg)

    def _on_finished(self, succeeded: int, total: int):
        log_write(self.log, f"── Done ({succeeded}/{total} succeeded) ──")
        self.run_btn.setEnabled(True)
        QMessageBox.information(
            self, "Done",
            f"Fibre batch complete!\n{succeeded}/{total} files processed.",
        )