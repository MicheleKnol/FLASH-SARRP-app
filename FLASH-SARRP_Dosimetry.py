"""
main
"""

import os
import sys

from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication, QMainWindow, QTabWidget

from DoseRate_Analysis import DoseRateAnalysisTab
from Fibre_Analysis import FibreTab
from Fibre_Prediction import PredictionTab
from Film_Analysis import FilmTab
from Film_Calibration import FilmCalibrationTab


def resource_path(relative_path):
    base_path = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("FLASH SARRP Geneva Dosimetry")
        self.resize(1600, 1000)

        tabs = QTabWidget()
        tabs.addTab(FilmCalibrationTab(), "Film Calibration")
        tabs.addTab(FilmTab(), "Film Analysis")
        tabs.addTab(FibreTab(), "Fibre Pulse Analysis")
        tabs.addTab(PredictionTab(), "Dose Predictions")
        tabs.addTab(DoseRateAnalysisTab(), "Dose Rate Predictions")

        self.setCentralWidget(tabs)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon('LogoApp.ico'))
    app.setApplicationName('FLASH-SARRP Dosimetry')
    app.setWindowIcon(QIcon('LogoApp.ico'))
    win = MainWindow()
    win.show()
    sys.exit(app.exec())

