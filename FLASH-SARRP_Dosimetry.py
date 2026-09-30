"""
main
"""

import sys

from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication, QMainWindow, QTabWidget

from Fibre_Analysis import FibreTab
from Fibre_Prediction import PredictionTab
from Film_Analysis import FilmTab
from Film_Calibration import FilmCalibrationTab


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

        self.setCentralWidget(tabs)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon('LogoApp.ico'))
    app.setApplicationName('FLASH-SARRP Dosimetry')
    app.setWindowIcon(QIcon('LogoApp.ico'))
    win = MainWindow()
    win.show()
    sys.exit(app.exec())

