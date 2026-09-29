# -*- coding: utf-8 -*-
"""

"""

import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QTabWidget
from Film_Analysis import FilmTab
from Fibre_Analysis import FibreTab
from Fibre_Prediction import PredictionTab

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("FLASH SARRP Geneva Dosimetry")
        self.resize(1200, 750)

        tabs = QTabWidget()
        # tabs.addTab(CalibTab(), "Film Calibration")
        tabs.addTab(FilmTab(), "Film Analysis")
        tabs.addTab(FibreTab(), "Fibre Pulse Analysis")
        tabs.addTab(PredictionTab(), "Dose Predictions")

        self.setCentralWidget(tabs)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())

