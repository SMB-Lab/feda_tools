from PyQt6 import QtWidgets
import pyqtgraph as pg

class PlotWidget(pg.PlotWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
