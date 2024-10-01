from PyQt6 import QtWidgets, QtCore
from .widgets import MatplotlibCanvas
from feda_tools.core import model
import numpy as np

class Fit23PreviewWindow(QtWidgets.QWidget):
    analysis_started = QtCore.pyqtSignal()

    def __init__(self, burst_data, fit23):
        super().__init__()
        self.burst_data = burst_data
        self.fit23 = fit23
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Fit23 Preview')
        layout = QtWidgets.QVBoxLayout()

        # Matplotlib canvas
        self.canvas = MatplotlibCanvas(self, width=5, height=4, dpi=100)
        layout.addWidget(self.canvas)

        # Run Analysis button
        self.run_button = QtWidgets.QPushButton('Run Full Analysis')
        layout.addWidget(self.run_button)
        self.run_button.clicked.connect(self.start_analysis)

        self.setLayout(layout)
        self.plot_fit23()

    def plot_fit23(self):
        # This is a placeholder for actual plotting code
        self.canvas.axes.clear()
        self.canvas.axes.plot([0, 1, 2], [0, 1, 0], label='Sample Fit')
        self.canvas.axes.set_xlabel('Time')
        self.canvas.axes.set_ylabel('Intensity')
        self.canvas.axes.legend()
        self.canvas.draw()

    def start_analysis(self):
        # Emit a signal or call a method to start the full analysis
        self.analysis_started.emit()