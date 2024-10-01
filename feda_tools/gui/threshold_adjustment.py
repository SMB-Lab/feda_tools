from PyQt6 import QtWidgets, QtCore
from .widgets import MatplotlibCanvas
from feda_tools.core import analysis as an
import numpy as np

class ThresholdAdjustmentWindow(QtWidgets.QWidget):
    threshold_changed = QtCore.pyqtSignal(float)

    def __init__(self, logrunavg, mu, std, initial_threshold, bins_y):
        super().__init__()
        self.logrunavg = logrunavg
        self.mu = mu
        self.std = std
        self.initial_threshold = initial_threshold
        self.bins_y = bins_y
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Threshold Adjustment')
        layout = QtWidgets.QVBoxLayout()

        # Matplotlib canvas
        self.canvas = MatplotlibCanvas(self, width=5, height=4, dpi=100)
        layout.addWidget(self.canvas)

        # Threshold slider
        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setMinimum(1)
        self.slider.setMaximum(10)
        self.slider.setValue(int(self.initial_threshold))
        self.slider.valueChanged.connect(self.update_plot)
        layout.addWidget(self.slider)

        # Threshold label
        self.label = QtWidgets.QLabel(f'Threshold: {self.initial_threshold} sigma')
        layout.addWidget(self.label)

        # Next button
        self.next_button = QtWidgets.QPushButton('Next')
        layout.addWidget(self.next_button)

        self.setLayout(layout)
        self.plot_data(self.initial_threshold)

    def plot_data(self, threshold_multiplier):
        threshold_value = self.mu - threshold_multiplier * self.std
        filtered_values = np.ma.masked_greater(self.logrunavg, threshold_value)

        self.canvas.axes.clear()
        self.canvas.axes.plot(self.logrunavg, label='Running Average', linestyle='None', marker='o', markersize=2)
        self.canvas.axes.plot(filtered_values, label='Threshold Values', linestyle='None', marker='.', markersize=2)
        self.canvas.axes.axhline(y=threshold_value, color='r', linestyle='--', label=f'Threshold ({threshold_multiplier}σ)')
        self.canvas.axes.set_xlabel('Photon Event #')
        self.canvas.axes.set_ylabel('log(Photon Interval Time)')
        self.canvas.axes.legend()
        self.canvas.draw()

    def update_plot(self, value):
        threshold_multiplier = value
        self.label.setText(f'Threshold: {threshold_multiplier} sigma')
        self.plot_data(threshold_multiplier)
        self.threshold_changed.emit(threshold_multiplier)