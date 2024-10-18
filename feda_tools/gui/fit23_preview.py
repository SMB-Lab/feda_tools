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
        self.canvas.axes.clear()

        counts_model = [x for x in self.fit23.model]
        
        # Plot data
        self.canvas.axes.semilogy([x for x in self.fit23.data]/np.max(self.counts), label='Data')
        
        # Plot IRF
        self.canvas.axes.semilogy(self.counts_irf/np.max(self.counts_irf), label='IRF')
        
        # Plot model
        self.canvas.axes.semilogy(counts_model/np.max(counts_model), label='Model')
        
        # Set y-axis limits
        self.canvas.axes.set_ylim(0.001, np.power(10,1))
        
        # Set labels and legend
        self.canvas.axes.set_ylabel(r'log(Counts)')
        self.canvas.axes.set_xlabel(r'Channel Nbr.')
        self.canvas.axes.legend()
        
        # Draw the canvas
        self.canvas.draw()


    def start_analysis(self):
        # Emit a signal or call a method to start the full analysis
        self.analysis_started.emit()