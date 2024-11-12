from PyQt6 import QtWidgets, QtCore
from .widgets import PlotWidget
from feda_tools.core import analysis as an
from feda_tools.core import data as dat
import numpy as np
import pyqtgraph as pg

class ThresholdAdjustmentWindow(QtWidgets.QWidget):
    threshold_changed = QtCore.pyqtSignal(float)
    analysis_started = QtCore.pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle('Threshold Adjustment')
        self.init_ui()

    def init_ui(self):
        layout = QtWidgets.QVBoxLayout()

        # File selection (PTU, IRF, BKG)
        file_selection_layout = QtWidgets.QFormLayout()
        self.file_ptu_input = QtWidgets.QLineEdit()
        self.file_irf_input = QtWidgets.QLineEdit()
        self.file_bkg_input = QtWidgets.QLineEdit()

        file_ptu_button = QtWidgets.QPushButton('Browse')
        file_ptu_button.clicked.connect(self.select_ptu_file)
        file_irf_button = QtWidgets.QPushButton('Browse')
        file_irf_button.clicked.connect(self.select_irf_file)
        file_bkg_button = QtWidgets.QPushButton('Browse')
        file_bkg_button.clicked.connect(self.select_bkg_file)

        file_selection_layout.addRow('Main PTU File:', self._create_file_input(self.file_ptu_input, file_ptu_button))
        file_selection_layout.addRow('IRF File:', self._create_file_input(self.file_irf_input, file_irf_button))
        file_selection_layout.addRow('Background File:', self._create_file_input(self.file_bkg_input, file_bkg_button))

        layout.addLayout(file_selection_layout)

        # Output directory selection
        output_dir_layout = QtWidgets.QHBoxLayout()
        output_dir_label = QtWidgets.QLabel('Output Directory:')
        self.output_dir_input = QtWidgets.QLineEdit()
        output_dir_button = QtWidgets.QPushButton('Browse')
        output_dir_button.clicked.connect(self.select_output_directory)
        output_dir_layout.addWidget(output_dir_label)
        output_dir_layout.addWidget(self.output_dir_input)
        output_dir_layout.addWidget(output_dir_button)
        
        layout.addLayout(output_dir_layout)

        # Chunk size input
        chunk_size_layout = QtWidgets.QHBoxLayout()
        chunk_size_label = QtWidgets.QLabel('Chunk Size:')
        self.chunk_size_input = QtWidgets.QLineEdit('30000')  # Default chunk size
        chunk_size_layout.addWidget(chunk_size_label)
        chunk_size_layout.addWidget(self.chunk_size_input)
        layout.addLayout(chunk_size_layout)

        # Load Data button
        load_data_button = QtWidgets.QPushButton('Load Data')
        load_data_button.clicked.connect(self.load_data)
        layout.addWidget(load_data_button)

        # PyQtGraph Plot Widget
        self.plot_widget = PlotWidget()
        layout.addWidget(self.plot_widget)

        # Threshold slider
        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setMinimum(1)
        self.slider.setMaximum(10)
        self.slider.setValue(4)  # Default threshold multiplier
        self.slider.valueChanged.connect(self.update_plot)
        self.slider.setEnabled(False)  # Disabled until data is loaded
        layout.addWidget(self.slider)

        # Threshold label
        self.label = QtWidgets.QLabel('Threshold: 4 sigma')
        layout.addWidget(self.label)

        # Next button
        self.next_button = QtWidgets.QPushButton('Next')
        self.next_button.clicked.connect(self.start_analysis)
        self.next_button.setEnabled(False)  # Disabled until data is loaded
        layout.addWidget(self.next_button)

        self.setLayout(layout)

    def _create_file_input(self, line_edit, button):
        container = QtWidgets.QHBoxLayout()
        container.addWidget(line_edit)
        container.addWidget(button)
        widget = QtWidgets.QWidget()
        widget.setLayout(container)
        return widget

    def select_ptu_file(self):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Select PTU File')
        if filename:
            self.file_ptu_input.setText(filename)

    def select_irf_file(self):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Select IRF File')
        if filename:
            self.file_irf_input.setText(filename)

    def select_bkg_file(self):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Select Background File')
        if filename:
            self.file_bkg_input.setText(filename)
            
    def select_output_directory(self):
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select Output Directory')
        if directory:
            self.output_dir_input.setText(directory)

    def load_data(self):
        file_ptu = self.file_ptu_input.text()
        file_irf = self.file_irf_input.text()
        file_bkg = self.file_bkg_input.text()
        chunk_size_text = self.chunk_size_input.text()

        if not all([file_ptu, file_irf, file_bkg, chunk_size_text]):
            QtWidgets.QMessageBox.warning(self, 'Error', 'Please select all files and specify chunk size.')
            return

        try:
            self.chunk_size = int(chunk_size_text)
            if self.chunk_size <= 0:
                raise ValueError
        except ValueError:
            QtWidgets.QMessageBox.warning(self, 'Error', 'Invalid chunk size. Please enter a positive integer.')
            return

        # Load data using core functions
        try:
            self.data_ptu, self.data_irf, self.data_bkg = dat.load_ptu_files(file_ptu, file_irf, file_bkg)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, 'Error', f'Failed to load data files:\n{e}')
            return

        # Prepare data for threshold adjustment
        self.prepare_data()
        self.slider.setEnabled(True)
        self.next_button.setEnabled(True)
        self.plot_data(self.slider.value())

    def prepare_data(self):
        # Extract data
        all_macro_times = self.data_ptu.macro_times
        all_micro_times = self.data_ptu.micro_times

        # Get resolutions
        macro_res = self.data_ptu.get_header().macro_time_resolution
        micro_res = self.data_ptu.get_header().micro_time_resolution

        # Calculate interphoton arrival times
        photon_time_intervals, _ = an.interphoton_arrival_times(
            all_macro_times, all_micro_times, macro_res, micro_res
        )

        # Calculate running average
        window_size = 30
        running_avg = an.running_average(photon_time_intervals, window_size)
        self.logrunavg = np.log10(running_avg)

        # Estimate background noise
        bins_y = 141
        self.mu, self.std, _, _, _ = an.estimate_background_noise(self.logrunavg, bins_y)

    def plot_data(self, threshold_multiplier):
        threshold_value = self.mu - threshold_multiplier * self.std
        filtered_values = np.ma.masked_greater(self.logrunavg, threshold_value)

        self.plot_widget.clear()

        max_points = 25000
        total_points = len(self.logrunavg)
        step = max(1, total_points // max_points)
        
        # Take evenly spaced points
        subset_indices = slice(0, total_points, step)
        
        # Plot the subsets
        self.plot_widget.plot(
            self.logrunavg[subset_indices], 
            pen=None, 
            symbol='o', 
            symbolSize=4, 
            name='Running Average'
        )
        self.plot_widget.plot(
            filtered_values[subset_indices], 
            pen=None, 
            symbol='o', 
            symbolSize=4, 
            brush='r', 
            name='Threshold Values'
        )
        
        # Add threshold line
        self.plot_widget.addLine(
            y=threshold_value, 
            pen=pg.mkPen('r', style=QtCore.Qt.PenStyle.DashLine)
        )

        # Set labels and legend
        self.plot_widget.setLabel('bottom', 'Photon Event #')
        self.plot_widget.setLabel('left', 'log(Photon Interval Time)')
        self.plot_widget.addLegend()

    def update_plot(self, value):
        threshold_multiplier = value
        self.label.setText(f'Threshold: {threshold_multiplier} sigma')
        self.plot_data(threshold_multiplier)
        self.threshold_changed.emit(threshold_multiplier)

    def start_analysis(self):
        # Proceed to Fit23 preview
        threshold_multiplier = self.slider.value()
        threshold_value = self.mu - threshold_multiplier * self.std
        burst_index, _ = dat.extract_greater(self.logrunavg, threshold_value)

        # Extract the entire chunk data
        from .fit23_preview import Fit23PreviewWindow

        self.fit23_window = Fit23PreviewWindow(
            self.data_ptu, self.data_irf, burst_index, self.chunk_size, self.output_dir_input.text()
        )
        self.fit23_window.show()
