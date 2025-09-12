from PyQt6 import QtWidgets, QtCore, QtGui
from .widgets import PlotWidget
from feda_tools.core import analysis as an
from feda_tools.core import data as dat
import numpy as np
import pyqtgraph as pg

class DashboardWidget(QtWidgets.QWidget):
    threshold_changed = QtCore.pyqtSignal(float)

    def __init__(self, state_handler):
        super().__init__()
        self.state_handler = state_handler
        self.setWindowTitle('FEDA Tools Dashboard')
        self.draggable_line = None  # Will hold the draggable threshold line
        self.init_ui()

    def init_ui(self):
        # Main grid layout for dashboard sections
        main_layout = QtWidgets.QGridLayout()
        main_layout.setSpacing(10)

        # =================================================================
        # TOP-LEFT: File Selection Group
        # =================================================================
        file_group = QtWidgets.QGroupBox("File Selection")
        file_layout = QtWidgets.QVBoxLayout()

        # File inputs with browse buttons
        self.file_ptu_input = QtWidgets.QLineEdit()
        self.file_irf_input = QtWidgets.QLineEdit()
        self.file_bkg_input = QtWidgets.QLineEdit()
        self.output_dir_input = QtWidgets.QLineEdit()

        file_ptu_button = QtWidgets.QPushButton('Browse')
        file_ptu_button.clicked.connect(self.select_ptu_file)
        file_irf_button = QtWidgets.QPushButton('Browse')
        file_irf_button.clicked.connect(self.select_irf_file)
        file_bkg_button = QtWidgets.QPushButton('Browse')
        file_bkg_button.clicked.connect(self.select_bkg_file)
        output_dir_button = QtWidgets.QPushButton('Browse')
        output_dir_button.clicked.connect(self.select_output_directory)

        # Create file input rows
        file_layout.addWidget(QtWidgets.QLabel('PTU File:'))
        file_layout.addLayout(self._create_file_row(self.file_ptu_input, file_ptu_button))
        
        file_layout.addWidget(QtWidgets.QLabel('IRF File:'))
        file_layout.addLayout(self._create_file_row(self.file_irf_input, file_irf_button))
        
        file_layout.addWidget(QtWidgets.QLabel('Background File:'))
        file_layout.addLayout(self._create_file_row(self.file_bkg_input, file_bkg_button))
        
        file_layout.addWidget(QtWidgets.QLabel('Output Directory:'))
        file_layout.addLayout(self._create_file_row(self.output_dir_input, output_dir_button))

        # Load data button in file selection
        self.load_data_button = QtWidgets.QPushButton('Load Data')
        self.load_data_button.clicked.connect(self.load_data)
        file_layout.addWidget(self.load_data_button)

        file_group.setLayout(file_layout)

        # =================================================================
        # BOTTOM-LEFT: Processing Controls Group
        # =================================================================
        processing_group = QtWidgets.QGroupBox("Processing Controls")
        processing_layout = QtWidgets.QVBoxLayout()

        # Chunk size input
        processing_layout.addWidget(QtWidgets.QLabel('Chunk Size:'))
        self.chunk_size_input = QtWidgets.QLineEdit('30000')
        processing_layout.addWidget(self.chunk_size_input)

        # Next button
        self.next_button = QtWidgets.QPushButton('Next →')
        self.next_button.setEnabled(False)
        processing_layout.addWidget(self.next_button)

        processing_group.setLayout(processing_layout)

        # =================================================================
        # TOP-RIGHT: Graph Group  
        # =================================================================
        graph_group = QtWidgets.QGroupBox("Threshold Analysis")
        graph_layout = QtWidgets.QVBoxLayout()

        # PyQtGraph Plot Widget
        self.plot_widget = PlotWidget()
        graph_layout.addWidget(self.plot_widget)

        graph_group.setLayout(graph_layout)

        # =================================================================
        # BOTTOM-RIGHT: Threshold Controls Group
        # =================================================================
        threshold_group = QtWidgets.QGroupBox("Threshold Control")
        threshold_layout = QtWidgets.QVBoxLayout()

        # Sigma controls
        sigma_layout = QtWidgets.QHBoxLayout()
        sigma_layout.addWidget(QtWidgets.QLabel('Sigma:'))
        
        self.sigma_spinbox = QtWidgets.QDoubleSpinBox()
        self.sigma_spinbox.setRange(0.1, 20.0)
        self.sigma_spinbox.setSingleStep(0.1)
        self.sigma_spinbox.setDecimals(1)
        self.sigma_spinbox.setValue(self.state_handler.config.get('threshold_multiplier_default', 4.0))
        self.sigma_spinbox.valueChanged.connect(self.on_sigma_changed)
        sigma_layout.addWidget(self.sigma_spinbox)

        # +/- buttons for fine adjustment
        minus_button = QtWidgets.QPushButton('−')
        minus_button.setMaximumWidth(30)
        minus_button.clicked.connect(lambda: self.adjust_sigma(-0.1))
        
        plus_button = QtWidgets.QPushButton('+')
        plus_button.setMaximumWidth(30)
        plus_button.clicked.connect(lambda: self.adjust_sigma(0.1))
        
        sigma_layout.addWidget(minus_button)
        sigma_layout.addWidget(plus_button)

        threshold_layout.addLayout(sigma_layout)

        # Threshold value display
        self.threshold_label = QtWidgets.QLabel(f"Threshold: {self.sigma_spinbox.value():.1f} σ")
        threshold_layout.addWidget(self.threshold_label)

        threshold_group.setLayout(threshold_layout)

        # =================================================================
        # Add all groups to main grid layout
        # =================================================================
        main_layout.addWidget(file_group, 0, 0)  # Top-left
        main_layout.addWidget(processing_group, 1, 0)  # Bottom-left
        main_layout.addWidget(graph_group, 0, 1)  # Top-right
        main_layout.addWidget(threshold_group, 1, 1)  # Bottom-right

        # Set column stretch to make graph area larger
        main_layout.setColumnStretch(0, 1)  # Left column
        main_layout.setColumnStretch(1, 2)  # Right column (graph area)

        self.setLayout(main_layout)

    def _create_file_row(self, line_edit, button):
        """Create a horizontal layout with line edit and button."""
        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(line_edit)
        layout.addWidget(button)
        return layout

    # =================================================================
    # File Selection Methods
    # =================================================================

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

    # =================================================================
    # Data Loading and Processing
    # =================================================================

    def load_data(self):
        file_ptu = self.file_ptu_input.text()
        file_irf = self.file_irf_input.text()
        file_bkg = self.file_bkg_input.text()
        chunk_size_text = self.chunk_size_input.text()

        if not all([file_ptu, file_irf, file_bkg, chunk_size_text]):
            QtWidgets.QMessageBox.warning(self, 'Error', 'Please select all files and specify chunk size.')
            return

        try:
            self.state_handler.chunk_size = int(chunk_size_text)
            if self.state_handler.chunk_size <= 0:
                raise ValueError
        except ValueError:
            QtWidgets.QMessageBox.warning(self, 'Error', 'Invalid chunk size. Please enter a positive integer.')
            return

        # Load data using core functions
        try:
            self.state_handler.data_ptu, self.state_handler.data_irf, self.state_handler.data_bkg = dat.load_ptu_files(file_ptu, file_irf, file_bkg)
            self.state_handler.file_ptu = file_ptu
            self.state_handler.file_irf = file_irf
            self.state_handler.file_bkg = file_bkg
            self.state_handler.output_directory = self.output_dir_input.text()
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, 'Error', f'Failed to load data files:\n{e}')
            return

        # Prepare data for threshold adjustment
        self.prepare_data()
        self.sigma_spinbox.setEnabled(True)
        self.next_button.setEnabled(True)
        self.plot_data(self.sigma_spinbox.value())

    def prepare_data(self):
        # Extract data
        all_macro_times = self.state_handler.data_ptu.macro_times
        all_micro_times = self.state_handler.data_ptu.micro_times

        # Get resolutions
        macro_res = self.state_handler.data_ptu.get_header().macro_time_resolution
        micro_res = self.state_handler.data_ptu.get_header().micro_time_resolution

        # Calculate interphoton arrival times
        photon_time_intervals, _ = an.interphoton_arrival_times(
            all_macro_times, all_micro_times, macro_res, micro_res
        )

        # Calculate running average
        window_size = self.state_handler.config.get('window_size', 30)
        running_avg = an.running_average(photon_time_intervals, window_size)
        self.logrunavg = np.log10(running_avg)

        # Estimate background noise
        bins_y = 141
        mu, std, _, _, _ = an.estimate_background_noise(self.logrunavg, bins_y)

        self.state_handler.mu = mu
        self.state_handler.std = std

    # =================================================================
    # Plotting and Threshold Control
    # =================================================================

    def plot_data(self, threshold_multiplier):
        threshold_value = self.state_handler.mu - threshold_multiplier * self.state_handler.std
        filtered_values = np.ma.masked_greater(self.logrunavg, threshold_value)

        self.plot_widget.clear()

        # Subsample for performance
        max_points = 25000
        total_points = len(self.logrunavg)
        step = max(1, total_points // max_points)
        subset_indices = slice(0, total_points, step)

        # Plot data points
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

        # Add draggable threshold line
        if self.draggable_line is not None:
            self.plot_widget.removeItem(self.draggable_line)
        
        self.draggable_line = pg.InfiniteLine(
            pos=threshold_value,
            angle=0,  # Horizontal line
            movable=True,
            pen=pg.mkPen('r', style=QtCore.Qt.PenStyle.DashLine, width=2)
        )
        self.draggable_line.sigPositionChanged.connect(self.on_line_moved)
        self.plot_widget.addItem(self.draggable_line)

        # Set labels and legend
        self.plot_widget.setLabel('bottom', 'Photon Event #')
        self.plot_widget.setLabel('left', 'log(Photon Interval Time)')
        self.plot_widget.addLegend()

    def on_sigma_changed(self, value):
        """Handle changes from the numerical sigma input."""
        self.threshold_label.setText(f"Threshold: {value:.1f} σ")
        self.plot_data(value)
        self.threshold_changed.emit(value)
        self.state_handler.threshold_multiplier = value

    def adjust_sigma(self, delta):
        """Adjust sigma by delta amount using +/- buttons."""
        new_value = self.sigma_spinbox.value() + delta
        self.sigma_spinbox.setValue(max(0.1, min(20.0, new_value)))

    def on_line_moved(self):
        """Handle draggable line movement - update sigma value."""
        if self.draggable_line is None or not hasattr(self, 'state_handler'):
            return
        
        threshold_value = self.draggable_line.value()
        
        # Calculate corresponding sigma from threshold position
        # threshold_value = mu - sigma * std
        # sigma = (mu - threshold_value) / std
        if self.state_handler.std != 0:
            sigma = (self.state_handler.mu - threshold_value) / self.state_handler.std
            sigma = max(0.1, min(20.0, sigma))  # Clamp to reasonable range
            
            # Update the spinbox without triggering its signal
            self.sigma_spinbox.blockSignals(True)
            self.sigma_spinbox.setValue(sigma)
            self.sigma_spinbox.blockSignals(False)
            
            # Update label and emit signal
            self.threshold_label.setText(f"Threshold: {sigma:.1f} σ")
            self.threshold_changed.emit(sigma)
            self.state_handler.threshold_multiplier = sigma