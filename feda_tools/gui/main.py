import sys
from PyQt6 import QtWidgets, QtCore
from feda_tools.gui.threshold_adjustment import ThresholdAdjustmentWindow
from feda_tools.gui.fit23_preview import Fit23PreviewWindow
from feda_tools.core import utilities as utils
from feda_tools.core import analysis as an
from feda_tools.core import data as dat
from feda_tools.core import model
from feda_tools.core import process
import pathlib
import numpy as np

class MainApplication(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('FEDA Tools GUI')
        self.threshold_multiplier = 4  # Default threshold
        self.init_ui()

    def init_ui(self):
        # Central widget
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        layout = QtWidgets.QVBoxLayout()

        # File selection
        self.file_ptu_input = QtWidgets.QLineEdit()
        self.file_irf_input = QtWidgets.QLineEdit()
        self.file_bkg_input = QtWidgets.QLineEdit()

        file_ptu_button = QtWidgets.QPushButton('Select PTU File')
        file_ptu_button.clicked.connect(self.select_ptu_file)

        file_irf_button = QtWidgets.QPushButton('Select IRF File')
        file_irf_button.clicked.connect(self.select_irf_file)

        file_bkg_button = QtWidgets.QPushButton('Select Background File')
        file_bkg_button.clicked.connect(self.select_bkg_file)

        layout.addWidget(QtWidgets.QLabel('Main PTU File:'))
        layout.addWidget(self.file_ptu_input)
        layout.addWidget(file_ptu_button)

        layout.addWidget(QtWidgets.QLabel('IRF File:'))
        layout.addWidget(self.file_irf_input)
        layout.addWidget(file_irf_button)

        layout.addWidget(QtWidgets.QLabel('Background File:'))
        layout.addWidget(self.file_bkg_input)
        layout.addWidget(file_bkg_button)

        # Next button
        next_button = QtWidgets.QPushButton('Next')
        next_button.clicked.connect(self.load_data)
        layout.addWidget(next_button)

        central_widget.setLayout(layout)

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

    def load_data(self):
        file_ptu = self.file_ptu_input.text()
        file_irf = self.file_irf_input.text()
        file_bkg = self.file_bkg_input.text()

        if not all([file_ptu, file_irf, file_bkg]):
            QtWidgets.QMessageBox.warning(self, 'Error', 'Please select all files.')
            return

        # Load data using core functions
        self.data_ptu, self.data_irf, self.data_bkg = dat.load_ptu_files(file_ptu, file_irf, file_bkg)

        # Proceed to threshold adjustment
        self.setup_threshold_adjustment()

    def setup_threshold_adjustment(self):
        # Extract data
        all_macro_times = self.data_ptu.macro_times
        all_micro_times = self.data_ptu.micro_times
        routing_channels = self.data_ptu.routing_channels

        # Get resolutions
        macro_res = self.data_ptu.get_header().macro_time_resolution
        micro_res = self.data_ptu.get_header().micro_time_resolution

        # Calculate interphoton arrival times
        photon_time_intervals, photon_ids = an.interphoton_arrival_times(
            all_macro_times, all_micro_times, macro_res, micro_res
        )

        # Calculate running average
        window_size = 30
        running_avg = an.running_average(photon_time_intervals, window_size)
        xarr = np.arange(window_size - 1, len(photon_time_intervals))
        self.logrunavg = np.log10(running_avg)

        # Estimate background noise
        bins = {"x": 141, "y": 141}
        bins_y = bins['y']
        self.mu, self.std, noise_mean, filtered_logrunavg, bins_logrunavg = an.estimate_background_noise(self.logrunavg, bins_y)

        # Open threshold adjustment window
        self.threshold_window = ThresholdAdjustmentWindow(
            self.logrunavg, self.mu, self.std, self.threshold_multiplier, bins_y
        )
        self.threshold_window.threshold_changed.connect(self.update_threshold)
        self.threshold_window.next_button.clicked.connect(self.setup_fit23_preview)
        self.threshold_window.show()

    def update_threshold(self, value):
        self.threshold_multiplier = value

    def setup_fit23_preview(self):
        # Extract bursts based on updated threshold
        threshold_value = self.mu - self.threshold_multiplier * self.std
        burst_index, filtered_values = dat.extract_greater(self.logrunavg, threshold_value)

        # Setup fit23 (using first burst as sample)
        counts_irf, _ = np.histogram(self.data_irf.micro_times, bins=128)
        counts_irf_nb = counts_irf.copy()
        counts_irf_nb[0:3] = 0
        counts_irf_nb[10:66] = 0
        counts_irf_nb[74:128] = 0
        fit23 = model.setup_fit23(
            num_bins=128,
            macro_res=self.data_ptu.get_header().macro_time_resolution,
            counts_irf_nb=counts_irf_nb,
            g_factor=1.04,
            l1_japan_corr=0.0308,
            l2_japan_corr=0.0368
        )

        # Open fit23 preview window
        self.fit23_window = Fit23PreviewWindow(burst_index[0], fit23)
        self.fit23_window.analysis_started.connect(self.run_full_analysis)
        self.fit23_window.show()

    def run_full_analysis(self):
        # Implement the code to run the full analysis
        # You can use a QThread to prevent freezing the GUI
        QtWidgets.QMessageBox.information(self, 'Analysis', 'Full analysis started.')
        # TODO: Implement full analysis logic

def main():
    app = QtWidgets.QApplication(sys.argv)
    main_window = MainApplication()
    main_window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()