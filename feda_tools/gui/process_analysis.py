from PyQt6 import QtWidgets, QtCore
from PyQt6.QtCore import QThread, pyqtSignal
from .widgets import PlotWidget
from feda_tools.core import process as proc
from feda_tools.core import analysis as an
from feda_tools.core import data as dat
from feda_tools.core import model
import numpy as np
import pandas as pd

class Worker(QtCore.QObject):
    progress_update = pyqtSignal(int)
    data_ready = pyqtSignal(list, list, list, list)
    finished = pyqtSignal()
    error_occurred = pyqtSignal(str)

    def __init__(self, state_handler, update_interval):
        super().__init__()
        self.state_handler = state_handler
        self.update_interval = update_interval

    @QtCore.pyqtSlot()
    def run(self):
        try:
            data_ptu = self.state_handler.data_ptu
            data_irf = self.state_handler.data_irf
            burst_index = self.state_handler.burst_index
            fit_params = self.state_handler.fit_params

            min_photon_count = self.state_handler.config.get('min_photon_count', 60)
            bg4_micro_time_min = self.state_handler.config.get('bg4_micro_time_min', 0)
            bg4_micro_time_max = self.state_handler.config.get('bg4_micro_time_max', 12499)
            g_factor = self.state_handler.config.get('g_factor', 1.04)
            l1_japan_corr = self.state_handler.config.get('l1_japan_corr', 0.0308)
            l2_japan_corr = self.state_handler.config.get('l2_japan_corr', 0.0368)
            bg4_bkg_para = self.state_handler.config.get('bg4_bkg_para', 0)
            bg4_bkg_perp = self.state_handler.config.get('bg4_bkg_perp', 0)
            num_bins = self.state_handler.config.get('num_bins', 128)

            counts_irf, _ = np.histogram(data_irf.micro_times, bins=num_bins)
            counts_irf_nb = counts_irf.copy()
            counts_irf_nb[0:3] = 0
            counts_irf_nb[10:66] = 0
            counts_irf_nb[74:128] = 0
            macro_res = data_ptu.get_header().macro_time_resolution
            fit23_model = model.setup_fit23(
                num_bins, macro_res, counts_irf_nb, g_factor, l1_japan_corr, l2_japan_corr
            )

            sg_sr = []
            mean_macro_time = []
            tau_values = []
            r_s_values = []
            total_bursts = len(burst_index)
            all_bi4_bur_df = []
            all_bg4_df = []

            for i, burst in enumerate(burst_index, 1):
                if len(burst) < min_photon_count:
                    continue

                bi4_bur_df, bg4_df = proc.process_single_burst(
                    burst,
                    data_ptu.macro_times,
                    data_ptu.micro_times,
                    data_ptu.routing_channels,
                    macro_res,
                    data_ptu.get_header().micro_time_resolution,
                    min_photon_count,
                    bg4_micro_time_min,
                    bg4_micro_time_max,
                    g_factor,
                    l1_japan_corr,
                    l2_japan_corr,
                    bg4_bkg_para,
                    bg4_bkg_perp,
                    fit23_model,
                    fit_params
                )

                if bi4_bur_df is not None and bg4_df is not None:
                    sg_sr_value = bg4_df['Ng-p-all'].values[0] / bg4_df['Ng-s-all'].values[0]
                    tau_value = bg4_df['Fit tau'].values[0]
                    r_s_value = bg4_df['Fit rs_scatter'].values[0]
                    mean_macro = bi4_bur_df['Mean Macro Time (ms)'].values[0]

                    sg_sr.append(sg_sr_value)
                    mean_macro_time.append(mean_macro)
                    tau_values.append(tau_value)
                    r_s_values.append(r_s_value)

                    all_bi4_bur_df.append(bi4_bur_df)
                    all_bg4_df.append(bg4_df)

                if i % self.update_interval == 0 or i == total_bursts:
                    self.progress_update.emit(int(i / total_bursts * 100))

            self.data_ready.emit(sg_sr, mean_macro_time, tau_values, r_s_values)

            bi4_bur_df_combined = pd.concat(all_bi4_bur_df, ignore_index=True)
            bg4_df_combined = pd.concat(all_bg4_df, ignore_index=True)

            output_directory = self.state_handler.output_directory
            file_path = self.state_handler.file_ptu
            dat.save_results(output_directory, file_path, bi4_bur_df_combined, bg4_df_combined)

            self.finished.emit()
        except Exception as e:
            self.error_occurred.emit(str(e))
            self.finished.emit()

class ProcessAnalysisWidget(QtWidgets.QWidget):
    def __init__(self, state_handler):
        super().__init__()
        self.state_handler = state_handler
        self.init_ui()
        self.thread = None
        self.worker = None

    def init_ui(self):
        self.setWindowTitle('Full Analysis')
        layout = QtWidgets.QVBoxLayout()

        # Progress bar
        self.progress_bar = QtWidgets.QProgressBar()
        layout.addWidget(self.progress_bar)

        # Update interval input
        interval_layout = QtWidgets.QHBoxLayout()
        interval_label = QtWidgets.QLabel('Update Interval (bursts):')
        self.interval_input = QtWidgets.QSpinBox()
        self.interval_input.setRange(1, 1000)
        self.interval_input.setValue(10)
        interval_layout.addWidget(interval_label)
        interval_layout.addWidget(self.interval_input)
        layout.addLayout(interval_layout)

        # Start Analysis Button
        self.start_button = QtWidgets.QPushButton('Start Full Analysis')
        self.start_button.clicked.connect(self.run_full_analysis)
        layout.addWidget(self.start_button)

        # Plot widgets
        plot_layout = QtWidgets.QHBoxLayout()
        self.plot_widget1 = PlotWidget()
        self.plot_widget2 = PlotWidget()
        self.plot_widget3 = PlotWidget()

        plot_layout.addWidget(self.plot_widget1)
        plot_layout.addWidget(self.plot_widget2)
        plot_layout.addWidget(self.plot_widget3)
        layout.addLayout(plot_layout)

        self.setLayout(layout)

    def run_full_analysis(self):
        if self.worker and self.thread.isRunning():
            QtWidgets.QMessageBox.warning(self, 'Processing', 'Analysis is already running.')
            return

        self.start_button.setEnabled(False)
        self.update_interval = self.interval_input.value()

        self.prepare_data()

        self.thread = QThread()
        self.worker = Worker(
            self.state_handler,
            self.update_interval
        )
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.progress_update.connect(self.update_progress)
        self.worker.data_ready.connect(self.update_plots)
        self.worker.finished.connect(self.analysis_finished)
        self.worker.error_occurred.connect(self.handle_error)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()

    def prepare_data(self):
        all_macro_times = self.state_handler.data_ptu.macro_times
        all_micro_times = self.state_handler.data_ptu.micro_times

        macro_res = self.state_handler.data_ptu.get_header().macro_time_resolution
        micro_res = self.state_handler.data_ptu.get_header().micro_time_resolution

        photon_time_intervals, _ = an.interphoton_arrival_times(
            all_macro_times, all_micro_times, macro_res, micro_res
        )

        window_size = self.state_handler.config.get('window_size', 30)
        running_avg = an.running_average(photon_time_intervals, window_size)
        logrunavg = np.log10(running_avg)

        threshold_multiplier = self.state_handler.threshold_multiplier or self.state_handler.config.get('threshold_multiplier_default', 4)
        threshold_value = self.state_handler.mu - threshold_multiplier * self.state_handler.std

        burst_index, _ = dat.extract_greater(logrunavg, threshold_value)
        self.state_handler.burst_index = burst_index

    @QtCore.pyqtSlot(int)
    def update_progress(self, value):
        self.progress_bar.setValue(value)

    @QtCore.pyqtSlot(list, list, list, list)
    def update_plots(self, sg_sr, mean_macro_time, tau_values, r_s_values):
        self.plot_widget1.clear()
        self.plot_widget1.plot(mean_macro_time, sg_sr, pen=None, symbol='o', symbolSize=5)
        self.plot_widget1.setLabel('bottom', 'Mean Macro Time (ms)')
        self.plot_widget1.setLabel('left', 'sg/sr')
        self.plot_widget1.setTitle('sg/sr vs Mean Macro Time')
        self.plot_widget1.showGrid(x=True, y=True)

        self.plot_widget2.clear()
        self.plot_widget2.plot(tau_values, sg_sr, pen=None, symbol='o', symbolSize=5, brush='g')
        self.plot_widget2.setLabel('bottom', 'Tau')
        self.plot_widget2.setLabel('left', 'sg/sr')
        self.plot_widget2.setTitle('sg/sr vs Tau')
        self.plot_widget2.showGrid(x=True, y=True)

        self.plot_widget3.clear()
        self.plot_widget3.plot(tau_values, r_s_values, pen=None, symbol='o', symbolSize=5, brush='r')
        self.plot_widget3.setLabel('bottom', 'Tau')
        self.plot_widget3.setLabel('left', 'r_s')
        self.plot_widget3.setTitle('r_s vs Tau')
        self.plot_widget3.showGrid(x=True, y=True)

    @QtCore.pyqtSlot()
    def analysis_finished(self):
        self.start_button.setEnabled(True)
        QtWidgets.QMessageBox.information(self, 'Success', 'Full analysis completed successfully.')

    @QtCore.pyqtSlot(str)
    def handle_error(self, message):
        QtWidgets.QMessageBox.warning(self, 'Error', f'An error occurred during analysis:\n{message}')
        self.start_button.setEnabled(True)
