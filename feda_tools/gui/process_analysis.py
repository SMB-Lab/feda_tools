from PyQt6 import QtWidgets, QtCore
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from .widgets import MatplotlibCanvas
from feda_tools.core import process as proc
from feda_tools.core import data as dat
from feda_tools.core import model
import numpy as np
import pandas as pd

class Worker(QtCore.QObject):
    progress_update = pyqtSignal(int)
    plot_update = pyqtSignal()
    finished = pyqtSignal()
    error_occurred = pyqtSignal(str)

    def __init__(self, data_ptu, data_irf, burst_index, chunk_size, fit_params, update_interval):
        super().__init__()
        self.data_ptu = data_ptu
        self.data_irf = data_irf
        self.burst_index = burst_index
        self.chunk_size = chunk_size
        self.fit_params = fit_params
        self.update_interval = update_interval
        self.sg_sr = []
        self.mean_macro_time = []
        self.tau_values = []
        self.r_s_values = []
        self.is_running = True
        self.bi4_bur_dfs = []  # List to hold bi4_bur_df for each burst
        self.bg4_dfs = []      # List to hold bg4_df for each burst

    @QtCore.pyqtSlot()
    def run(self):
        try:
            # Example parameters, adjust as needed
            min_photon_count = 60
            bg4_micro_time_min = 0
            bg4_micro_time_max = 12499
            g_factor = 1.04
            l1_japan_corr = 0.0308
            l2_japan_corr = 0.0368
            bg4_bkg_para = 0
            bg4_bkg_perp = 0
            num_bins = 128

            # Prepare data
            total_bursts = len(self.burst_index)
            self.progress_update.emit(0)

            # Initialize fit23 model
            counts_irf, _ = np.histogram(self.data_irf.micro_times, bins=num_bins)
            counts_irf_nb = counts_irf.copy()
            counts_irf_nb[0:3] = 0
            counts_irf_nb[10:66] = 0
            counts_irf_nb[74:128] = 0
            macro_res = self.data_ptu.get_header().macro_time_resolution
            fit23_model = model.setup_fit23(
                num_bins, macro_res, counts_irf_nb, g_factor, l1_japan_corr, l2_japan_corr
            )

            # Processing bursts
            for i, burst in enumerate(self.burst_index, 1):
                if not self.is_running:
                    break  # Allow for future extension to stop processing

                # Process each burst individually
                bi4_bur_df, bg4_df = proc.process_single_burst(
                    burst,
                    self.data_ptu.macro_times,
                    self.data_ptu.micro_times,
                    self.data_ptu.routing_channels,
                    macro_res,
                    self.data_ptu.get_header().micro_time_resolution,
                    min_photon_count,
                    bg4_micro_time_min,
                    bg4_micro_time_max,
                    g_factor,
                    l1_japan_corr,
                    l2_japan_corr,
                    bg4_bkg_para,
                    bg4_bkg_perp,
                    fit23_model,
                    self.fit_params
                )

                if bi4_bur_df is not None and bg4_df is not None:
                    # Collect dataframes
                    self.bi4_bur_dfs.append(bi4_bur_df)
                    self.bg4_dfs.append(bg4_df)
                    try:
                        sg_sr_value = bg4_df['Ng-p-all'].values[0] / bg4_df['Ng-s-all'].values[0]
                        tau_value = bg4_df['Fit tau'].values[0]
                        r_s_value = bg4_df['Fit rs_scatter'].values[0]
                        mean_macro = bi4_bur_df['Mean Macro Time (ms)'].values[0]

                        self.sg_sr.append(sg_sr_value)
                        self.mean_macro_time.append(mean_macro)
                        self.tau_values.append(tau_value)
                        self.r_s_values.append(r_s_value)
                    except Exception as e:
                        self.error_occurred.emit(f"Error extracting plot data for burst {i}: {e}")

                # Update progress bar
                self.progress_update.emit(i)

                # Update plots every n bursts
                if i % self.update_interval == 0:
                    self.plot_update.emit()

            # Final plot update
            self.plot_update.emit()
            self.finished.emit()

        except Exception as e:
            self.error_occurred.emit(str(e))
            self.finished.emit()

    def stop(self):
        self.is_running = False

class ProcessAnalysisWindow(QtWidgets.QWidget):
    def __init__(self, data_ptu, data_irf, burst_index, chunk_size, fit_params, output_directory):
        super().__init__()
        self.data_ptu = data_ptu
        self.data_irf = data_irf
        self.burst_index = burst_index
        self.chunk_size = chunk_size
        self.fit_params = fit_params
        self.output_directory = output_directory
        self.file_ptu = "output.ptu" # Change this later to the actual filename for now it gets a placeholder file_ptu  # Store the PTU file path

        self.init_ui()
        self.thread = None
        self.worker = None

        # Data containers for plots
        self.sg_sr = []
        self.mean_macro_time = []
        self.tau_values = []
        self.r_s_values = []

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
        self.interval_input.setValue(10)  # Default update interval
        interval_layout.addWidget(interval_label)
        interval_layout.addWidget(self.interval_input)
        layout.addLayout(interval_layout)

        # Start Analysis Button
        self.start_button = QtWidgets.QPushButton('Start Full Analysis')
        self.start_button.clicked.connect(self.run_full_analysis)
        layout.addWidget(self.start_button)

        # Plot area
        plot_layout = QtWidgets.QHBoxLayout()
        self.canvas1 = MatplotlibCanvas(self, width=5, height=4, dpi=100)
        self.canvas2 = MatplotlibCanvas(self, width=5, height=4, dpi=100)
        self.canvas3 = MatplotlibCanvas(self, width=5, height=4, dpi=100)

        plot_layout.addWidget(self.canvas1)
        plot_layout.addWidget(self.canvas2)
        plot_layout.addWidget(self.canvas3)

        layout.addLayout(plot_layout)

        self.setLayout(layout)

    def run_full_analysis(self):
        if self.worker and self.thread.isRunning():
            QtWidgets.QMessageBox.warning(self, 'Processing', 'Analysis is already running.')
            return

        self.start_button.setEnabled(False)
        self.update_interval = self.interval_input.value()

        # Initialize Worker and Thread
        self.thread = QThread()
        self.worker = Worker(
            self.data_ptu,
            self.data_irf,
            self.burst_index,
            self.chunk_size,
            self.fit_params,
            self.update_interval
        )
        self.worker.moveToThread(self.thread)

        # Connect signals and slots
        self.thread.started.connect(self.worker.run)
        self.worker.progress_update.connect(self.update_progress)
        self.worker.plot_update.connect(self.update_plots)
        self.worker.finished.connect(self.analysis_finished)
        self.worker.error_occurred.connect(self.handle_error)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        # Start the thread
        self.thread.start()

    @QtCore.pyqtSlot(int)
    def update_progress(self, value):
        self.progress_bar.setValue(value)

    @QtCore.pyqtSlot()
    def update_plots(self):
        # sg/sr vs Mean Macro Time
        self.canvas1.axes.clear()
        self.canvas1.axes.scatter(self.worker.mean_macro_time, self.worker.sg_sr, s=10, alpha=0.7)
        self.canvas1.axes.set_xlabel('Mean Macro Time (ms)')
        self.canvas1.axes.set_ylabel('sg/sr')
        self.canvas1.axes.set_title('sg/sr vs Mean Macro Time')
        self.canvas1.axes.grid(True)
        self.canvas1.draw()

        # sg/sr vs Tau
        self.canvas2.axes.clear()
        self.canvas2.axes.scatter(self.worker.tau_values, self.worker.sg_sr, s=10, alpha=0.7, color='green')
        self.canvas2.axes.set_xlabel('Tau')
        self.canvas2.axes.set_ylabel('sg/sr')
        self.canvas2.axes.set_title('sg/sr vs Tau')
        self.canvas2.axes.grid(True)
        self.canvas2.draw()

        # r_s vs Tau
        self.canvas3.axes.clear()
        self.canvas3.axes.scatter(self.worker.tau_values, self.worker.r_s_values, s=10, alpha=0.7, color='red')
        self.canvas3.axes.set_xlabel('Tau')
        self.canvas3.axes.set_ylabel('r_s')
        self.canvas3.axes.set_title('r_s vs Tau')
        self.canvas3.axes.grid(True)
        self.canvas3.draw()

    @QtCore.pyqtSlot()
    def analysis_finished(self):
        self.start_button.setEnabled(True)
        # Combine dataframes
        bi4_bur_df_combined = pd.concat(self.worker.bi4_bur_dfs, ignore_index=True)
        bg4_df_combined = pd.concat(self.worker.bg4_dfs, ignore_index=True)
        # Save results
        dat.save_results(self.output_directory, self.file_ptu, bi4_bur_df_combined, bg4_df_combined)
        QtWidgets.QMessageBox.information(self, 'Success', 'Full analysis completed successfully.')

    @QtCore.pyqtSlot(str)
    def handle_error(self, message):
        QtWidgets.QMessageBox.warning(self, 'Error', f'An error occurred during analysis:\n{message}')
        self.start_button.setEnabled(True)
