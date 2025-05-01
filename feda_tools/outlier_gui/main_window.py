import sys
import os
import glob
import numpy as np
import pandas as pd
import pyqtgraph as pg
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLineEdit, QLabel, QFileDialog, QMessageBox,
    QDoubleSpinBox, QFormLayout, QStatusBar
)
from PyQt6.QtCore import Qt, QThread, pyqtSlot

# Import the worker from the sibling module
try:
    from .worker import Worker
except ImportError:
    from worker import Worker

# Assuming the core module is importable
try:
    from ..core.outlier_detection import (
        load_burst_data,
        detect_time_difference_outliers,
        # detect_linear_projection_outliers, # Not needed directly in main window anymore
        detect_linear_residual_outliers,
        calculate_sg_sr
    )
except ImportError:
    import sys
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(os.path.dirname(script_dir))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    try:
        from feda_tools.core.outlier_detection import (
            load_burst_data,
            detect_time_difference_outliers,
            # detect_linear_projection_outliers,
            detect_linear_residual_outliers,
            calculate_sg_sr
        )
    except ImportError as e:
        print(f"Error importing outlier_detection: {e}. Using dummy functions.")
        # Dummy functions
        def load_burst_data(*args, **kwargs): return (pd.DataFrame({'time_diff': [], 'Duration (ms)': [], 'Number of Photons': [], 'linear_resid': [], 'Mean Macro Time (ms)': []}), pd.DataFrame(), pd.DataFrame())
        def detect_time_difference_outliers(*args, **kwargs): return pd.Series(dtype=bool), np.nan, np.nan
        # def detect_linear_projection_outliers(*args, **kwargs): return pd.Series(dtype=bool)
        def detect_linear_residual_outliers(*args, **kwargs): return pd.Series(dtype=bool), np.nan, np.nan, np.nan
        def calculate_sg_sr(*args, **kwargs): return pd.Series(dtype=float)


# Configure pyqtgraph
pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')


class OutlierMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FEDA Tools - Outlier Detection")
        self.setGeometry(100, 100, 1400, 850)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # Data storage
        self.preview_bur_df = None
        self.current_processing_bur_df = None
        self.input_dir = None
        self.output_dir = None

        # State
        self.is_running = False
        self.is_paused = False

        # Worker thread
        self.thread = None
        self.worker = None

        self._create_widgets()
        self._create_layouts()
        self._initialize_plots() # Initialize plots first
        self._connect_signals() # Then connect signals

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready. Select Input Directory to load preview.")

    def _create_widgets(self):
        # Input Directory
        self.input_dir_label = QLabel("Input Directory:")
        self.input_dir_edit = QLineEdit()
        self.input_dir_edit.setReadOnly(True)
        self.input_dir_button = QPushButton("Browse...")

        # Output Directory
        self.output_dir_label = QLabel("Output Directory:")
        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setReadOnly(True)
        self.output_dir_button = QPushButton("Browse...")

        # Parameters
        self.time_cutoff_spin = QDoubleSpinBox()
        self.time_cutoff_spin.setRange(0.0, 100.0)
        self.time_cutoff_spin.setValue(1.0)
        self.time_cutoff_spin.setDecimals(2)
        self.time_cutoff_spin.setSingleStep(0.1)
        self.time_cutoff_spin.setToolTip("Z-score cutoff for time difference.")

        self.resid_cutoff_spin = QDoubleSpinBox()
        self.resid_cutoff_spin.setRange(0.0, 100.0)
        self.resid_cutoff_spin.setValue(3.0)
        self.resid_cutoff_spin.setDecimals(2)
        self.resid_cutoff_spin.setSingleStep(0.1)
        self.resid_cutoff_spin.setToolTip("Z-score cutoff for linear fit residuals (Duration vs Photons).")

        self.proj_cutoff_spin = QDoubleSpinBox()
        self.proj_cutoff_spin.setRange(0.0, 100.0)
        self.proj_cutoff_spin.setValue(3.0)
        self.proj_cutoff_spin.setDecimals(2)
        self.proj_cutoff_spin.setSingleStep(0.1)
        self.proj_cutoff_spin.setToolTip("Z-score cutoff for linear projection distance (n_sum_g vs n_sum_r). Used in filtering, not plotted.")

        # Run/Pause/Resume Button
        self.run_pause_button = QPushButton("Run Processing")

        # Plot Widgets
        self.time_diff_hist_widget = pg.PlotWidget(name="TimeDiffHist")
        self.linear_fit_scatter_widget = pg.PlotWidget(name="LinearFitScatter")
        self.sg_sr_scatter_widget = pg.PlotWidget(name="SgSrScatter")


    def _create_layouts(self):
        top_controls_layout = QHBoxLayout()
        dir_select_layout = QGridLayout()
        dir_select_layout.addWidget(self.input_dir_label, 0, 0)
        dir_select_layout.addWidget(self.input_dir_edit, 0, 1)
        dir_select_layout.addWidget(self.input_dir_button, 0, 2)
        dir_select_layout.addWidget(self.output_dir_label, 1, 0)
        dir_select_layout.addWidget(self.output_dir_edit, 1, 1)
        dir_select_layout.addWidget(self.output_dir_button, 1, 2)
        top_controls_layout.addLayout(dir_select_layout, 2)

        param_layout = QFormLayout()
        param_layout.addRow("Time Diff Z-Cutoff:", self.time_cutoff_spin)
        param_layout.addRow("Residual Z-Cutoff:", self.resid_cutoff_spin)
        param_layout.addRow("Projection Z-Cutoff (Filtering Only):", self.proj_cutoff_spin)
        top_controls_layout.addLayout(param_layout, 1)

        run_button_layout = QVBoxLayout()
        run_button_layout.addWidget(self.run_pause_button)
        run_button_layout.addStretch()
        top_controls_layout.addLayout(run_button_layout)

        plots_layout = QGridLayout()
        plots_layout.addWidget(self.time_diff_hist_widget, 0, 0)
        plots_layout.addWidget(self.linear_fit_scatter_widget, 0, 1)
        plots_layout.addWidget(self.sg_sr_scatter_widget, 1, 0, 1, 2)

        self.layout.addLayout(top_controls_layout)
        self.layout.addLayout(plots_layout)
        self.layout.setStretchFactor(plots_layout, 1)

    def _initialize_plots(self):
        # Plot 1: Time Difference Histogram
        self.time_diff_hist_widget.setLabel('left', 'Frequency')
        self.time_diff_hist_widget.setLabel('bottom', 'Time Difference (Duration Green - Yellow)')
        self.time_diff_hist_widget.setTitle('Time Difference Distribution (Preview)')
        self.time_diff_hist_item = pg.BarGraphItem(x=[], height=[], width=1, brush='b', name='Frequency')
        self.time_diff_hist_widget.addItem(self.time_diff_hist_item)
        self.time_diff_cutoff_line_upper = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('r', style=Qt.PenStyle.DashLine))
        self.time_diff_cutoff_line_lower = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('r', style=Qt.PenStyle.DashLine))
        self.time_diff_hist_widget.addItem(self.time_diff_cutoff_line_upper)
        self.time_diff_hist_widget.addItem(self.time_diff_cutoff_line_lower)

        # Plot 2: Linear Fit (Duration vs Photons) Scatter + Bands
        self.linear_fit_scatter_widget.setLabel('left', 'Number of Photons')
        self.linear_fit_scatter_widget.setLabel('bottom', 'Duration (ms)')
        self.linear_fit_scatter_widget.setTitle('Linear Fit: Duration vs Photons (Preview)')
        self.linear_fit_scatter_all = pg.ScatterPlotItem(pen=pg.mkPen(None), brush=pg.mkBrush(0, 0, 200, 150), size=5, name='Included')
        self.linear_fit_scatter_outliers = pg.ScatterPlotItem(pen=pg.mkPen('r'), brush=pg.mkBrush(255, 0, 0, 200), size=7, symbol='x', name='Residual Outlier')
        self.linear_fit_line = pg.PlotCurveItem(pen=pg.mkPen('g', width=2), name='Linear Fit')
        self.linear_fit_band_upper = pg.PlotCurveItem(pen=pg.mkPen('k', style=Qt.PenStyle.DashLine), name='+Z Residual Cutoff')
        self.linear_fit_band_lower = pg.PlotCurveItem(pen=pg.mkPen('k', style=Qt.PenStyle.DashLine), name='-Z Residual Cutoff')
        self.linear_fit_scatter_widget.addItem(self.linear_fit_scatter_all)
        self.linear_fit_scatter_widget.addItem(self.linear_fit_scatter_outliers)
        self.linear_fit_scatter_widget.addItem(self.linear_fit_line)
        self.linear_fit_scatter_widget.addItem(self.linear_fit_band_upper)
        self.linear_fit_scatter_widget.addItem(self.linear_fit_band_lower)
        self.linear_fit_scatter_widget.addLegend()

        # Plot 3: Sg/Sr vs Macro Time Scatter
        self.sg_sr_scatter_widget.setLabel('left', 'Sg/Sr')
        self.sg_sr_scatter_widget.setLabel('bottom', 'Mean Macro Time (ms)')
        self.sg_sr_scatter_widget.setTitle('Sg/Sr vs Macro Time (Preview)')
        self.sg_sr_scatter_widget.setLogMode(y=True)
        self.sg_sr_scatter_included = pg.ScatterPlotItem(pen=pg.mkPen(None), brush=pg.mkBrush(128, 0, 128, 150), size=5, name='Included (Time|Resid Filter)') # Purple
        self.sg_sr_scatter_outliers = pg.ScatterPlotItem(pen=pg.mkPen('r'), brush=pg.mkBrush(255, 0, 0, 200), size=7, symbol='o', name='Outlier (Time|Resid Filter)') # Red circle
        self.sg_sr_scatter_widget.addItem(self.sg_sr_scatter_included)
        self.sg_sr_scatter_widget.addItem(self.sg_sr_scatter_outliers)
        self.sg_sr_scatter_widget.addLegend()


    def _connect_signals(self):
        self.input_dir_button.clicked.connect(self.select_input_dir)
        self.output_dir_button.clicked.connect(self.select_output_dir)
        self.run_pause_button.clicked.connect(self.handle_run_pause_resume)
        self.time_cutoff_spin.valueChanged.connect(lambda: self.update_preview_plots() if not self.is_running else None)
        self.resid_cutoff_spin.valueChanged.connect(lambda: self.update_preview_plots() if not self.is_running else None)

    # --- Directory Selection & Preview Loading ---
    def select_input_dir(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Input Directory")
        if directory:
            self.input_dir = directory
            self.input_dir_edit.setText(directory)
            self.status_bar.showMessage(f"Input directory set. Loading preview...")
            QApplication.processEvents()
            self.load_preview_data()

    def select_output_dir(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if directory:
            self.output_dir = directory
            self.output_dir_edit.setText(directory)
            self.status_bar.showMessage(f"Output directory set: {directory}")

    def load_preview_data(self):
        self.preview_bur_df = None
        self.clear_all_plots("Preview")
        if not self.input_dir:
            self.status_bar.showMessage("Select an input directory first.")
            return

        try:
            search_pattern = os.path.join(self.input_dir, '**', 'bi4_bur', '*.bur')
            preview_files = sorted(glob.glob(search_pattern, recursive=True))
            if not preview_files:
                self.log_error(f"No .bur files found for preview in {self.input_dir}")
                return

            first_file = preview_files[0]
            base_path = os.path.dirname(os.path.dirname(first_file))
            split_name = os.path.splitext(os.path.basename(first_file))[0]

            self.status_bar.showMessage(f"Loading preview: {os.path.basename(first_file)}...")
            QApplication.processEvents()

            self.preview_bur_df, _, _ = load_burst_data(base_path, split_name)

            if self.preview_bur_df is None or self.preview_bur_df.empty:
                self.log_error(f"Failed to load or empty preview data for {split_name}")
                self.preview_bur_df = None
                return

            self.status_bar.showMessage(f"Preview loaded ({os.path.basename(first_file)}). Adjust parameters or run processing.")
            self.update_preview_plots()

        except Exception as e:
            self.log_error(f"Error loading preview data: {type(e).__name__} - {e}")
            self.preview_bur_df = None
            self.clear_all_plots("Preview")

    # --- Plotting Logic ---

    def update_preview_plots(self):
        """Updates plots using self.preview_bur_df and current spinbox values."""
        if self.preview_bur_df is None or self.preview_bur_df.empty:
            return

        time_cutoff = self.time_cutoff_spin.value()
        resid_cutoff = self.resid_cutoff_spin.value()

        try:
            # Run detection needed for plotting
            time_outliers, mean_diff, std_diff = detect_time_difference_outliers(self.preview_bur_df, time_cutoff)
            resid_outliers, slope, intercept, std_resid = detect_linear_residual_outliers(self.preview_bur_df, resid_cutoff)
            sg_sr = calculate_sg_sr(self.preview_bur_df)

            # Pass calculated params to the plotting function
            self._update_plot_data(
                self.preview_bur_df, sg_sr,
                time_outliers, (mean_diff, std_diff),
                resid_outliers, (slope, intercept, std_resid),
                "Preview"
            )

        except Exception as e:
            self.log_error(f"Error updating preview plots: {type(e).__name__} - {e}")


    def _update_plot_data(self, bur_df, sg_sr_series,
                          time_outliers, time_params,
                          resid_outliers, resid_params,
                          mode="Preview"):
        """
        Helper function to update all plots with new data, outlier masks, and calculated parameters.
        time_params = (mean_diff, std_diff)
        resid_params = (slope, intercept, std_resid)
        """
        if bur_df is None or bur_df.empty:
            self.clear_all_plots(mode)
            return

        title_suffix = f"({mode})"
        mean_diff, std_diff = time_params
        slope, intercept, std_resid = resid_params

        # --- Plot 1: Time Difference Histogram ---
        self.time_diff_hist_widget.setTitle(f'Time Difference Distribution {title_suffix}')
        if 'time_diff' in bur_df.columns and not pd.isna(mean_diff) and not pd.isna(std_diff):
            time_diff_vals = bur_df['time_diff'].dropna()
            if not time_diff_vals.empty:
                # Use calculated mean/std for thresholds
                upper_thresh = mean_diff + self.time_cutoff_spin.value() * std_diff
                lower_thresh = mean_diff - self.time_cutoff_spin.value() * std_diff

                y, x = np.histogram(time_diff_vals, bins='auto')
                bin_width = (x[1] - x[0]) * 0.9 if len(x) > 1 else 1
                self.time_diff_hist_item.setOpts(x=x[:-1], height=y, width=bin_width)
                self.time_diff_cutoff_line_upper.setPos(upper_thresh)
                self.time_diff_cutoff_line_lower.setPos(lower_thresh)
                self.time_diff_cutoff_line_upper.setVisible(True)
                self.time_diff_cutoff_line_lower.setVisible(True)
            else:
                self.clear_time_diff_plot(mode) # Clear if no valid data
        else:
            self.clear_time_diff_plot(mode)
            if 'time_diff' not in bur_df.columns:
                 self.time_diff_hist_widget.setTitle(f'Time Difference (Column Missing) {title_suffix}')
            else:
                 self.time_diff_hist_widget.setTitle(f'Time Difference (Cannot Calc Stats) {title_suffix}')


        # --- Plot 2: Linear Fit (Duration vs Photons) ---
        self.linear_fit_scatter_widget.setTitle(f'Linear Fit: Duration vs Photons {title_suffix}')
        required_cols_fit = ['Duration (ms)', 'Number of Photons']
        # Check if fit parameters are valid
        if all(col in bur_df.columns for col in required_cols_fit) and not any(pd.isna(p) for p in resid_params):
            # Update scatter plots
            all_scatter_data = [{'pos': (x, y), 'data': 1} for x, y in zip(bur_df['Duration (ms)'], bur_df['Number of Photons'])]
            outlier_scatter_data = [{'pos': (x, y), 'data': 1} for x, y in zip(bur_df.loc[resid_outliers, 'Duration (ms)'], bur_df.loc[resid_outliers, 'Number of Photons'])]
            self.linear_fit_scatter_all.setData(all_scatter_data)
            self.linear_fit_scatter_outliers.setData(outlier_scatter_data)

            # Generate points for lines using received params
            # Use actual data range for line extent
            valid_durations = bur_df['Duration (ms)'].dropna()
            if not valid_durations.empty:
                x_line = np.array([valid_durations.min(), valid_durations.max()])
                y_line = slope * x_line + intercept
                # Calculate bands using received std_resid
                upper_band_val = self.resid_cutoff_spin.value() * std_resid
                lower_band_val = -self.resid_cutoff_spin.value() * std_resid
                y_upper_band = y_line + upper_band_val
                y_lower_band = y_line + lower_band_val

                self.linear_fit_line.setData(x_line, y_line)
                self.linear_fit_band_upper.setData(x_line, y_upper_band)
                self.linear_fit_band_lower.setData(x_line, y_lower_band)

                self.linear_fit_line.setVisible(True)
                self.linear_fit_band_upper.setVisible(True)
                self.linear_fit_band_lower.setVisible(True)
            else: # No valid duration data
                 self.clear_linear_fit_plot(mode)

        else: # Missing columns or invalid fit params
            self.clear_linear_fit_plot(mode)
            if not all(col in bur_df.columns for col in required_cols_fit):
                 self.linear_fit_scatter_widget.setTitle(f'Linear Fit (Columns Missing) {title_suffix}')
            else:
                 self.linear_fit_scatter_widget.setTitle(f'Linear Fit (Fit Failed) {title_suffix}')


        # --- Plot 3: Sg/Sr vs Macro Time ---
        self.sg_sr_scatter_widget.setTitle(f'Sg/Sr vs Macro Time {title_suffix}')
        required_cols_sgsr = ['Mean Macro Time (ms)']
        if all(col in bur_df.columns for col in required_cols_sgsr) and sg_sr_series is not None:
            sgsr_data = pd.DataFrame({'macro_time': bur_df['Mean Macro Time (ms)'], 'sgsr': sg_sr_series}).dropna()
            if not sgsr_data.empty:
                # Combine outlier masks (True if outlier in EITHER time OR residual)
                combined_outliers = time_outliers | resid_outliers
                aligned_outliers = combined_outliers.reindex(sgsr_data.index).fillna(False)

                included_mask = ~aligned_outliers
                outlier_mask = aligned_outliers

                included_scatter_data = [{'pos': (mt, sgsr), 'data': 1} for mt, sgsr in zip(sgsr_data.loc[included_mask, 'macro_time'], sgsr_data.loc[included_mask, 'sgsr'])]
                outlier_scatter_data = [{'pos': (mt, sgsr), 'data': 1} for mt, sgsr in zip(sgsr_data.loc[outlier_mask, 'macro_time'], sgsr_data.loc[outlier_mask, 'sgsr'])]

                self.sg_sr_scatter_included.setData(included_scatter_data)
                self.sg_sr_scatter_outliers.setData(outlier_scatter_data)
            else:
                self.clear_sgsr_plot(mode)
        else:
            self.clear_sgsr_plot(mode)
            if not all(col in bur_df.columns for col in required_cols_sgsr):
                 self.sg_sr_scatter_widget.setTitle(f'Sg/Sr vs Macro Time (Column Missing) {title_suffix}')
            else:
                 self.sg_sr_scatter_widget.setTitle(f'Sg/Sr vs Macro Time (SgSr Calc Failed) {title_suffix}')


    def clear_all_plots(self, mode="Preview"):
        self.clear_time_diff_plot(mode)
        self.clear_linear_fit_plot(mode)
        self.clear_sgsr_plot(mode)

    def clear_time_diff_plot(self, mode="Preview"):
        title_suffix = f"({mode})"
        self.time_diff_hist_item.setOpts(x=[], height=[])
        self.time_diff_cutoff_line_upper.setVisible(False)
        self.time_diff_cutoff_line_lower.setVisible(False)
        self.time_diff_hist_widget.setTitle(f'Time Difference Distribution {title_suffix}')

    def clear_linear_fit_plot(self, mode="Preview"):
        title_suffix = f"({mode})"
        self.linear_fit_scatter_all.clear()
        self.linear_fit_scatter_outliers.clear()
        self.linear_fit_line.clear(); self.linear_fit_line.setVisible(False)
        self.linear_fit_band_upper.clear(); self.linear_fit_band_upper.setVisible(False)
        self.linear_fit_band_lower.clear(); self.linear_fit_band_lower.setVisible(False)
        self.linear_fit_scatter_widget.setTitle(f'Linear Fit: Duration vs Photons {title_suffix}')

    def clear_sgsr_plot(self, mode="Preview"):
        title_suffix = f"({mode})"
        self.sg_sr_scatter_included.clear()
        self.sg_sr_scatter_outliers.clear()
        self.sg_sr_scatter_widget.setTitle(f'Sg/Sr vs Macro Time {title_suffix}')


    # --- Worker Interaction and Control ---

    @pyqtSlot(str, pd.DataFrame)
    def on_worker_processing_file(self, filepath, bur_df):
        """Stores current data for live plotting (outliers plotted in next step)."""
        self.current_processing_bur_df = bur_df
        self.status_bar.showMessage(f"Processing: {os.path.basename(filepath)}")
        # Clear plots before receiving outlier data for this file
        self.clear_all_plots(f"Live: {os.path.basename(filepath)}")

    # Updated slot signature
    @pyqtSlot(pd.Series, tuple, pd.Series, tuple, pd.Series)
    def on_worker_outliers_detected(self, time_mask, time_params, resid_mask, resid_params, proj_mask):
        """Updates plots with detected outliers and parameters for the current file."""
        if self.current_processing_bur_df is not None:
            try:
                current_sgsr = calculate_sg_sr(self.current_processing_bur_df)
                # Pass all received data to the plotting function
                self._update_plot_data(
                    self.current_processing_bur_df, current_sgsr,
                    time_mask, time_params,
                    resid_mask, resid_params,
                    "Live"
                )
            except Exception as e:
                self.log_error(f"Error updating live plots: {type(e).__name__} - {e}")
        else:
            print("Warning: Received outlier data from worker, but no current_processing_bur_df is set.")


    def handle_run_pause_resume(self):
        if not self.is_running:
            # Start Processing
            self.input_dir = self.input_dir_edit.text()
            self.output_dir = self.output_dir_edit.text()
            if not self.input_dir or not self.output_dir:
                QMessageBox.warning(self, "Missing Information", "Please select both input and output directories.")
                return
            if not os.path.isdir(self.input_dir):
                 QMessageBox.critical(self, "Invalid Input", f"Input directory does not exist: {self.input_dir}")
                 return

            time_cutoff = self.time_cutoff_spin.value()
            proj_cutoff = self.proj_cutoff_spin.value()
            resid_cutoff = self.resid_cutoff_spin.value()

            self.status_bar.showMessage("Starting batch processing...")
            self.run_pause_button.setText("Pause")
            self.set_controls_enabled(False)
            self.is_running = True
            self.is_paused = False

            self.thread = QThread()
            self.worker = Worker(self.input_dir, self.output_dir, time_cutoff, proj_cutoff, resid_cutoff)
            self.worker.moveToThread(self.thread)

            # Connect signals
            self.worker.progress.connect(self.on_worker_progress)
            self.worker.error.connect(self.on_worker_error)
            self.worker.finished.connect(self.on_worker_finished)
            self.worker.processing_file.connect(self.on_worker_processing_file)
            self.worker.outliers_detected.connect(self.on_worker_outliers_detected) # Connect updated slot

            # Thread management
            self.thread.started.connect(self.worker.run)
            self.worker.finished.connect(self.thread.quit)
            self.worker.finished.connect(self.worker.deleteLater)
            self.thread.finished.connect(self.thread.deleteLater)
            self.thread.finished.connect(self.reset_worker_state)

            self.thread.start()

        elif self.is_running and not self.is_paused:
            # Pause Processing
            if self.worker:
                self.worker.pause()
                self.is_paused = True
                self.run_pause_button.setText("Resume")
                self.set_controls_enabled(True)
                self.status_bar.showMessage("Processing paused. Adjust parameters if needed.")

        elif self.is_running and self.is_paused:
            # Resume Processing
            if self.worker:
                time_cutoff = self.time_cutoff_spin.value()
                proj_cutoff = self.proj_cutoff_spin.value()
                resid_cutoff = self.resid_cutoff_spin.value()
                self.worker.update_parameters(time_cutoff, proj_cutoff, resid_cutoff)
                self.worker.resume()
                self.is_paused = False
                self.run_pause_button.setText("Pause")
                self.set_controls_enabled(False)
                self.status_bar.showMessage("Resuming processing...")

    def set_controls_enabled(self, enabled):
        self.input_dir_button.setEnabled(enabled)
        self.output_dir_button.setEnabled(enabled)
        self.time_cutoff_spin.setEnabled(enabled)
        self.proj_cutoff_spin.setEnabled(enabled)
        self.resid_cutoff_spin.setEnabled(enabled)

    @pyqtSlot(str)
    def on_worker_progress(self, message):
        self.status_bar.showMessage(message)

    @pyqtSlot(str)
    def on_worker_error(self, message):
        self.log_error(message)

    @pyqtSlot()
    def on_worker_finished(self):
        if self.is_running:
             self.status_bar.showMessage("Batch processing finished.")
        self.run_pause_button.setText("Run Processing")
        self.set_controls_enabled(True)
        self.is_running = False
        self.is_paused = False

    @pyqtSlot()
    def reset_worker_state(self):
         self.thread = None
         self.worker = None
         self.current_processing_bur_df = None

    def log_error(self, message):
         print(f"ERROR: {message}")
         self.status_bar.showMessage(f"Error: {message[:100]}...")

    def closeEvent(self, event):
        if self.is_running and self.worker:
            self.status_bar.showMessage("Window closed, stopping processing...")
            self.worker.stop()
            if self.thread:
                self.thread.quit()
                if not self.thread.wait(2000):
                     print("Warning: Worker thread did not stop gracefully.")
        event.accept()

# Main execution block
if __name__ == "__main__":
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    main_window = OutlierMainWindow()
    main_window.show()
    if app is QApplication.instance():
        sys.exit(app.exec())