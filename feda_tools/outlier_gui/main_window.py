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
    # Fallback for running script directly
    from worker import Worker

# Assuming the core module is importable
try:
    # Try relative import first
    from ..core.outlier_detection import (
        load_burst_data,
        detect_time_difference_outliers,
        detect_linear_projection_outliers,
        detect_linear_residual_outliers
    )
except ImportError:
    # Fallback for development
    import sys
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(os.path.dirname(script_dir))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    try:
        from feda_tools.core.outlier_detection import (
            load_burst_data,
            detect_time_difference_outliers,
            detect_linear_projection_outliers,
            detect_linear_residual_outliers
        )
    except ImportError as e:
        print(f"Error importing outlier_detection: {e}. Using dummy functions.")
        # Dummy functions
        def load_burst_data(*args, **kwargs): return (pd.DataFrame({'time_diff': [], 'n_sum_g': [], 'n_sum_r': [], 'linear_resid': []}), pd.DataFrame(), pd.DataFrame())
        def detect_time_difference_outliers(*args, **kwargs): return pd.Series(dtype=bool)
        def detect_linear_projection_outliers(*args, **kwargs): return pd.Series(dtype=bool)
        def detect_linear_residual_outliers(*args, **kwargs): return pd.Series(dtype=bool)


# Configure pyqtgraph
pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')


class OutlierMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FEDA Tools - Interactive Outlier Detection & Batch Processing")
        self.setGeometry(100, 100, 1200, 800)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # Data storage
        self.preview_bur_df = None
        self.current_processing_bur_df = None # Holds data for live plot updates during run
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
        self._connect_signals()
        self._initialize_plots()

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

        self.proj_cutoff_spin = QDoubleSpinBox()
        self.proj_cutoff_spin.setRange(0.0, 100.0)
        self.proj_cutoff_spin.setValue(3.0)
        self.proj_cutoff_spin.setDecimals(2)
        self.proj_cutoff_spin.setSingleStep(0.1)
        self.proj_cutoff_spin.setToolTip("Z-score cutoff for linear projection distance.")

        self.resid_cutoff_spin = QDoubleSpinBox()
        self.resid_cutoff_spin.setRange(0.0, 100.0)
        self.resid_cutoff_spin.setValue(3.0)
        self.resid_cutoff_spin.setDecimals(2)
        self.resid_cutoff_spin.setSingleStep(0.1)
        self.resid_cutoff_spin.setToolTip("Z-score cutoff for linear fit residuals.")

        # Run/Pause/Resume Button
        self.run_pause_button = QPushButton("Run Processing")

        # Plot Widgets
        self.time_plot_widget = pg.PlotWidget(name="TimeDiffPlot")
        self.proj_plot_widget = pg.PlotWidget(name="ProjectionPlot")
        self.resid_plot_widget = pg.PlotWidget(name="ResidualPlot")

        # Status Display (Optional - can use status bar mainly)
        # self.status_display = QTextEdit()
        # self.status_display.setReadOnly(True)
        # self.status_display.setFixedHeight(100) # Limit height

    def _create_layouts(self):
        # Top Controls Layout
        top_controls_layout = QHBoxLayout()

        # Directory Selection
        dir_select_layout = QGridLayout()
        dir_select_layout.addWidget(self.input_dir_label, 0, 0)
        dir_select_layout.addWidget(self.input_dir_edit, 0, 1)
        dir_select_layout.addWidget(self.input_dir_button, 0, 2)
        dir_select_layout.addWidget(self.output_dir_label, 1, 0)
        dir_select_layout.addWidget(self.output_dir_edit, 1, 1)
        dir_select_layout.addWidget(self.output_dir_button, 1, 2)
        top_controls_layout.addLayout(dir_select_layout, 2) # Stretch factor 2

        # Parameters
        param_layout = QFormLayout()
        param_layout.addRow("Time Diff Z-Cutoff:", self.time_cutoff_spin)
        param_layout.addRow("Projection Z-Cutoff:", self.proj_cutoff_spin)
        param_layout.addRow("Residual Z-Cutoff:", self.resid_cutoff_spin)
        top_controls_layout.addLayout(param_layout, 1) # Stretch factor 1

        # Run Button
        run_button_layout = QVBoxLayout()
        run_button_layout.addWidget(self.run_pause_button)
        run_button_layout.addStretch()
        top_controls_layout.addLayout(run_button_layout)


        # Plots Layout
        plots_layout = QGridLayout()
        plots_layout.addWidget(self.time_plot_widget, 0, 0)
        plots_layout.addWidget(self.proj_plot_widget, 0, 1)
        plots_layout.addWidget(self.resid_plot_widget, 1, 0, 1, 2)

        # Main Layout
        self.layout.addLayout(top_controls_layout)
        self.layout.addLayout(plots_layout)
        # self.layout.addWidget(self.status_display) # Optional status box
        self.layout.setStretchFactor(plots_layout, 1) # Give plots more space

    def _initialize_plots(self):
        # Time Difference Plot
        self.time_plot_widget.setLabel('left', 'Time Difference')
        self.time_plot_widget.setLabel('bottom', 'Burst Index')
        self.time_plot_widget.setTitle('Time Difference Outliers (Preview)')
        self.time_plot_all_scatter = pg.ScatterPlotItem(pen=pg.mkPen(None), brush=pg.mkBrush(0, 0, 200, 150), size=5, name='Included')
        self.time_plot_outlier_scatter = pg.ScatterPlotItem(pen=pg.mkPen('r'), brush=pg.mkBrush(255, 0, 0, 200), size=7, symbol='x', name='Outlier')
        self.time_plot_widget.addItem(self.time_plot_all_scatter)
        self.time_plot_widget.addItem(self.time_plot_outlier_scatter)
        self.time_plot_widget.addLegend()

        # Projection Plot
        self.proj_plot_widget.setLabel('left', 'n_sum_g')
        self.proj_plot_widget.setLabel('bottom', 'n_sum_r')
        self.proj_plot_widget.setTitle('Linear Projection Outliers (Preview)')
        self.proj_plot_all_scatter = pg.ScatterPlotItem(pen=pg.mkPen(None), brush=pg.mkBrush(0, 200, 0, 150), size=5, name='Included')
        self.proj_plot_outlier_scatter = pg.ScatterPlotItem(pen=pg.mkPen('r'), brush=pg.mkBrush(255, 0, 0, 200), size=7, symbol='x', name='Outlier')
        self.proj_plot_widget.addItem(self.proj_plot_all_scatter)
        self.proj_plot_widget.addItem(self.proj_plot_outlier_scatter)
        self.proj_plot_widget.addLegend()

        # Residual Plot (Histogram)
        self.resid_plot_widget.setLabel('left', 'Count')
        self.resid_plot_widget.setLabel('bottom', 'Linear Residual Value')
        self.resid_plot_widget.setTitle('Linear Residual Outliers (Preview)')
        self.resid_hist_all = None
        self.resid_hist_outliers = None
        self.resid_plot_widget.addLegend()

    def _connect_signals(self):
        self.input_dir_button.clicked.connect(self.select_input_dir)
        self.output_dir_button.clicked.connect(self.select_output_dir)
        self.run_pause_button.clicked.connect(self.handle_run_pause_resume)
        # Update preview plots when parameters change IF NOT running
        self.time_cutoff_spin.valueChanged.connect(lambda: self.update_preview_plots() if not self.is_running else None)
        self.proj_cutoff_spin.valueChanged.connect(lambda: self.update_preview_plots() if not self.is_running else None)
        self.resid_cutoff_spin.valueChanged.connect(lambda: self.update_preview_plots() if not self.is_running else None)

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
        self.preview_bur_df = None # Clear previous preview
        self.clear_plots("Preview") # Clear plots before loading new preview
        if not self.input_dir:
            self.status_bar.showMessage("Select an input directory first.")
            return

        try:
            # Find the first .bur file recursively
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

            # Load data using the core function
            self.preview_bur_df, _, _ = load_burst_data(base_path, split_name) # Ignore red/green for preview plot

            if self.preview_bur_df is None or self.preview_bur_df.empty:
                self.log_error(f"Failed to load or empty preview data for {split_name}")
                self.preview_bur_df = None
                return

            self.status_bar.showMessage(f"Preview loaded ({os.path.basename(first_file)}). Adjust parameters or run processing.")
            self.update_preview_plots() # Update plots with loaded preview data

        except Exception as e:
            self.log_error(f"Error loading preview data: {e}")
            self.preview_bur_df = None
            self.clear_plots("Preview")

    def update_preview_plots(self):
        """Updates plots using self.preview_bur_df and current spinbox values."""
        if self.preview_bur_df is None or self.preview_bur_df.empty:
            # self.clear_plots("Preview") # Don't clear if just parameters changed without data
            return

        time_cutoff = self.time_cutoff_spin.value()
        proj_cutoff = self.proj_cutoff_spin.value()
        resid_cutoff = self.resid_cutoff_spin.value()

        try:
            time_outliers = detect_time_difference_outliers(self.preview_bur_df, time_cutoff)
            proj_outliers = detect_linear_projection_outliers(self.preview_bur_df, proj_cutoff)
            resid_outliers = detect_linear_residual_outliers(self.preview_bur_df, resid_cutoff)

            self._update_plot_data(self.preview_bur_df, time_outliers, proj_outliers, resid_outliers, "Preview")
            # self.status_bar.showMessage("Preview plots updated.") # Can be noisy

        except Exception as e:
            self.log_error(f"Error updating preview plots: {e}")
            # Optionally clear plots on error during update
            # self.clear_plots("Preview")

    @pyqtSlot(str, pd.DataFrame)
    def on_worker_processing_file(self, filepath, bur_df):
        """Slot for worker signal: stores current data for live plotting."""
        self.current_processing_bur_df = bur_df # Store the data
        self.status_bar.showMessage(f"Processing: {os.path.basename(filepath)}")
        # Clear previous outliers before showing new data points
        self.clear_plots(f"Processing: {os.path.basename(filepath)}")
        # Plot all points of the new file immediately
        self._plot_all_points(bur_df)


    @pyqtSlot(pd.Series, pd.Series, pd.Series)
    def on_worker_outliers_detected(self, time_mask, proj_mask, resid_mask):
        """Slot for worker signal: updates plots with detected outliers for the current file."""
        if self.current_processing_bur_df is not None:
            self._update_plot_data(self.current_processing_bur_df, time_mask, proj_mask, resid_mask, "Live")
        else:
            print("Warning: Received outlier data from worker, but no current_processing_bur_df is set.")


    def _plot_all_points(self, bur_df):
         """Helper to plot all points of a df without outlier info yet."""
         if bur_df is None or bur_df.empty: return
         try:
             # Time Plot
             if 'time_diff' in bur_df.columns:
                 time_all_data = [{'pos': (i, y), 'data': 1} for i, y in enumerate(bur_df['time_diff'])]
                 self.time_plot_all_scatter.setData(time_all_data)
             else: self.time_plot_all_scatter.clear()
             self.time_plot_outlier_scatter.clear() # Clear old outliers

             # Projection Plot
             if 'n_sum_r' in bur_df.columns and 'n_sum_g' in bur_df.columns:
                 proj_all_data = [{'pos': (x, y), 'data': 1} for x, y in zip(bur_df['n_sum_r'], bur_df['n_sum_g'])]
                 self.proj_plot_all_scatter.setData(proj_all_data)
             else: self.proj_plot_all_scatter.clear()
             self.proj_plot_outlier_scatter.clear() # Clear old outliers

             # Residual Plot (Histogram of all)
             if 'linear_resid' in bur_df.columns:
                 resid_vals = bur_df['linear_resid'].dropna()
                 if not resid_vals.empty:
                     y_all, x_all = np.histogram(resid_vals, bins='auto')
                     if self.resid_hist_all: self.resid_plot_widget.removeItem(self.resid_hist_all)
                     if self.resid_hist_outliers: self.resid_plot_widget.removeItem(self.resid_hist_outliers); self.resid_hist_outliers = None
                     self.resid_hist_all = pg.BarGraphItem(x=x_all[:-1], height=y_all, width=np.diff(x_all)[0]*0.8, brush='b', name='All Residuals')
                     self.resid_plot_widget.addItem(self.resid_hist_all)
                 else:
                      if self.resid_hist_all: self.resid_plot_widget.removeItem(self.resid_hist_all); self.resid_hist_all = None
                      if self.resid_hist_outliers: self.resid_plot_widget.removeItem(self.resid_hist_outliers); self.resid_hist_outliers = None
             else:
                  if self.resid_hist_all: self.resid_plot_widget.removeItem(self.resid_hist_all); self.resid_hist_all = None
                  if self.resid_hist_outliers: self.resid_plot_widget.removeItem(self.resid_hist_outliers); self.resid_hist_outliers = None

         except Exception as e:
              self.log_error(f"Error in _plot_all_points: {e}")


    def _update_plot_data(self, bur_df, time_outliers, proj_outliers, resid_outliers, mode="Preview"):
        """Helper function to update all plots with new data and outlier masks."""
        if bur_df is None or bur_df.empty:
            self.clear_plots(mode)
            return

        title_suffix = f"({mode})" if mode == "Preview" else f"(Live: {os.path.basename(self.input_dir_edit.text())})" # Improve live title later if needed

        try:
            # --- Update Time Plot ---
            if 'time_diff' in bur_df.columns:
                time_all_data = [{'pos': (i, y), 'data': 1} for i, y in enumerate(bur_df['time_diff'])]
                time_outlier_data = [{'pos': (i, y), 'data': 1} for i, y in enumerate(bur_df.loc[time_outliers, 'time_diff'])]
                self.time_plot_all_scatter.setData(time_all_data)
                self.time_plot_outlier_scatter.setData(time_outlier_data)
                self.time_plot_widget.setTitle(f'Time Difference Outliers ({time_outliers.sum()} found) {title_suffix}')
            else:
                self.time_plot_all_scatter.clear()
                self.time_plot_outlier_scatter.clear()
                self.time_plot_widget.setTitle(f'Time Difference Outliers (Column Missing) {title_suffix}')


            # --- Update Projection Plot ---
            if 'n_sum_r' in bur_df.columns and 'n_sum_g' in bur_df.columns:
                proj_all_data = [{'pos': (x, y), 'data': 1} for x, y in zip(bur_df['n_sum_r'], bur_df['n_sum_g'])]
                proj_outlier_data = [{'pos': (x, y), 'data': 1} for x, y in zip(bur_df.loc[proj_outliers, 'n_sum_r'], bur_df.loc[proj_outliers, 'n_sum_g'])]
                self.proj_plot_all_scatter.setData(proj_all_data)
                self.proj_plot_outlier_scatter.setData(proj_outlier_data)
                self.proj_plot_widget.setTitle(f'Linear Projection Outliers ({proj_outliers.sum()} found) {title_suffix}')
            else:
                 self.proj_plot_all_scatter.clear()
                 self.proj_plot_outlier_scatter.clear()
                 self.proj_plot_widget.setTitle(f'Linear Projection Outliers (Columns Missing) {title_suffix}')


            # --- Update Residual Plot (Histogram) ---
            if 'linear_resid' in bur_df.columns:
                resid_vals = bur_df['linear_resid'].dropna()
                if not resid_vals.empty:
                    resid_outlier_vals = resid_vals[resid_outliers.reindex(resid_vals.index).fillna(False)]
                    y_all, x_all = np.histogram(resid_vals, bins='auto')
                    y_outliers, _ = np.histogram(resid_outlier_vals, bins=x_all)

                    if self.resid_hist_all: self.resid_plot_widget.removeItem(self.resid_hist_all)
                    if self.resid_hist_outliers: self.resid_plot_widget.removeItem(self.resid_hist_outliers)

                    # Adjust width slightly to avoid overlap if needed
                    bar_width = np.diff(x_all)[0] * 0.4
                    self.resid_hist_all = pg.BarGraphItem(x=x_all[:-1], height=y_all, width=bar_width, brush='b', name='All Residuals')
                    self.resid_hist_outliers = pg.BarGraphItem(x=x_all[:-1] + bar_width, height=y_outliers, width=bar_width, brush='r', name='Outlier Residuals') # Offset slightly

                    self.resid_plot_widget.addItem(self.resid_hist_all)
                    self.resid_plot_widget.addItem(self.resid_hist_outliers)
                    self.resid_plot_widget.setTitle(f'Linear Residual Outliers ({resid_outliers.sum()} found) {title_suffix}')
                else:
                     self.clear_residual_histogram()
                     self.resid_plot_widget.setTitle(f'Linear Residual Outliers (No Valid Data) {title_suffix}')
            else:
                self.clear_residual_histogram()
                self.resid_plot_widget.setTitle(f'Linear Residual Outliers (Column Missing) {title_suffix}')

        except Exception as e:
            self.log_error(f"Error updating plot data: {e}")
            # Optionally clear plots on error
            # self.clear_plots(mode)

    def clear_plots(self, mode="Preview"):
        """Clears all plot data and resets titles."""
        title_suffix = f"({mode})"
        self.time_plot_all_scatter.clear()
        self.time_plot_outlier_scatter.clear()
        self.proj_plot_all_scatter.clear()
        self.proj_plot_outlier_scatter.clear()
        self.clear_residual_histogram()
        self.time_plot_widget.setTitle(f'Time Difference Outliers {title_suffix}')
        self.proj_plot_widget.setTitle(f'Linear Projection Outliers {title_suffix}')
        self.resid_plot_widget.setTitle(f'Linear Residual Outliers {title_suffix}')

    def clear_residual_histogram(self):
        if self.resid_hist_all: self.resid_plot_widget.removeItem(self.resid_hist_all); self.resid_hist_all = None
        if self.resid_hist_outliers: self.resid_plot_widget.removeItem(self.resid_hist_outliers); self.resid_hist_outliers = None


    def handle_run_pause_resume(self):
        if not self.is_running:
            # --- Start Processing ---
            self.input_dir = self.input_dir_edit.text() # Get current dirs
            self.output_dir = self.output_dir_edit.text()

            if not self.input_dir or not self.output_dir:
                QMessageBox.warning(self, "Missing Information", "Please select both input and output directories.")
                return
            if not os.path.isdir(self.input_dir):
                 QMessageBox.critical(self, "Invalid Input", f"Input directory does not exist: {self.input_dir}")
                 return

            # Get initial parameters
            time_cutoff = self.time_cutoff_spin.value()
            proj_cutoff = self.proj_cutoff_spin.value()
            resid_cutoff = self.resid_cutoff_spin.value()

            self.status_bar.showMessage("Starting batch processing...")
            # self.status_display.clear() # Optional: clear log box
            self.run_pause_button.setText("Pause")
            self.set_controls_enabled(False) # Disable controls during run
            self.is_running = True
            self.is_paused = False

            # Setup and run thread
            self.thread = QThread()
            self.worker = Worker(self.input_dir, self.output_dir, time_cutoff, proj_cutoff, resid_cutoff)
            self.worker.moveToThread(self.thread)

            # Connect worker signals to slots
            self.worker.progress.connect(self.on_worker_progress)
            self.worker.error.connect(self.on_worker_error)
            self.worker.finished.connect(self.on_worker_finished)
            self.worker.processing_file.connect(self.on_worker_processing_file)
            self.worker.outliers_detected.connect(self.on_worker_outliers_detected)

            # Clean up thread resources when finished
            self.thread.started.connect(self.worker.run)
            self.worker.finished.connect(self.thread.quit) # Quit thread when worker finishes
            self.worker.finished.connect(self.worker.deleteLater)
            self.thread.finished.connect(self.thread.deleteLater)
            # Ensure cleanup happens after main window slots are done
            self.thread.finished.connect(self.reset_worker_state)


            self.thread.start()

        elif self.is_running and not self.is_paused:
            # --- Pause Processing ---
            if self.worker:
                self.worker.pause()
                self.is_paused = True
                self.run_pause_button.setText("Resume")
                self.set_controls_enabled(True) # Re-enable parameter controls when paused
                self.status_bar.showMessage("Processing paused. Adjust parameters if needed.")

        elif self.is_running and self.is_paused:
            # --- Resume Processing ---
            if self.worker:
                # Get potentially updated parameters
                time_cutoff = self.time_cutoff_spin.value()
                proj_cutoff = self.proj_cutoff_spin.value()
                resid_cutoff = self.resid_cutoff_spin.value()
                self.worker.update_parameters(time_cutoff, proj_cutoff, resid_cutoff)
                self.worker.resume()
                self.is_paused = False
                self.run_pause_button.setText("Pause")
                self.set_controls_enabled(False) # Disable controls again
                self.status_bar.showMessage("Resuming processing...")

    def set_controls_enabled(self, enabled):
        """Enable/disable controls, keeping run button always enabled (but text changes)."""
        self.input_dir_button.setEnabled(enabled)
        self.output_dir_button.setEnabled(enabled)
        self.time_cutoff_spin.setEnabled(enabled)
        self.proj_cutoff_spin.setEnabled(enabled)
        self.resid_cutoff_spin.setEnabled(enabled)
        # Run/Pause button is handled separately

    @pyqtSlot(str)
    def on_worker_progress(self, message):
        self.status_bar.showMessage(message)
        # self.status_display.append(message) # Optional log box update

    @pyqtSlot(str)
    def on_worker_error(self, message):
        self.log_error(message) # Use existing log_error

    @pyqtSlot()
    def on_worker_finished(self):
        if self.is_running: # Avoid message if stopped manually
             self.status_bar.showMessage("Batch processing finished.")
        self.run_pause_button.setText("Run Processing")
        self.set_controls_enabled(True)
        self.is_running = False
        self.is_paused = False
        # Worker/thread cleanup is handled by finished signal connections

    @pyqtSlot()
    def reset_worker_state(self):
         """ Ensure worker and thread are reset after thread finishes"""
         self.thread = None
         self.worker = None
         self.current_processing_bur_df = None # Clear live data ref


    def log_error(self, message):
         print(f"ERROR: {message}") # Log to console
         self.status_bar.showMessage(f"Error: {message[:100]}...")
         # Avoid too many popups during batch processing, rely on status bar/console
         # QMessageBox.warning(self, "Error", message)

    def closeEvent(self, event):
        # Ensure thread is stopped if window is closed
        if self.is_running and self.worker:
            self.status_bar.showMessage("Window closed, stopping processing...")
            self.worker.stop() # Request worker stop
            if self.thread:
                self.thread.quit()
                if not self.thread.wait(2000): # Wait max 2 seconds
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