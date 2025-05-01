import os
import glob
import pandas as pd
import numpy as np
import traceback
import time
from PyQt6.QtCore import QObject, pyqtSignal, QMutex, QWaitCondition, QMutexLocker

# Assuming the core module is importable
try:
    from ..core.outlier_detection import (
        load_burst_data,
        detect_time_difference_outliers,
        # detect_linear_projection_outliers, # Removed
        detect_linear_residual_outliers,
        calculate_sg_sr,
        apply_filters,
        get_combined_outliers # Updated signature
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
            # detect_linear_projection_outliers, # Removed
            detect_linear_residual_outliers,
            calculate_sg_sr,
            apply_filters,
            get_combined_outliers
        )
    except ImportError as e:
        print(f"Error importing outlier_detection: {e}. Using dummy functions.")
        # Dummy functions
        def load_burst_data(*args, **kwargs): return (pd.DataFrame({'time_diff': [], 'n_sum_g': [], 'n_sum_r': [], 'linear_resid': []}), pd.DataFrame(), pd.DataFrame())
        def detect_time_difference_outliers(*args, **kwargs): return pd.Series(dtype=bool), np.nan, np.nan
        # def detect_linear_projection_outliers(*args, **kwargs): return pd.Series(dtype=bool) # Removed
        def detect_linear_residual_outliers(*args, **kwargs): return pd.Series(dtype=bool), np.nan, np.nan, np.nan
        def calculate_sg_sr(*args, **kwargs): return pd.Series(dtype=float)
        def apply_filters(df, *args, **kwargs): return df
        def get_combined_outliers(*args, **kwargs): return pd.Series(dtype=bool)


class Worker(QObject):
    # --- Signals ---
    progress = pyqtSignal(str)
    error = pyqtSignal(str)
    finished = pyqtSignal()
    processing_file = pyqtSignal(str, pd.DataFrame)
    # Updated signal: removed projection mask
    outliers_detected = pyqtSignal(
        pd.Series, # time_mask
        tuple,     # (mean_diff, std_diff)
        pd.Series, # resid_mask
        tuple      # (slope, intercept, std_resid)
    )

    # Removed proj_cutoff from constructor
    def __init__(self, input_dir, output_dir, time_cutoff, resid_cutoff):
        super().__init__()
        self.input_dir = input_dir
        self.output_dir = output_dir
        self._time_cutoff = time_cutoff
        # self._proj_cutoff = proj_cutoff # Removed
        self._resid_cutoff = resid_cutoff
        self._is_running = True
        self._is_paused = False
        self._mutex = QMutex()
        self._pause_cond = QWaitCondition()

    # --- Public Methods for Control ---
    def stop(self):
        with QMutexLocker(self._mutex):
            self._is_running = False
            if self._is_paused:
                self._is_paused = False
                self._pause_cond.wakeAll()

    def pause(self):
        with QMutexLocker(self._mutex):
            if self._is_running:
                 self._is_paused = True

    def resume(self):
        with QMutexLocker(self._mutex):
            if self._is_paused:
                self._is_paused = False
                self._pause_cond.wakeAll()

    # Removed proj_cutoff from update_parameters
    def update_parameters(self, time_cutoff, resid_cutoff):
        with QMutexLocker(self._mutex):
            self._time_cutoff = time_cutoff
            # self._proj_cutoff = proj_cutoff # Removed
            self._resid_cutoff = resid_cutoff

    # --- Main Processing Logic ---
    def run(self):
        try:
            self.progress.emit("Starting processing...")
            search_pattern = os.path.join(self.input_dir, '**', 'bi4_bur', '*.bur')
            self.progress.emit(f"Searching for files matching: {search_pattern}")
            bur_files = sorted(glob.glob(search_pattern, recursive=True))

            if not bur_files:
                self.error.emit(f"No .bur files found recursively within '{os.path.join(self.input_dir, 'bi4_bur')}'")
                self.finished.emit()
                return

            total_files = len(bur_files)
            self.progress.emit(f"Found {total_files} .bur files to process.")

            for i, bur_file_path in enumerate(bur_files):
                # --- Check for Stop/Pause ---
                with QMutexLocker(self._mutex):
                    if not self._is_running:
                        self.progress.emit("Processing stopped by user.")
                        break
                    while self._is_paused:
                        self.progress.emit("Processing paused...")
                        self._pause_cond.wait(self._mutex)
                        if not self._is_running:
                            self.progress.emit("Processing stopped while paused.")
                            break
                    if not self._is_running: break

                    current_time_cutoff = self._time_cutoff
                    # current_proj_cutoff = self._proj_cutoff # Removed
                    current_resid_cutoff = self._resid_cutoff

                # --- Process Single File ---
                split_name_base = os.path.splitext(os.path.basename(bur_file_path))[0]
                self.progress.emit(f"Processing file {i+1}/{total_files}: {split_name_base}.bur")

                try:
                    bi4_bur_dir = os.path.dirname(bur_file_path)
                    base_input_dir = os.path.dirname(bi4_bur_dir)
                    bur_df, red_df, green_df = load_burst_data(base_input_dir, split_name_base)

                    if bur_df is None or bur_df.empty:
                         self.progress.emit(f"Warning: Skipping {split_name_base} due to loading error or empty data.")
                         continue

                    self.processing_file.emit(bur_file_path, bur_df.copy())

                    # --- Outlier Detection ---
                    time_outliers, mean_diff, std_diff = detect_time_difference_outliers(bur_df, current_time_cutoff)
                    resid_outliers, slope, intercept, std_resid = detect_linear_residual_outliers(bur_df, current_resid_cutoff)
                    # proj_outliers = detect_linear_projection_outliers(bur_df, current_proj_cutoff) # Removed call

                    # Emit masks AND plotting parameters (without projection mask)
                    self.outliers_detected.emit(
                        time_outliers.copy(),
                        (mean_diff, std_diff),
                        resid_outliers.copy(),
                        (slope, intercept, std_resid)
                        # proj_outliers removed
                    )

                    # --- Filtering & Saving ---
                    # Use the updated helper function (without proj_cutoff)
                    combined_outliers = get_combined_outliers(bur_df, current_time_cutoff, current_resid_cutoff)
                    keep_mask = ~combined_outliers
                    filtered_bur_df = apply_filters(bur_df.copy(), keep_mask)

                    if filtered_bur_df.empty:
                        self.progress.emit(f"Warning: All bursts filtered out for {split_name_base}. Skipping save.")
                        continue

                    sg_sr_series = calculate_sg_sr(bur_df)
                    sg_sr_aligned = sg_sr_series.reindex(filtered_bur_df.index)
                    filtered_bur_df['S_g/S_r'] = sg_sr_aligned

                    if red_df is None or green_df is None:
                         self.error.emit(f"Error: Red or Green dataframe missing for {split_name_base}. Skipping save.")
                         continue
                    filtered_red_df = apply_filters(red_df.copy(), keep_mask)
                    filtered_green_df = apply_filters(green_df.copy(), keep_mask)

                    # Prepare Output Paths
                    relative_path_bur = os.path.relpath(bur_file_path, start=self.input_dir)
                    output_bur_path = os.path.join(self.output_dir, relative_path_bur)
                    relative_dir = os.path.dirname(relative_path_bur)
                    base_relative_dir = os.path.dirname(relative_dir)
                    output_red_path = os.path.join(self.output_dir, base_relative_dir, 'br4', split_name_base + '.br4')
                    output_green_path = os.path.join(self.output_dir, base_relative_dir, 'bg4', split_name_base + '.bg4')

                    os.makedirs(os.path.dirname(output_bur_path), exist_ok=True)
                    os.makedirs(os.path.dirname(output_red_path), exist_ok=True)
                    os.makedirs(os.path.dirname(output_green_path), exist_ok=True)

                    filtered_bur_df.to_csv(output_bur_path, sep='\t', index=False)
                    if filtered_red_df is not None: filtered_red_df.to_csv(output_red_path, sep='\t', index=False)
                    if filtered_green_df is not None: filtered_green_df.to_csv(output_green_path, sep='\t', index=False)

                    self.progress.emit(f"Saved filtered data for {split_name_base}")

                except Exception as e:
                    self.error.emit(f"Error processing {split_name_base}: {type(e).__name__} - {e}")
                    self.error.emit(traceback.format_exc())

            # --- Loop Finished ---
            with QMutexLocker(self._mutex):
                 if self._is_running:
                      self.progress.emit("Processing finished.")

        except Exception as e:
            self.error.emit(f"Critical Error during processing setup: {type(e).__name__} - {e}")
            self.error.emit(traceback.format_exc())
        finally:
            self.finished.emit()