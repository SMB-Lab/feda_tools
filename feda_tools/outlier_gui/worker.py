import os
import glob
import pandas as pd
import traceback
import time
from PyQt6.QtCore import QObject, pyqtSignal, QMutex, QWaitCondition, QMutexLocker

# Assuming the core module is importable
try:
    # Try relative import first
    from ..core.outlier_detection import (
        load_burst_data,
        detect_time_difference_outliers,
        detect_linear_projection_outliers,
        detect_linear_residual_outliers,
        calculate_sg_sr,
        apply_filters
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
            detect_linear_residual_outliers,
            calculate_sg_sr,
            apply_filters
        )
    except ImportError as e:
        print(f"Error importing outlier_detection: {e}. Using dummy functions.")
        # Dummy functions (ensure they return expected types)
        def load_burst_data(*args, **kwargs): return (pd.DataFrame({'time_diff': [], 'n_sum_g': [], 'n_sum_r': [], 'linear_resid': []}), pd.DataFrame(), pd.DataFrame())
        def detect_time_difference_outliers(*args, **kwargs): return pd.Series(dtype=bool)
        def detect_linear_projection_outliers(*args, **kwargs): return pd.Series(dtype=bool)
        def detect_linear_residual_outliers(*args, **kwargs): return pd.Series(dtype=bool)
        def calculate_sg_sr(*args, **kwargs): return pd.Series(dtype=float)
        def apply_filters(df, *args, **kwargs): return df


class Worker(QObject):
    # --- Signals ---
    progress = pyqtSignal(str)
    error = pyqtSignal(str)
    finished = pyqtSignal()
    # New signals for live updates
    processing_file = pyqtSignal(str, pd.DataFrame) # Emits filepath and loaded bur_df
    outliers_detected = pyqtSignal(pd.Series, pd.Series, pd.Series) # Emits boolean masks

    def __init__(self, input_dir, output_dir, time_cutoff, proj_cutoff, resid_cutoff):
        super().__init__()
        self.input_dir = input_dir
        self.output_dir = output_dir
        # Store initial parameters
        self._time_cutoff = time_cutoff
        self._proj_cutoff = proj_cutoff
        self._resid_cutoff = resid_cutoff
        # State flags
        self._is_running = True
        self._is_paused = False
        # Thread synchronization
        self._mutex = QMutex()
        self._pause_cond = QWaitCondition()

    # --- Public Methods for Control ---

    def stop(self):
        """Requests the worker to stop processing."""
        with QMutexLocker(self._mutex):
            self._is_running = False
            if self._is_paused: # Wake up if paused to allow exit
                self._is_paused = False
                self._pause_cond.wakeAll()

    def pause(self):
        """Requests the worker to pause processing."""
        with QMutexLocker(self._mutex):
            if self._is_running: # Only pause if running
                 self._is_paused = True

    def resume(self):
        """Requests the worker to resume processing."""
        with QMutexLocker(self._mutex):
            if self._is_paused:
                self._is_paused = False
                self._pause_cond.wakeAll() # Wake up the waiting thread

    def update_parameters(self, time_cutoff, proj_cutoff, resid_cutoff):
        """Updates the parameters used by the worker (thread-safe)."""
        with QMutexLocker(self._mutex):
            self._time_cutoff = time_cutoff
            self._proj_cutoff = proj_cutoff
            self._resid_cutoff = resid_cutoff
            # No need to wake thread here, parameters are read in the loop

    # --- Main Processing Logic ---

    def run(self):
        try:
            self.progress.emit("Starting processing...")
            search_pattern = os.path.join(self.input_dir, '**', 'bi4_bur', '*.bur')
            self.progress.emit(f"Searching for files matching: {search_pattern}")
            bur_files = sorted(glob.glob(search_pattern, recursive=True)) # Sort for consistent order

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
                        break # Exit the loop

                    while self._is_paused:
                        self.progress.emit("Processing paused...")
                        self._pause_cond.wait(self._mutex) # Wait until resume() is called
                        # Re-check if stopped while paused
                        if not self._is_running:
                            self.progress.emit("Processing stopped while paused.")
                            break # Exit the loop after waking up if stopped

                    if not self._is_running: # Check again after potential wake-up-and-stop
                         break

                    # Read current parameters within the lock for thread safety
                    current_time_cutoff = self._time_cutoff
                    current_proj_cutoff = self._proj_cutoff
                    current_resid_cutoff = self._resid_cutoff

                # --- Process Single File ---
                split_name_base = os.path.splitext(os.path.basename(bur_file_path))[0]
                self.progress.emit(f"Processing file {i+1}/{total_files}: {split_name_base}.bur")

                try:
                    # Determine base path
                    bi4_bur_dir = os.path.dirname(bur_file_path)
                    base_input_dir = os.path.dirname(bi4_bur_dir)

                    # Load data (uses updated core function)
                    bur_df, red_df, green_df = load_burst_data(base_input_dir, split_name_base)

                    if bur_df is None or bur_df.empty:
                         self.progress.emit(f"Warning: Skipping {split_name_base} due to loading error or empty data.")
                         continue # Skip to next file

                    # Emit data for live plotting
                    self.processing_file.emit(bur_file_path, bur_df.copy()) # Send a copy

                    # --- Outlier Detection (using current parameters) ---
                    time_outliers = detect_time_difference_outliers(bur_df, current_time_cutoff)
                    proj_outliers = detect_linear_projection_outliers(bur_df, current_proj_cutoff)
                    resid_outliers = detect_linear_residual_outliers(bur_df, current_resid_cutoff)

                    # Emit outlier masks for live plotting
                    self.outliers_detected.emit(time_outliers.copy(), proj_outliers.copy(), resid_outliers.copy())

                    # Combine filters & Apply
                    combined_outliers = time_outliers | proj_outliers | resid_outliers
                    keep_mask = ~combined_outliers
                    filtered_bur_df = apply_filters(bur_df.copy(), keep_mask)

                    if filtered_bur_df.empty:
                        self.progress.emit(f"Warning: All bursts filtered out for {split_name_base}. Skipping save.")
                        continue

                    # Calculate Sg/Sr & Add to filtered df
                    sg_sr_series = calculate_sg_sr(bur_df) # Calculate on original
                    sg_sr_aligned = sg_sr_series.reindex(filtered_bur_df.index)
                    filtered_bur_df['S_g/S_r'] = sg_sr_aligned

                    # Filter Red and Green Data
                    if red_df is None or green_df is None:
                         self.error.emit(f"Error: Red or Green dataframe became None unexpectedly for {split_name_base}. Skipping save.")
                         continue
                    filtered_red_df = apply_filters(red_df.copy(), keep_mask) # Also filter red/green based on bur mask
                    filtered_green_df = apply_filters(green_df.copy(), keep_mask)

                    # Prepare Output Paths
                    relative_path_bur = os.path.relpath(bur_file_path, start=self.input_dir)
                    output_bur_path = os.path.join(self.output_dir, relative_path_bur)
                    relative_dir = os.path.dirname(relative_path_bur)
                    base_relative_dir = os.path.dirname(relative_dir)
                    output_red_path = os.path.join(self.output_dir, base_relative_dir, 'br4', split_name_base + '.br4')
                    output_green_path = os.path.join(self.output_dir, base_relative_dir, 'bg4', split_name_base + '.bg4')

                    # Create output directories
                    os.makedirs(os.path.dirname(output_bur_path), exist_ok=True)
                    os.makedirs(os.path.dirname(output_red_path), exist_ok=True)
                    os.makedirs(os.path.dirname(output_green_path), exist_ok=True)

                    # Save Filtered Data
                    filtered_bur_df.to_csv(output_bur_path, sep='\t', index=False)
                    if filtered_red_df is not None: filtered_red_df.to_csv(output_red_path, sep='\t', index=False)
                    if filtered_green_df is not None: filtered_green_df.to_csv(output_green_path, sep='\t', index=False)

                    self.progress.emit(f"Saved filtered data for {split_name_base}")

                except Exception as e:
                    # Log errors for individual files but continue processing others
                    self.error.emit(f"Error processing {split_name_base}: {type(e).__name__} - {e}")
                    self.error.emit(traceback.format_exc())

                # Optional small delay to allow GUI updates
                # time.sleep(0.01)


            # --- Loop Finished ---
            with QMutexLocker(self._mutex):
                 if self._is_running: # Only emit finished if not stopped prematurely
                      self.progress.emit("Processing finished.")

        except Exception as e:
            # Catch critical errors during setup (e.g., glob)
            self.error.emit(f"Critical Error during processing setup: {type(e).__name__} - {e}")
            self.error.emit(traceback.format_exc())
        finally:
            self.finished.emit() # Always emit finished signal