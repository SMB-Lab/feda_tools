import os
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
import warnings

# === Utility Functions ===

def drop_empty_columns(df):
    """Drops columns that are entirely zero or NaN."""
    if df is None: return None
    df = df.loc[:, (df != 0).any(axis=0)]
    df = df.dropna(axis=1, how='all')
    return df

def drop_empty_rows(df):
    """Drops rows where all numeric columns are zero or NaN."""
    if df is None: return None
    numeric_cols = df.select_dtypes(include=np.number).columns
    if not numeric_cols.empty:
        empty_rows_mask = df[numeric_cols].apply(lambda row: (row == 0) | row.isna(), axis=1).all(axis=1)
        return df[~empty_rows_mask]
    return df # Return original if no numeric columns

def drop_empty(df):
    """Applies both drop_empty_columns and drop_empty_rows."""
    df = drop_empty_columns(df)
    df = drop_empty_rows(df)
    return df

# === Core Data Loading and Calculation Functions ===

def _calculate_linear_residuals(bur_df):
    """Helper to calculate linear residuals (Duration vs Photons) and fit params."""
    required_cols_resid = ['Duration (ms)', 'Number of Photons']
    if not all(col in bur_df.columns for col in required_cols_resid):
        return pd.Series(np.nan, index=bur_df.index), np.nan, np.nan, np.nan # resid, slope, intercept, std_resid

    valid_data = bur_df[required_cols_resid].dropna()
    if len(valid_data) < 2: # Need at least 2 points for a fit
        return pd.Series(np.nan, index=bur_df.index), np.nan, np.nan, np.nan

    x_data = valid_data['Duration (ms)'].values.reshape(-1, 1)
    y_data = valid_data['Number of Photons'].values

    try:
        model = LinearRegression()
        model.fit(x_data, y_data)
        slope = model.coef_[0]
        intercept = model.intercept_

        # Calculate predictions and residuals on the original dataframe to keep index alignment
        y_pred = model.predict(bur_df['Duration (ms)'].values.reshape(-1, 1))
        residuals = pd.Series(bur_df['Number of Photons'] - y_pred, index=bur_df.index)

        # Calculate std dev based on the valid residuals used for fitting
        valid_residuals = residuals.loc[valid_data.index]
        std_resid = valid_residuals.std()
        if std_resid == 0 or pd.isna(std_resid): # Handle cases with no variance
             std_resid = np.nan

        return residuals, slope, intercept, std_resid

    except Exception as fit_err:
         warnings.warn(f"Linear regression fit failed for residual calculation: {fit_err}. Returning NaNs.", UserWarning)
         return pd.Series(np.nan, index=bur_df.index), np.nan, np.nan, np.nan


def load_burst_data(base_path, split_name):
    """
    Loads, cleans, and preprocesses burst data (.bur, .br4, .bg4) for a given split.

    Args:
        base_path (str): The directory containing 'bi4_bur', 'br4', 'bg4' subdirectories.
        split_name (str): The base name of the split files (without extension).

    Returns:
        tuple: (bur_df, red_df, green_df) pandas DataFrames, or (None, None, None) on error.
               bur_df will have added 'time_diff' and 'linear_resid' columns if possible.
    """
    bur_path = os.path.join(base_path, "bi4_bur", f"{split_name}.bur")
    red_path = os.path.join(base_path, "br4", f"{split_name}.br4")
    green_path = os.path.join(base_path, "bg4", f"{split_name}.bg4")

    try:
        if not os.path.exists(bur_path): raise FileNotFoundError(f"Burst file not found: {bur_path}")
        if not os.path.exists(red_path): raise FileNotFoundError(f"Red file not found: {red_path}")
        if not os.path.exists(green_path): raise FileNotFoundError(f"Green file not found: {green_path}")

        bur_df = pd.read_csv(bur_path, sep="\t")
        red_df = pd.read_csv(red_path, sep="\t")
        green_df = pd.read_csv(green_path, sep="\t")

        bur_df = drop_empty(bur_df)
        red_df = drop_empty(red_df)
        green_df = drop_empty(green_df)

        if bur_df is None or bur_df.empty:
            warnings.warn(f"Empty or invalid burst data after cleaning for {split_name}", UserWarning)
            return None, red_df, green_df

        # --- Pre-calculations ---
        # 1. Time Difference
        required_cols_time = ['Duration (green) (ms)', 'Duration (yellow) (ms)']
        if all(col in bur_df.columns for col in required_cols_time):
            bur_df['time_diff'] = bur_df['Duration (green) (ms)'] - bur_df['Duration (yellow) (ms)']
        else:
            warnings.warn(f"Missing columns for 'time_diff' calculation in {split_name}. Adding NaN column.", UserWarning)
            bur_df['time_diff'] = np.nan

        # 2. Linear Residual (Duration vs Number of Photons)
        residuals, _, _, _ = _calculate_linear_residuals(bur_df) # Calculate residuals here
        bur_df['linear_resid'] = residuals # Add the calculated residuals column

        return bur_df, red_df, green_df

    except FileNotFoundError as e:
        warnings.warn(str(e), UserWarning)
        return None, None, None
    except pd.errors.EmptyDataError:
        warnings.warn(f"Empty file encountered for split {split_name}", UserWarning)
        return None, None, None
    except Exception as e:
        warnings.warn(f"Error loading data for split {split_name}: {type(e).__name__} - {e}", UserWarning)
        return None, None, None


def detect_time_difference_outliers(bur_df, z_cutoff):
    """
    Detects outliers based on the z-score of the 'time_diff' column.

    Args:
        bur_df (pd.DataFrame): Burst DataFrame containing a 'time_diff' column.
        z_cutoff (float): The absolute z-score threshold for outlier detection.

    Returns:
        tuple: (outlier_mask, mean_diff, std_diff)
               outlier_mask (pd.Series): Boolean Series, True indicates an outlier.
               mean_diff (float): Mean of the 'time_diff' data used.
               std_diff (float): Standard deviation of the 'time_diff' data used.
               Returns (all-False Series, NaN, NaN) on error or invalid data.
    """
    default_return = (pd.Series(False, index=bur_df.index if bur_df is not None else None), np.nan, np.nan)

    if bur_df is None or 'time_diff' not in bur_df.columns:
        warnings.warn("Missing 'time_diff' column for outlier detection.", UserWarning)
        return default_return

    diff_data = bur_df['time_diff'].dropna()
    if len(diff_data) < 2: # Need at least 2 points for std dev
         warnings.warn("'time_diff' column has < 2 valid values. Cannot calculate z-scores.", UserWarning)
         return default_return

    mean_diff = diff_data.mean()
    std_diff = diff_data.std()

    if std_diff == 0 or pd.isna(std_diff):
         warnings.warn("'time_diff' column has zero or NaN standard deviation. Cannot calculate z-scores.", UserWarning)
         return (pd.Series(False, index=bur_df.index), mean_diff, std_diff if not pd.isna(std_diff) else np.nan)

    z_scores = (diff_data - mean_diff) / std_diff
    outlier_mask = abs(z_scores) > z_cutoff

    full_outlier_mask = outlier_mask.reindex(bur_df.index, fill_value=False)
    return full_outlier_mask, mean_diff, std_diff


def detect_linear_residual_outliers(bur_df, z_cutoff):
    """
    Detects outliers based on the z-score of the 'linear_resid' column
    (residuals from Duration vs Number of Photons fit).

    Args:
        bur_df (pd.DataFrame): Burst DataFrame containing a 'linear_resid' column.
        z_cutoff (float): The absolute z-score threshold for outlier detection.

    Returns:
        tuple: (outlier_mask, slope, intercept, std_resid)
               outlier_mask (pd.Series): Boolean Series, True indicates an outlier.
               slope (float): Slope of the Duration vs Photons fit.
               intercept (float): Intercept of the Duration vs Photons fit.
               std_resid (float): Standard deviation of the residuals used for z-score.
               Returns (all-False Series, NaN, NaN, NaN) on error or invalid data.
    """
    default_return = (pd.Series(False, index=bur_df.index if bur_df is not None else None), np.nan, np.nan, np.nan)

    if bur_df is None or 'linear_resid' not in bur_df.columns:
        warnings.warn("Missing 'linear_resid' column for outlier detection.", UserWarning)
        return default_return

    residuals, slope, intercept, std_resid = _calculate_linear_residuals(bur_df)

    if pd.isna(slope) or pd.isna(intercept) or pd.isna(std_resid):
         warnings.warn("Fit parameters or residual std dev could not be calculated. Cannot determine residual outliers.", UserWarning)
         # Return fit params even if std_resid is bad, but mask is False
         return (pd.Series(False, index=bur_df.index), slope, intercept, std_resid)

    resid_data = bur_df['linear_resid'].dropna()
    if resid_data.empty:
         warnings.warn("'linear_resid' column is empty after dropna. Cannot calculate z-scores.", UserWarning)
         return (pd.Series(False, index=bur_df.index), slope, intercept, std_resid)

    z_scores = resid_data / std_resid
    outlier_mask = abs(z_scores) > z_cutoff

    full_outlier_mask = outlier_mask.reindex(bur_df.index, fill_value=False)
    return full_outlier_mask, slope, intercept, std_resid


def calculate_sg_sr(bur_df):
    """
    Calculates the S_g/S_r ratio.

    Args:
        bur_df (pd.DataFrame): Burst DataFrame with necessary columns.

    Returns:
        pd.Series: Series containing the S_g/S_r ratio, or NaNs if columns missing.
                   Index matches bur_df.
    """
    required_cols = ['Number of Photons (green)', 'Duration (green) (ms)']
    if bur_df is None or not all(col in bur_df.columns for col in required_cols):
        warnings.warn(f"Missing required columns {required_cols} for Sg/Sr calculation.", UserWarning)
        return pd.Series(np.nan, index=bur_df.index if bur_df is not None else None)

    duration_g = bur_df['Duration (green) (ms)']
    valid_duration_mask = (duration_g.notna()) & (duration_g != 0)

    sg_sr = pd.Series(np.nan, index=bur_df.index)
    photons_g = pd.to_numeric(bur_df['Number of Photons (green)'], errors='coerce')
    sg_sr[valid_duration_mask] = photons_g[valid_duration_mask] / duration_g[valid_duration_mask]

    return sg_sr


def apply_filters(df, keep_mask):
    """
    Filters a DataFrame based on a boolean mask.

    Args:
        df (pd.DataFrame): The DataFrame to filter.
        keep_mask (pd.Series): Boolean mask (same index as df). True values are kept.

    Returns:
        pd.DataFrame: The filtered DataFrame. Returns original df if mask is invalid.
    """
    if df is None: return None
    if not isinstance(keep_mask, pd.Series) or not pd.api.types.is_bool_dtype(keep_mask):
        warnings.warn("Invalid keep_mask provided to apply_filters. Must be a boolean Series.", UserWarning)
        return df
    if not df.index.equals(keep_mask.index):
         warnings.warn("Index mismatch between DataFrame and keep_mask in apply_filters.", UserWarning)
         try:
             aligned_mask = keep_mask.reindex(df.index, fill_value=False)
             return df[aligned_mask]
         except Exception:
              warnings.warn("Could not align mask index. Returning original DataFrame.", UserWarning)
              return df

    return df[keep_mask]

def get_combined_outliers(bur_df, time_cutoff, resid_cutoff):
    """
    Runs relevant outlier detection methods and returns a combined mask.
    (Excludes projection method).

    Args:
        bur_df (pd.DataFrame): Burst data.
        time_cutoff (float): Z-score cutoff for time difference.
        resid_cutoff (float): Z-score cutoff for linear residuals.

    Returns:
        pd.Series: Combined boolean mask where True indicates an outlier by time OR residual method.
    """
    time_outliers, _, _ = detect_time_difference_outliers(bur_df, time_cutoff)
    resid_outliers, _, _, _ = detect_linear_residual_outliers(bur_df, resid_cutoff)

    # Combine using logical OR. Ensure masks are aligned to bur_df index.
    combined = (time_outliers.reindex(bur_df.index, fill_value=False) |
                resid_outliers.reindex(bur_df.index, fill_value=False))
    return combined