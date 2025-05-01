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
    df = df.dropna(axis=1, how='all') # Drop columns that are entirely NaN
    return df

def drop_empty_rows(df):
    """Drops rows where all numeric columns are zero or NaN."""
    if df is None: return None
    numeric_cols = df.select_dtypes(include=np.number).columns
    # Check if all numeric values in a row are 0 or NaN
    empty_rows_mask = df[numeric_cols].apply(lambda row: (row == 0) | row.isna(), axis=1).all(axis=1)
    return df[~empty_rows_mask]

def drop_empty(df):
    """Applies both drop_empty_columns and drop_empty_rows."""
    df = drop_empty_columns(df)
    df = drop_empty_rows(df)
    return df

# === Core Data Loading and Calculation Functions ===

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

        # Clean data
        bur_df = drop_empty(bur_df)
        red_df = drop_empty(red_df)
        green_df = drop_empty(green_df)

        if bur_df is None or bur_df.empty:
            warnings.warn(f"Empty or invalid burst data after cleaning for {split_name}", UserWarning)
            return None, red_df, green_df # Return potentially valid red/green even if bur is bad

        # --- Pre-calculations ---
        # 1. Time Difference
        required_cols_time = ['Duration (green) (ms)', 'Duration (yellow) (ms)']
        if all(col in bur_df.columns for col in required_cols_time):
            bur_df['time_diff'] = bur_df['Duration (green) (ms)'] - bur_df['Duration (yellow) (ms)']
        else:
            warnings.warn(f"Missing columns for 'time_diff' calculation in {split_name}. Skipping.", UserWarning)
            bur_df['time_diff'] = np.nan # Add NaN column if calculation fails

        # 2. Linear Residual (Duration vs Number of Photons)
        required_cols_resid = ['Duration (ms)', 'Number of Photons']
        if all(col in bur_df.columns for col in required_cols_resid):
            # Drop NaNs before fitting
            valid_data = bur_df[required_cols_resid].dropna()
            if not valid_data.empty:
                x_data = valid_data['Duration (ms)'].values.reshape(-1, 1)
                y_data = valid_data['Number of Photons'].values
                try:
                    model = LinearRegression()
                    model.fit(x_data, y_data)
                    y_pred = model.predict(bur_df['Duration (ms)'].values.reshape(-1, 1))
                    # Assign residuals back, aligning index
                    residuals = pd.Series(bur_df['Number of Photons'] - y_pred, index=bur_df.index)
                    bur_df['linear_resid'] = residuals
                except Exception as fit_err:
                     warnings.warn(f"Linear regression fit failed for residual calculation in {split_name}: {fit_err}. Skipping.", UserWarning)
                     bur_df['linear_resid'] = np.nan
            else:
                 warnings.warn(f"No valid data for residual calculation in {split_name} after dropping NaNs. Skipping.", UserWarning)
                 bur_df['linear_resid'] = np.nan
        else:
            warnings.warn(f"Missing columns for 'linear_resid' calculation in {split_name}. Skipping.", UserWarning)
            bur_df['linear_resid'] = np.nan # Add NaN column if calculation fails


        return bur_df, red_df, green_df

    except FileNotFoundError as e:
        warnings.warn(str(e), UserWarning)
        return None, None, None
    except pd.errors.EmptyDataError:
        warnings.warn(f"Empty file encountered for split {split_name}", UserWarning)
        return None, None, None # Or return empty DFs if preferred
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
        pd.Series: Boolean Series (same index as bur_df), True indicates an outlier.
                   Returns an all-False Series if 'time_diff' is missing or invalid.
    """
    if bur_df is None or 'time_diff' not in bur_df.columns:
        warnings.warn("Missing 'time_diff' column for outlier detection.", UserWarning)
        return pd.Series(False, index=bur_df.index if bur_df is not None else None)

    diff_data = bur_df['time_diff'].dropna()
    if diff_data.empty or diff_data.std() == 0:
         warnings.warn("'time_diff' column has no variance or is empty after dropna. Cannot calculate z-scores.", UserWarning)
         return pd.Series(False, index=bur_df.index) # No outliers if no variance

    z_scores = stats.zscore(diff_data)
    outlier_mask = abs(z_scores) > z_cutoff

    # Reindex mask to match original bur_df index, filling missing with False
    return outlier_mask.reindex(bur_df.index, fill_value=False)


def detect_linear_projection_outliers(bur_df, z_cutoff):
    """
    Detects outliers based on the z-score of the perpendicular distance
    from the linear regression line of 'n_sum_g' vs 'n_sum_r'.

    Args:
        bur_df (pd.DataFrame): Burst DataFrame with 'n_sum_g' and 'n_sum_r'.
        z_cutoff (float): The absolute z-score threshold for outlier detection.

    Returns:
        pd.Series: Boolean Series (same index as bur_df), True indicates an outlier.
                   Returns an all-False Series if required columns are missing/invalid.
    """
    required_cols = ['n_sum_g', 'n_sum_r']
    if bur_df is None or not all(col in bur_df.columns for col in required_cols):
        warnings.warn(f"Missing required columns {required_cols} for projection outlier detection.", UserWarning)
        return pd.Series(False, index=bur_df.index if bur_df is not None else None)

    # Drop rows with NaNs in required columns for fitting and distance calculation
    valid_data = bur_df[required_cols].dropna()
    if valid_data.empty:
         warnings.warn("No valid data for projection outlier detection after dropping NaNs.", UserWarning)
         return pd.Series(False, index=bur_df.index)

    x_data = valid_data['n_sum_r'].values.reshape(-1, 1)
    y_data = valid_data['n_sum_g'].values

    if len(valid_data) < 2:
         warnings.warn("Not enough valid data points (<2) for linear regression in projection outlier detection.", UserWarning)
         return pd.Series(False, index=bur_df.index)

    try:
        model = LinearRegression()
        model.fit(x_data, y_data)
        slope = model.coef_[0]
        intercept = model.intercept_

        # Calculate perpendicular distance for each point in valid_data
        # Distance = |slope*x - y + intercept| / sqrt(slope^2 + 1)
        distances = np.abs(slope * valid_data['n_sum_r'] - valid_data['n_sum_g'] + intercept) / np.sqrt(slope**2 + 1)

        if distances.std() == 0:
             warnings.warn("Perpendicular distances have zero variance. Cannot calculate z-scores for projection outliers.", UserWarning)
             return pd.Series(False, index=bur_df.index)

        z_scores = stats.zscore(distances)
        outlier_mask = abs(z_scores) > z_cutoff

        # Reindex mask to match original bur_df index, filling missing with False
        return outlier_mask.reindex(bur_df.index, fill_value=False)

    except Exception as e:
        warnings.warn(f"Error during linear projection outlier detection: {type(e).__name__} - {e}", UserWarning)
        return pd.Series(False, index=bur_df.index)


def detect_linear_residual_outliers(bur_df, z_cutoff):
    """
    Detects outliers based on the z-score of the 'linear_resid' column.

    Args:
        bur_df (pd.DataFrame): Burst DataFrame containing a 'linear_resid' column.
        z_cutoff (float): The absolute z-score threshold for outlier detection.

    Returns:
        pd.Series: Boolean Series (same index as bur_df), True indicates an outlier.
                   Returns an all-False Series if 'linear_resid' is missing or invalid.
    """
    if bur_df is None or 'linear_resid' not in bur_df.columns:
        warnings.warn("Missing 'linear_resid' column for outlier detection.", UserWarning)
        return pd.Series(False, index=bur_df.index if bur_df is not None else None)

    resid_data = bur_df['linear_resid'].dropna()
    if resid_data.empty or resid_data.std() == 0:
         warnings.warn("'linear_resid' column has no variance or is empty after dropna. Cannot calculate z-scores.", UserWarning)
         return pd.Series(False, index=bur_df.index) # No outliers if no variance

    z_scores = stats.zscore(resid_data)
    outlier_mask = abs(z_scores) > z_cutoff

    # Reindex mask to match original bur_df index, filling missing with False
    return outlier_mask.reindex(bur_df.index, fill_value=False)


def calculate_sg_sr(bur_df):
    """
    Calculates the S_g/S_r ratio.

    Args:
        bur_df (pd.DataFrame): Burst DataFrame with necessary columns.

    Returns:
        pd.Series: Series containing the S_g/S_r ratio, or NaNs if columns missing.
                   Index matches bur_df.
    """
    required_cols = ['Number of Photons (green)', 'Duration (green) (ms)'] # Assuming Sr is proportional to Duration
    if bur_df is None or not all(col in bur_df.columns for col in required_cols):
        warnings.warn(f"Missing required columns {required_cols} for Sg/Sr calculation.", UserWarning)
        return pd.Series(np.nan, index=bur_df.index if bur_df is not None else None)

    # Avoid division by zero or NaN duration
    duration_g = bur_df['Duration (green) (ms)']
    valid_duration_mask = (duration_g.notna()) & (duration_g != 0)

    sg_sr = pd.Series(np.nan, index=bur_df.index)
    sg_sr[valid_duration_mask] = bur_df.loc[valid_duration_mask, 'Number of Photons (green)'] / duration_g[valid_duration_mask]

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
         # Attempt to align if possible, otherwise return original df
         try:
             aligned_mask = keep_mask.reindex(df.index, fill_value=False)
             return df[aligned_mask]
         except Exception:
              warnings.warn("Could not align mask index. Returning original DataFrame.", UserWarning)
              return df

    return df[keep_mask]