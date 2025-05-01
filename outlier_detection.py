# %% [markdown]
# ## Initialization

# %% [markdown]
# ### Imports + Definitions

# %%
import os

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# %%
def drop_empty_columns(df):
    df = df.loc[:, (df != 0).any(axis=0)]
    df = df.dropna(axis=1)
    return df

def drop_empty_rows(df):
    # Get only numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns
    
    # Check if all numeric values in a row are 0
    empty_rows = (df[numeric_cols] == 0).all(axis=1)
    
    # Return dataframe with non-empty rows
    return df[~empty_rows]


def drop_empty(df):
    df = drop_empty_columns(df)
    df = drop_empty_rows(df)
    return df

# %%
# base_path = "data/Combined_old_thresholds/burstwise_All 0.2944#60/"
# split_path = "Split_After_Adjust_HF_54000s_pinhole6-000003_0"

# red_df = pd.read_csv(os.path.join(base_path, "br4", split_path) + ".br4", sep="\t")
# green_df = pd.read_csv(os.path.join(base_path, "bg4", split_path) + ".bg4", sep="\t")
# bur_df = pd.read_csv(os.path.join(base_path, "bi4_bur", split_path) + ".bur", sep="\t")

# red_df = drop_empty(red_df)
# green_df = drop_empty(green_df)
# bur_df = drop_empty(bur_df)

# df = pd.concat([red_df, green_df], axis=1)

import os
import pandas as pd
import ipywidgets as widgets
from IPython.display import display

def load_data_for_split(base_path, split_path):
    """Load data for a specific split path"""
    try:
        red_df = pd.read_csv(os.path.join(base_path, "br4", split_path) + ".br4", sep="\t")
        green_df = pd.read_csv(os.path.join(base_path, "bg4", split_path) + ".bg4", sep="\t")
        bur_df = pd.read_csv(os.path.join(base_path, "bi4_bur", split_path) + ".bur", sep="\t")
        
        red_df = drop_empty(red_df)
        green_df = drop_empty(green_df)
        bur_df = drop_empty(bur_df)
        
        df = pd.concat([red_df, green_df], axis=1)
        
        return red_df, green_df, bur_df, df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None, None

def get_available_split_paths(base_path):
    """Get all available split paths from the base directory"""
    # Look in br4 folder since all splits should have .br4 files
    br4_path = os.path.join(base_path, "br4")
    if not os.path.exists(br4_path):
        return []
    
    # Get all br4 files and extract their base names
    files = [f for f in os.listdir(br4_path) if f.endswith('.br4')]
    split_paths = [os.path.splitext(f)[0] for f in files]
    
    return split_paths

# Set the base path
base_path = "data/Combined_old_thresholds/burstwise_All 0.2944#60/"

# Get available split paths
split_paths = get_available_split_paths(base_path)

# Create dropdown widget
split_dropdown = widgets.Dropdown(
    options=split_paths,
    description='Split Path:',
    style={'description_width': 'initial'},
    layout=widgets.Layout(width='80%')
)

# Create load button
load_button = widgets.Button(
    description='Load Data',
    button_style='primary',
    tooltip='Click to load data for selected split path'
)

# Output widget for displaying status
output = widgets.Output()

# Define button click handler
def on_load_button_clicked(b):
    with output:
        output.clear_output()
        print(f"Loading data for: {split_dropdown.value}")
        red_df, green_df, bur_df, df = load_data_for_split(base_path, split_dropdown.value)
        if df is not None:
            print(f"Data loaded successfully! Shape: {df.shape}")
            # Make the dataframes available in the global namespace
            globals()['red_df'] = red_df
            globals()['green_df'] = green_df
            globals()['bur_df'] = bur_df
            globals()['df'] = df
            globals()['split_path'] = split_dropdown.value

# Register the button click handler
load_button.on_click(on_load_button_clicked)

# %% [markdown]
# ### Data Loading

# %%
bur_df = None

# Display widgets
display(widgets.VBox([
    widgets.HTML(value="<h3>Select a split path:</h3>"),
    split_dropdown,
    load_button,
    output
]))

# %% [markdown]
# ### Data Frame Display

# %%
bur_df.head(n=10)


# First fig :
#  - TGX-TRR
#  - Actual burst duration


# Sg/Sr:
# - Green Count / Red Count
# Sg = Number of Photons (green) / Duration (green) (ms)
# Sr = S prompt red (kHz) | 2-180

# %%
red_df.head()

# %%
green_df.head()

# %%
df.head()

# %%
green_df['gamma (green)'].describe()

# %% [markdown]
# ## Exploratory Analysis + Initial Outlier Detection Attempts

# %% [markdown]
# Analyzing the normal distribution of $T_{GX} - T_{RR}$ (Green Duration - Yellow Duration)

# %%
# Set the z-score cutoff as a constant
Z_SCORE_CUTOFF = 1

# Calculate the difference
bur_df['Time Difference'] = bur_df['Duration (green) (ms)'] - bur_df['Duration (yellow) (ms)']

# Calculate z-scores for the Time Difference
from scipy import stats
bur_df['z_score'] = stats.zscore(bur_df['Time Difference'])

# Create a mask for outliers (absolute z-score > Z_SCORE_CUTOFF)
outlier_mask = abs(bur_df['z_score']) > Z_SCORE_CUTOFF

# Calculate the threshold values corresponding to z-score = ±Z_SCORE_CUTOFF
mean_diff = bur_df['Time Difference'].mean()
std_diff = bur_df['Time Difference'].std()
upper_threshold = mean_diff + Z_SCORE_CUTOFF * std_diff
lower_threshold = mean_diff - Z_SCORE_CUTOFF * std_diff

# Scatter plot
plt.figure(figsize=(10, 6))
# Plot all points, coloring based on absolute z-score
plt.scatter(range(len(bur_df)), bur_df['Time Difference'], 
            c=['red' if abs(z) > Z_SCORE_CUTOFF else 'purple' for z in bur_df['z_score']], 
            alpha=0.7)

plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
plt.axhline(y=upper_threshold, color='red', linestyle='--', alpha=0.7, 
            label=f'Upper threshold (z={Z_SCORE_CUTOFF}): {upper_threshold:.2f}')
plt.axhline(y=lower_threshold, color='red', linestyle='--', alpha=0.7,
            label=f'Lower threshold (z=-{Z_SCORE_CUTOFF}): {lower_threshold:.2f}')
            
plt.xlabel("Data Point Index")
plt.ylabel("Duration (green) (ms) - Duration (yellow) (ms)")
plt.title(f"Difference Between Green and Yellow Burst Durations\nPoints with |z-score| > {Z_SCORE_CUTOFF} shown in red")
plt.legend()
plt.tight_layout()
plt.show()

# Histogram - single histogram with threshold lines
plt.figure(figsize=(10, 6))
# Plot all data in one histogram
sns.histplot(bur_df['Time Difference'], kde=True, color='purple')

# Add vertical lines for z-score = ±Z_SCORE_CUTOFF thresholds
plt.axvline(x=upper_threshold, color='red', linestyle='--', 
            label=f'z-score = +{Z_SCORE_CUTOFF} (value = {upper_threshold:.2f})')
plt.axvline(x=lower_threshold, color='red', linestyle='--', 
            label=f'z-score = -{Z_SCORE_CUTOFF} (value = {lower_threshold:.2f})')

# Add zero line
plt.axvline(x=0, color='gray', linestyle='--', alpha=0.7)

plt.xlabel("Duration (green) (ms) - Duration (yellow) (ms)")
plt.ylabel("Frequency")
plt.title("Distribution of Differences Between Green and Yellow Burst Durations")
plt.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# They would usually arbitrarilly cut, for the above data maybe 1 would be the cuttoff

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression

# 1. Visualize the original data
plt.figure(figsize=(10, 6))
sns.scatterplot(bur_df, x="Duration (ms)", y="Number of Photons")
plt.title("Original Data Visualization")
plt.grid(True, alpha=0.3)
plt.show()

# 2. Fit linear regression to the data
x_data = bur_df['Duration (ms)'].values.reshape(-1, 1)
y_data = bur_df['Number of Photons'].values

# Create and fit the linear regression model
model = LinearRegression()
model.fit(x_data, y_data)
slope = model.coef_[0]
intercept = model.intercept_

# 3. Project points onto the regression line
# For each point (x_i, y_i), calculate the perpendicular projection
# onto the regression line y = slope*x + intercept
projected_points = np.zeros((len(x_data), 2))
for i, (x_i, y_i) in enumerate(zip(x_data.flatten(), y_data)):
    # Formula for perpendicular projection of (x_i, y_i) onto line y = mx + b
    x_proj = (x_i + slope*y_i - slope*intercept) / (1 + slope**2)
    y_proj = slope * x_proj + intercept
    projected_points[i] = [x_proj, y_proj]

# Calculate the distances along the regression line
# We can use the x-coordinates of the projected points as a measure of position
projections = projected_points[:, 0]  

# 4. Calculate z-scores of the projections
projection_z_scores = stats.zscore(projections)

# 5. Identify outliers based on z-scores (typically |z| > 3 is considered an outlier)
z_threshold = 3.0
is_outlier = np.abs(projection_z_scores) > z_threshold

# Add outlier flag to the dataframe
bur_df['is_outlier'] = is_outlier

# 6. Visualize the linear fit with outliers
plt.figure(figsize=(10, 6))

# Plot data points with outlier coloring
colors = ['blue' if not outlier else 'red' for outlier in is_outlier]
plt.scatter(x_data, y_data, c=colors, edgecolors='k')

# Plot the linear regression line
x_smooth = np.linspace(min(x_data.flatten()), max(x_data.flatten()), 1000)
y_smooth = slope * x_smooth + intercept
plt.plot(x_smooth, y_smooth, 'g-', linewidth=2, label='Linear Fit')

# Add the projection lines for outliers to visualize
for i, (x_i, y_i) in enumerate(zip(x_data.flatten(), y_data)):
    if is_outlier[i]:
        # Get the projection point calculated earlier
        x_proj, y_proj = projected_points[i]
        
        # Draw a line from the data point to its perpendicular projection
        plt.plot([x_i, x_proj], [y_i, y_proj], 'r--', alpha=0.5)

plt.xlabel('Duration (ms)')
plt.ylabel('Number of Photons')
plt.title('Linear Regression with Outlier Detection using Projection Z-Scores')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# 7. Print model information
y_pred = model.predict(x_data)
residuals = y_data - y_pred
rmse = np.sqrt(np.mean(residuals**2))

print(f"Linear model: y = {slope:.4f} * x + {intercept:.4f}")
print(f"RMSE: {rmse:.2f}")
print(f"Z-score threshold: {z_threshold}")
print(f"Number of outliers: {sum(is_outlier)} out of {len(is_outlier)}")

# 8. Show histogram of z-scores
plt.figure(figsize=(10, 6))
plt.hist(projection_z_scores, bins=30, alpha=0.7)
plt.axvline(z_threshold, color='r', linestyle='--', label=f'+{z_threshold} threshold')
plt.axvline(-z_threshold, color='r', linestyle='--', label=f'-{z_threshold} threshold')
plt.xlabel('Z-score of Projection')
plt.ylabel('Frequency')
plt.title('Histogram of Projection Z-Scores')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ruptures as rpt

# Ensure the data is sorted by "Duration (ms)"
bur_df = bur_df.sort_values(by="Duration (ms)").reset_index(drop=True)

# Prepare the signal for change point detection.
# We use both "Duration (ms)" and "Number of Photons" as a 2D signal.
signal = bur_df[['Duration (ms)', 'Number of Photons']].values

# Use the PELT algorithm with a linear model.
algo = rpt.Pelt(model="linear").fit(signal)
# Use a penalty value to detect change points. Adjust 'pen' as needed.
result = algo.predict(pen=10)
print("Detected change point indices:", result)

# The result list contains indices where each segment ends.
# We take the first change point as the cutoff.
change_point_index = result[0]
linear_cutoff = bur_df.loc[change_point_index - 1, "Duration (ms)"]
print("Linear region cutoff (Duration):", linear_cutoff)

# Visualization: Plot the full data with a vertical line at the detected change point.
plt.figure(figsize=(10, 6))
sns.scatterplot(data=bur_df, x="Duration (ms)", y="Number of Photons", label='Data')
plt.axvline(x=linear_cutoff, color='red', linestyle='--', label=f'Change point at {linear_cutoff:.2f}')
plt.title("Change Point Detection using ruptures")
plt.xlabel("Duration (ms)")
plt.ylabel("Number of Photons")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Filter the dataset to include only the points in the linear region.
linear_df = bur_df[bur_df["Duration (ms)"] <= linear_cutoff]
print(f"Number of points in linear region: {len(linear_df)} out of {len(bur_df)}")

# Visualize the filtered (linear) data.
plt.figure(figsize=(10, 6))
sns.scatterplot(data=linear_df, x="Duration (ms)", y="Number of Photons")
plt.xlabel("Duration (ms)")
plt.ylabel("Number of Photons")
plt.title("Filtered Data (Linear Region)")
plt.grid(True, alpha=0.3)
plt.show()

# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Assuming df is your DataFrame
# Calculate the ratio of green signal to red signal
bur_df['S_g/S_r'] = bur_df['Number of Photons (green)'] / bur_df['Duration (green) (ms)']

# Create the plot
plt.figure(figsize=(10, 6))

sns.scatterplot(x='Mean Macro Time (ms)', y='S_g/S_r', data=bur_df)
plt.xlabel('Macro Time')

plt.yscale('log')

plt.ylabel('S_g/S_r (log scale)')
plt.title('Ratio of Green Signal to Red Signal vs Macro Time (Log Scale)')
plt.grid(True, which="both", ls="-")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Filtered Data Graphs

# %%
# Filter data based on the previous outlier detection methods
# 1. Filter based on z-score cutoff for Time Difference (TGX-TRR)
time_diff_filter = abs(bur_df['z_score']) <= Z_SCORE_CUTOFF

# 2. Filter based on quadratic regression (not using ruptures)
quad_filter = ~bur_df['is_outlier']  # Use the 'is_outlier' flag from quadratic regression

# Combine filters
filtered_bur_df = bur_df[time_diff_filter & quad_filter].copy()

print(f"Original data points: {len(bur_df)}")
print(f"After time difference filter: {len(bur_df[time_diff_filter])}")
print(f"After quadratic regression filter: {len(bur_df[quad_filter])}")
print(f"After combined filtering: {len(filtered_bur_df)}")

# %% Graph 1: TGX-TRR for filtered data
plt.figure(figsize=(10, 6))
# Plot filtered points
plt.scatter(range(len(filtered_bur_df)), filtered_bur_df['Time Difference'], 
            color='purple', alpha=0.7)

plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
plt.axhline(y=upper_threshold, color='red', linestyle='--', alpha=0.7, 
            label=f'Upper threshold (z={Z_SCORE_CUTOFF}): {upper_threshold:.2f}')
plt.axhline(y=lower_threshold, color='red', linestyle='--', alpha=0.7,
            label=f'Lower threshold (z=-{Z_SCORE_CUTOFF}): {lower_threshold:.2f}')
            
plt.xlabel("Data Point Index")
plt.ylabel("Duration (green) (ms) - Duration (yellow) (ms)")
plt.title(f"Filtered: Difference Between Green and Yellow Burst Durations\nPoints with |z-score| <= {Z_SCORE_CUTOFF}")
plt.legend()
plt.tight_layout()
plt.show()

# Histogram for filtered data
plt.figure(figsize=(10, 6))
sns.histplot(filtered_bur_df['Time Difference'], kde=True, color='purple')

# Add vertical lines for z-score = ±Z_SCORE_CUTOFF thresholds
plt.axvline(x=upper_threshold, color='red', linestyle='--', 
            label=f'z-score = +{Z_SCORE_CUTOFF} (value = {upper_threshold:.2f})')
plt.axvline(x=lower_threshold, color='red', linestyle='--', 
            label=f'z-score = -{Z_SCORE_CUTOFF} (value = {lower_threshold:.2f})')

# Add zero line
plt.axvline(x=0, color='gray', linestyle='--', alpha=0.7)

plt.xlabel("Duration (green) (ms) - Duration (yellow) (ms)")
plt.ylabel("Frequency")
plt.title("Filtered: Distribution of Differences Between Green and Yellow Burst Durations")
plt.legend()
plt.tight_layout()
plt.show()

# %% Graph 2: Duration vs Number of Photons with quadratic fit for filtered data
# 1. Visualize the filtered data
plt.figure(figsize=(10, 6))
sns.scatterplot(filtered_bur_df, x="Duration (ms)", y="Number of Photons", color='blue')
plt.title("Filtered Data: Duration vs Number of Photons")
plt.grid(True, alpha=0.3)

# Refit quadratic regression to the filtered data
x_data_filtered = filtered_bur_df['Duration (ms)']
y_data_filtered = filtered_bur_df['Number of Photons']

# Fit the quadratic model on filtered data
params_filtered, covariance_filtered = optimize.curve_fit(quad_func, x_data_filtered, y_data_filtered)
a_filtered, b_filtered, c_filtered = params_filtered

# Plot the new quadratic regression curve
x_smooth = np.linspace(min(x_data_filtered), max(x_data_filtered), 1000)
y_smooth = quad_func(x_smooth, a_filtered, b_filtered, c_filtered)
plt.plot(x_smooth, y_smooth, 'g-', linewidth=2, label='Quadratic Fit (Filtered)')

plt.xlabel('Duration (ms)')
plt.ylabel('Number of Photons')
plt.legend()
plt.show()

# Print model information for filtered data
print(f"Filtered quadratic model: y = {a_filtered:.4f} * x² + {b_filtered:.4f} * x + {c_filtered:.4f}")
print(f"Original quadratic model: y = {a:.4f} * x² + {b:.4f} * x + {c:.4f}")

# %% Graph 3: Mean Macro Time vs S_g/S_r for filtered data
plt.figure(figsize=(10, 6))

sns.scatterplot(x='Mean Macro Time (ms)', y='S_g/S_r', data=filtered_bur_df, color='blue')
plt.xlabel('Macro Time')

plt.yscale('log')

plt.ylabel('S_g/S_r (log scale)')
plt.title('Filtered: Ratio of Green Signal to Red Signal vs Macro Time (Log Scale)')
plt.grid(True, which="both", ls="-")
plt.tight_layout()
plt.show()

# Add comparison statistics
print("\nComparison Statistics:")
print(f"Original data - Mean S_g/S_r: {bur_df['S_g/S_r'].mean():.4f}, Std: {bur_df['S_g/S_r'].std():.4f}")
print(f"Filtered data - Mean S_g/S_r: {filtered_bur_df['S_g/S_r'].mean():.4f}, Std: {filtered_bur_df['S_g/S_r'].std():.4f}")


