import numpy.ma as ma

### define function for extracting the unmasked segments from the thresholded data.
def extract_greater(data, threshold_value):
    filtered_values = ma.masked_greater(data, threshold_value)
    burst_index = extract_unmasked_indices(filtered_values)
    return burst_index, filtered_values

def extract_unmasked_indices(masked_array):
    unmasked_indices_lists = []
    current_indices = []

    # iterate through masked array and collect unmasked index segments
    for i, value in enumerate(masked_array):
        if ma.is_masked(value):
            if current_indices:
                unmasked_indices_lists.append(current_indices)
                current_indices = []
        else:
            current_indices.append(i)

    # handle the last segment
    if current_indices:
        unmasked_indices_lists.append(current_indices)

    return unmasked_indices_lists