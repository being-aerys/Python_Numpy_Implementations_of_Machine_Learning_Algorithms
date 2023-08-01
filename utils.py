import numpy as np


def minmax_normalize_2d_array(arr, min_val=0, max_val=1):
    # Calculate the minimum and maximum values along the columns (axis=0)
    mins = np.min(arr, axis=0)
    maxs = np.max(arr, axis=0)

    # Avoid division by zero by adding a small value (epsilon)
    epsilon = 1e-8

    # Normalize each column
    normalized_arr = min_val + (arr - mins) * (max_val - min_val) / (
        maxs - mins + epsilon
    )

    return normalized_arr
