import numpy as np


def minmax_normalize_2d_array(arr, min_val=0, max_val=1):
    """
    Perform Min-Max normalization on a 2D NumPy array.

    The Min-Max normalization scales the input array to a new range defined by the provided minimum and maximum values.
    The normalization is done column-wise (along axis 0).

    Parameters:
        arr (numpy.ndarray): The input 2D NumPy array to be normalized.
        min_val (float, optional): The minimum value of the new range. Default is 0.
        max_val (float, optional): The maximum value of the new range. Default is 1.

    Returns:
        numpy.ndarray: The normalized 2D NumPy array with values in the range [min_val, max_val].
    """
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


def calculate_accuracy(actual, predicted):
    """
    Function to calculate prediction accuracy.

    Parameters:
        actual (list or numpy array): The actual ground truth values.
        predicted (list or numpy array): The predicted values.

    Returns:
        float: Prediction accuracy as a percentage.
    """
    # Ensure that both actual and predicted are of the same length
    if len(actual) != len(predicted):
        raise ValueError("Both actual and predicted should be of the same length.")

    # Count the number of correct predictions
    correct_predictions = sum(
        1 for true, pred in zip(actual, predicted) if true == pred
    )

    # Calculate the prediction accuracy
    accuracy = correct_predictions / len(actual) * 100.0

    return accuracy


# sample ground truth values and predicted values
actual_values = [1, 0, 1, 1, 0, 1, 0, 1]
predicted_values = [1, 1, 1, 0, 0, 1, 0, 1]

# calculate prediction accuracy
accuracy = calculate_accuracy(actual_values, predicted_values)
print("Prediction Accuracy: {:.2f}%".format(accuracy))
