import numpy as np

def decimate_freq_a_array(
        slant_main,
        slant_side,
        target_runw):
    """decimate target_runw of main band to have same size with side band
    assuming slant_main and slant_side are evenly spaced

    Parameters
    ----------
    slant_main : numpy.ndarray
        slant range array of frequency A band
    slant_side : numpy.ndarray
        slant range array of frequency B band
    target_runw : numpy.ndarray
        RUNW array of frequency A band
        width of target_runw should be same with length of slant_main

    Returns
    -------
    decimated_array : numpy.ndarray
        decimated RUNW array
    """
    _, width = target_runw.shape

    first_index = np.argmin(np.abs(slant_main - slant_side[0]))
    spacing_main = slant_main[1] - slant_main[0]
    spacing_side = slant_side[1] - slant_side[0]

    resampling_scale_factor = int(np.round(spacing_side / spacing_main))

    # ensure the decimated width matches len(slant_side)
    n_side = len(slant_side)
    decimate_width_end = first_index + n_side * resampling_scale_factor
    if decimate_width_end > width:
        # shift start left so it fits; clamp to 0
        first_index = max(0, width - n_side * resampling_scale_factor)
        decimate_width_end = first_index + n_side * resampling_scale_factor
    decimated_array = target_runw[
        :, first_index:decimate_width_end:resampling_scale_factor]

    if decimated_array.shape[1] < n_side:
        pad_cols = n_side - decimated_array.shape[1]
        pad = np.zeros((decimated_array.shape[0], pad_cols),
                       dtype=decimated_array.dtype)
        decimated_array = np.concatenate([decimated_array, pad], axis=1)
    return decimated_array

def interpolate_freq_b_array(
        slant_main,
        slant_side,
        array_side):
    """interpolate array that have the size of side band (frequency B)
    to have same size with main band assuming slant_main and slant_side
    are evenly spaced

    Parameters
    ----------
    slant_main : numpy.ndarray
        slant range array of frequency A band
    slant_side : numpy.ndarray
        slant range array of frequency B band
    array_side : numpy.ndarray
        array with same size of side-band (frequencyB)
        width of array_side should be same with length of slant_side

    Returns
    -------
    array_main : numpy.ndarray
        oversampled array
    """
    row_side, _ = array_side.shape
    array_main = np.zeros([row_side, len(slant_main)])

    for row_ind in range(0, row_side):

        array_main[row_ind, :] = np.interp(slant_main,
                                           slant_side,
                                           array_side[row_ind, :])

    return array_main
