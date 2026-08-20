import numpy as np
import h5py
import warnings

from scipy.ndimage import distance_transform_edt


def check_qfsp_flag(slc_path):
    """
    Check whether an RSLC reports an input-data exception.

    Older RSLC products may not contain the
    `hasInputDataException` dataset. Such products are treated as not
    reporting an input-data exception.

    Parameters
    ----------
    slc_path : path-like
        Path to the input RSLC HDF5 product.

    Returns
    -------
    bool
        ``True`` if the product reports an input-data exception; otherwise
        ``False``.

    Warns
    -----
    RuntimeWarning
        If the RSLC does not contain the expected dataset.
    """
    qfsp_path = (
        "/science/LSAR/identification/"
        "hasInputDataException"
    )

    with h5py.File(slc_path, "r") as src:
        if qfsp_path not in src:
            warnings.warn(
                f"{slc_path} does not contain {qfsp_path}; "
                "assuming no input-data exception.",
                RuntimeWarning,
                stacklevel=2,
            )
            return False

        value = src[qfsp_path][()]

    return bool(np.asarray(value).item())


def _fit_2d_polynomial_surface(data, valid_mask, order=2):
    """
    Fit a two-dimensional polynomial surface to valid data samples.

    The pixel coordinates are normalized using the mean and standard
    deviation of the valid samples before the least-squares fit. Polynomial
    terms up to cubic order are supported.

    Parameters
    ----------
    data : numpy.ndarray
        Two-dimensional array containing the values to fit.
    valid_mask : numpy.ndarray
        Boolean array with the same shape as `data`. Pixels marked as
        ``True`` are eligible for use in the polynomial fit. Non-finite
        values in `data` are excluded regardless of this mask.
    order : int, optional
        Polynomial order. Supported values are 0 through 3, corresponding
        to constant, linear, quadratic, and cubic surfaces, respectively.
        Defaults to 2.

    Returns
    -------
    surface : numpy.ndarray
        Two-dimensional fitted polynomial surface with the same shape as
        `data`.
    """
    if not isinstance(order, (int, np.integer)) or isinstance(order, bool):
        raise TypeError("order must be an integer")

    if not 0 <= order <= 3:
        raise ValueError("order must be between 0 and 3")
    nrows, ncols = data.shape
    yy, xx = np.indices((nrows, ncols))

    good = valid_mask & np.isfinite(data)
    if np.count_nonzero(good) < 10:
        raise ValueError("Not enough valid pixels for 2D polynomial fit.")

    x = xx[good].astype(float)
    y = yy[good].astype(float)
    z = data[good].astype(float)

    x_mean = x.mean()
    y_mean = y.mean()
    x_std = x.std() if x.std() > 0 else 1.0
    y_std = y.std() if y.std() > 0 else 1.0

    xn = (x - x_mean) / x_std
    yn = (y - y_mean) / y_std

    terms = [np.ones_like(xn)]
    if order >= 1:
        terms += [xn, yn]
    if order >= 2:
        terms += [xn**2, xn * yn, yn**2]
    if order >= 3:
        terms += [xn**3, (xn**2) * yn, xn * (yn**2), yn**3]

    A = np.vstack(terms).T
    coeff, _, _, _ = np.linalg.lstsq(A, z, rcond=None)

    yy_full, xx_full = np.indices((nrows, ncols))
    xn_full = (xx_full.astype(float) - x_mean) / x_std
    yn_full = (yy_full.astype(float) - y_mean) / y_std

    full_terms = [np.ones_like(xn_full)]
    if order >= 1:
        full_terms += [xn_full, yn_full]
    if order >= 2:
        full_terms += [xn_full**2, xn_full * yn_full, yn_full**2]
    if order >= 3:
        full_terms += [xn_full**3, (xn_full**2) * yn_full, xn_full * (yn_full**2), yn_full**3]

    surface = np.zeros((nrows, ncols), dtype=float)
    for c, t in zip(coeff, full_terms):
        surface += c * t

    return surface


def _find_column_groups(artifact_mask, min_fraction_rows=0.3, min_group_width=2):
    """
    Find contiguous column groups affected by an artifact.

    A column is considered affected when the fraction of artifact pixels
    along its rows is greater than or equal to `min_fraction_rows`.
    Contiguous affected columns are grouped, and groups narrower than
    `min_group_width` are discarded.

    Parameters
    ----------
    artifact_mask : numpy.ndarray
        Two-dimensional boolean array in which ``True`` indicates an
        artifact-affected pixel.
    min_fraction_rows : float, optional
        Minimum fraction of rows containing artifact pixels required for
        a column to be considered affected. Expected to be between 0 and 1.
        Defaults to 0.3.
    min_group_width : int, optional
        Minimum number of contiguous affected columns required to retain
        a group. Defaults to 2.

    Returns
    -------
    groups : list of tuple of int
        List of ``(start, end)`` column-index pairs. Each interval follows
        Python slicing convention: `start` is inclusive and `end` is
        exclusive.
    """
    nrows, ncols = artifact_mask.shape
    frac = artifact_mask.sum(axis=0) / max(nrows, 1)
    affected_cols = frac >= min_fraction_rows

    groups = []
    in_group = False
    start = None

    for j in range(ncols):
        if affected_cols[j] and not in_group:
            start = j
            in_group = True
        elif not affected_cols[j] and in_group:
            end = j
            if end - start >= min_group_width:
                groups.append((start, end))
            in_group = False

    if in_group:
        end = ncols
        if end - start >= min_group_width:
            groups.append((start, end))

    return groups


def _moving_average_1d(x, win):
    """
    Apply a NaN-aware moving average to a one-dimensional array.

    The moving average is computed using only finite samples within each
    window. Non-finite samples do not contribute to either the sum or the
    number of samples used to calculate the average.

    Parameters
    ----------
    x : array_like
        One-dimensional input data.
    win : int
        Size of the moving-average window. If `win` is less than or equal
        to 1, a copy of the input array is returned without smoothing.

    Returns
    -------
    out : numpy.ndarray
        Smoothed floating-point array with the same shape as `x`. Output
        samples are NaN where the corresponding window contains no finite
        input samples.
    """
    x = np.asarray(x, dtype=float)
    if win <= 1:
        return x.copy()

    valid = np.isfinite(x).astype(float)
    x0 = np.where(np.isfinite(x), x, 0.0)
    kernel = np.ones(win, dtype=float)

    num = np.convolve(x0, kernel, mode="same")
    den = np.convolve(valid, kernel, mode="same")

    out = np.full_like(x, np.nan, dtype=float)
    ok = den > 0
    out[ok] = num[ok] / den[ok]
    return out


def make_feather_weight(
    artifact_mask,
    inner_shrink=2,
    outer_feather=10,
):
    """
    Create a smooth feathering weight for artifact correction.

    Parameters
    ----------
    artifact_mask : 2D bool array
        True where the artifact correction should be strongest.

    inner_shrink : int
        Pixels inside the artifact edge where weight begins tapering.
        Larger values make the full-strength region smaller.

    outer_feather : int
        Number of pixels outside artifact_mask over which correction tapers to 0.

    Returns
    -------
    weight : 2D float array
        Smooth weight from 0 to 1.
    """
    artifact_mask = np.asarray(artifact_mask, dtype=bool)

    if inner_shrink < 0:
        raise ValueError("inner_shrink must be non-negative")

    if outer_feather < 0:
        raise ValueError("outer_feather must be non-negative")

    if inner_shrink == 0 and outer_feather == 0:
        return artifact_mask.astype(np.float32)

    dist_inside = distance_transform_edt(artifact_mask)
    dist_outside = distance_transform_edt(~artifact_mask)

    # Positive inside the artifact and negative outside.
    signed_distance = dist_inside - dist_outside

    transition_width = inner_shrink + outer_feather

    weight = (
        signed_distance + outer_feather
    ) / float(transition_width)

    weight = np.clip(weight, 0.0, 1.0)

    # Smooth the linear feather weight using a cubic polynomial
    #
    #     S(w) = a*w**3 + b*w**2 + c*w + d.
    #
    # Requiring S(0)=0, S(1)=1, and zero endpoint slopes,
    # S'(0)=S'(1)=0, gives a=-2, b=3, and c=d=0. Thus,
    #
    #     S(w) = 3*w**2 - 2*w**3 = w**2 * (3 - 2*w).
    #
    # The zero endpoint slopes reduce correction-gradient discontinuities
    # at the feather boundaries.
    weight = weight * weight * (3.0 - 2.0 * weight)

    return weight.astype(np.float32)


def correct_qfsp_phase_artifact(
    phase,
    fit_background_mask,
    artifact_mask,
    background_order=2,
    min_fraction_rows=0.3,
    min_group_width=2,
    template_smooth_win=3,
    inner_shrink=2,
    outer_feather=10,
    fill_value=np.nan,
    ):
    """
    Estimate and remove a range-dependent qFSP phase artifact.

    A smooth two-dimensional background surface is first fitted to valid,
    non-artifact pixels and subtracted from the input phase. For each
    contiguous group of artifact-affected range columns, a one-dimensional
    artifact template is estimated from the residual phase by averaging over
    rows. The template is optionally smoothed, extended beyond the detected
    artifact columns, replicated along the row direction, and subtracted
    from the original phase using a feathered correction weight.

    Parameters
    ----------
    phase : numpy.ndarray
        Two-dimensional differential interferogram phase array with shape
        ``(nrows, ncols)``. Non-finite values and zero-valued pixels are
        treated as invalid and are excluded from background fitting and
        template estimation.

    fit_background_mask : numpy.ndarray
        Boolean array with the same shape as ``phase`` identifying pixels
        available for background fitting and artifact-template estimation.
        Artifact pixels do not need to be removed from this mask: they are
        excluded internally from the background fit using ``artifact_mask``
        but retained for template estimation. Pixels set to ``False`` are
        excluded from both operations.

    artifact_mask : numpy.ndarray
        Boolean array with the same shape as ``phase``. Pixels set to
        ``True`` identify locations affected by the qFSP artifact. The mask
        is reduced to affected range-column groups using
        ``min_fraction_rows`` and ``min_group_width``.

    background_order : int, default=2
        Polynomial order of the two-dimensional surface fitted to the valid,
        non-artifact phase pixels. For example, 0 fits a constant, 1 fits a
        planar surface, and 2 includes quadratic terms.

    min_fraction_rows : float, default=0.3
        Minimum fraction of all rows that must be marked as artifact-affected
        for a range column to be included in an artifact group. This value
        should normally be in the interval ``[0, 1]``.

    min_group_width : int, default=2
        Minimum number of consecutive affected range columns required to
        retain an artifact group. Groups narrower than this value are
        ignored.

    template_smooth_win : int, default=3
        Window length, in range pixels, used to smooth each one-dimensional
        artifact template. Values less than or equal to 1 disable template
        smoothing.

    inner_shrink : int, default=2
        Width, in pixels, of the transition applied inside the artifact-mask
        boundary by ``make_feather_weight``. This reduces the correction
        strength near the inner boundary of the detected artifact region.

    outer_feather : int, default=10
        Number of pixels by which the artifact model and correction region
        are extended outside each detected artifact group. The correction
        weight tapers across this region to reduce boundary discontinuities.

    fill_value : float, default=numpy.nan
        Value used to initialize ``artifact_2d`` at pixels where no artifact
        model is estimated. Non-finite fill values prevent correction at
        those pixels through ``valid_corr``.

    Returns
    -------
    result : dict
        Dictionary containing the following entries:

        ``"corrected_phase"`` : numpy.ndarray
            Copy of the input phase after subtracting the weighted artifact
            model where a valid correction is available.

        ``"artifact_2d"`` : numpy.ndarray
            Estimated two-dimensional artifact model. Each one-dimensional
            range template is replicated along the row direction.

        ``"background"`` : numpy.ndarray
            Fitted two-dimensional polynomial background surface.

        ``"residual"`` : numpy.ndarray
            Detrended phase, computed as ``phase - background``.

        ``"fit_mask"`` : numpy.ndarray
            Boolean mask of valid, non-artifact pixels used for background
            fitting.

        ``"template_estimation_mask"`` : numpy.ndarray
            Boolean mask of valid artifact pixels used to estimate the
            one-dimensional templates.

        ``"groups"`` : list of tuple of int
            Detected artifact-column intervals represented as ``(c0, c1)``.
            Each interval follows Python slicing convention: ``c0`` is
            included and ``c1`` is excluded.

        ``"templates"`` : list of numpy.ndarray
            One-dimensional residual-phase template estimated for each
            detected artifact group. A template may contain non-finite
            values where insufficient valid samples are available.

        ``"correction_weight"`` : numpy.ndarray
            Two-dimensional feather weight applied to the artifact model.

        ``"valid_corr"`` : numpy.ndarray
            Boolean mask identifying pixels where the correction was
            actually applied.
    """
    phase = np.asarray(phase, dtype=float)
    fit_background_mask = np.asarray(fit_background_mask, dtype=bool)
    artifact_mask = np.asarray(artifact_mask, dtype=bool)

    if phase.ndim != 2:
        raise ValueError("phase must be 2D")

    nrows, ncols = phase.shape

    for name, arr in [
        ("fit_background_mask", fit_background_mask),
        ("artifact_mask", artifact_mask),
    ]:
        if arr.shape != phase.shape:
            raise ValueError(f"{name} must have same shape as phase")

    valid_phase = np.isfinite(phase) & (phase != 0)

    # Pixels used to fit the background: clean, valid, non-artifact pixels.
    fit_mask = (
        fit_background_mask
        & (~artifact_mask)
        & valid_phase
    )

    # Pixels used to estimate the artifact template: valid artifact pixels.
    template_estimation_mask = (
        artifact_mask
        & valid_phase
        & fit_background_mask
    )

    background = _fit_2d_polynomial_surface(
        phase,
        fit_mask,
        order=background_order,
    )

    residual = phase - background

    groups = _find_column_groups(
        artifact_mask,
        min_fraction_rows=min_fraction_rows,
        min_group_width=min_group_width,
    )

    artifact_2d = np.full_like(phase, fill_value, dtype=float)
    templates = []

    for c0, c1 in groups:
        group_residual = residual[:, c0:c1]

        valid_for_template = (
            template_estimation_mask[:, c0:c1]
            & np.isfinite(group_residual)
        )

        tmp = np.where(valid_for_template, group_residual, np.nan)
        template = np.nanmean(tmp, axis=0)

        if np.all(~np.isfinite(template)):
            templates.append(template)
            continue

        if template_smooth_win > 1:
            template = _moving_average_1d(template, template_smooth_win)

        templates.append(template)

        # Extend the artifact model slightly outside the artifact columns
        # so feathering can smoothly taper the correction.
        c0_ext = max(0, c0 - outer_feather)
        c1_ext = min(ncols, c1 + outer_feather)

        template_ext = np.interp(
            np.arange(c0_ext, c1_ext),
            np.arange(c0, c1),
            template,
            left=template[0],
            right=template[-1],
        )

        artifact_group_ext = np.tile(template_ext[None, :], (nrows, 1))

        valid_ext = np.isfinite(artifact_group_ext)
        artifact_2d[:, c0_ext:c1_ext][valid_ext] = (
            artifact_group_ext[valid_ext]
        )

    correction_weight = make_feather_weight(
        artifact_mask,
        inner_shrink=inner_shrink,
        outer_feather=outer_feather,
    )

    corrected_phase = phase.copy()

    valid_corr = (
        (correction_weight > 0)
        & np.isfinite(artifact_2d)
        & np.isfinite(corrected_phase)
    )

    corrected_phase[valid_corr] -= (
        correction_weight[valid_corr] * artifact_2d[valid_corr]
    )

    return {
        "corrected_phase": corrected_phase,
        "artifact_2d": artifact_2d,
        "background": background,
        "residual": residual,
        "fit_mask": fit_mask,
        "template_estimation_mask": template_estimation_mask,
        "groups": groups,
        "templates": templates,
        "correction_weight": correction_weight,
        "valid_corr": valid_corr,
    }
