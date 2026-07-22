import numpy as np
import h5py
from scipy.ndimage import distance_transform_edt


def check_qfsp_flag(slc_path):
    qFSP_path = '/science/LSAR/identification/hasInputDataException/'
    with h5py.File(slc_path) as src:
        qfsp_flag = src[qFSP_path][()]
    return qfsp_flag


def _fit_2d_polynomial_surface(data, valid_mask, order=2):
    """
    Fit a global 2D polynomial surface to data using valid_mask pixels only.
    """
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
    Find contiguous column groups affected in many rows based on artifact_mask.
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
    NaN-aware moving average for 1D array.
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

    # Distance inside artifact region
    dist_inside = distance_transform_edt(artifact_mask)

    # Distance outside artifact region
    dist_outside = distance_transform_edt(~artifact_mask)

    weight = np.zeros(artifact_mask.shape, dtype=np.float32)

    # Inside artifact area: mostly full correction
    inside = artifact_mask
    weight[inside] = 1.0

    # Taper near inner boundary
    if inner_shrink > 0:
        inner_edge = inside & (dist_inside <= inner_shrink)
        weight[inner_edge] = dist_inside[inner_edge] / float(inner_shrink)

    # Outside artifact area: apply small tapered correction into neighborhood
    if outer_feather > 0:
        outside_feather = (~artifact_mask) & (dist_outside <= outer_feather)
        weight[outside_feather] = (
            1.0 - dist_outside[outside_feather] / float(outer_feather)
        )

    # Smoothstep curve: smoother transition than linear
    weight = weight * weight * (3.0 - 2.0 * weight)

    return weight


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
    Detrend phase, estimate smooth qFSP artifact from artifact pixels,
    then subtract the smooth artifact from the original phase.

    Parameters
    ----------
    phase : np.ndarray
        2D differential interferogram phase.

    fit_background_mask : np.ndarray
        Valid clean pixels used to estimate the smooth background.
        This should usually exclude artifact_mask.

    artifact_mask : np.ndarray
        Pixels affected by qFSP artifact. The artifact template is estimated
        from these pixels, and correction is applied to these pixels with
        feathering near the boundary.
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
