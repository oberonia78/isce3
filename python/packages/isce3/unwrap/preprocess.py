import pathlib
import journal
import numpy as np
from osgeo import gdal, osr

from scipy.ndimage import median_filter, map_coordinates


def preprocess_wrapped_igram(igram, coherence, mask=None,
                             mask_type='coherence', threshold=0.5,
                             filter_size=9,
                             filling_enabled=True,
                             filling_method='distance_interpolator',
                             distance=5):
    '''
    Preprocess wrapped interferograms prior to phase unwrapping.

    Removes invalid pixels in wrapped interferograms based on
    user-defined metric. Invalid pixels may be identified using
    1) a water mask; 2) thresholding low-coherence pixels; 3) thresholding
    the median absolute deviation of the interferogram phase from the local median.
    Invalid pixels are replaced with values computed with a distance-weighted
    interpolation approach from Chen et al., 2015. The magnitude of the complex
    interferogram is discarded.

    Parameters
    ----------
    igram: numpy.ndarray
        Wrapped interferogram to pre-process
    coherence: numpy.ndarray
        Normalized InSAR coherence
    mask: numpy.ndarray or None
        Optional binary mask (1: invalid; 0: valid) to identify invalid pixels.
        If a mask is provided, data-driven masking is not performed (other
        masking options are ignored;
    mask_type: str, {'median_filter', 'coherence', 'water'}, optional
        Type of mask to identify invalid pixels
        'median_filter':
        Compute mask of invalid pixels by thresholding the median absolute
        deviation w.r.t. the local neighborhood around each pixel.

        'coherence':
        The default mode. Compute mask of invalid pixels by thresholding
        the normalized InSAR coherence.

        'water':
        Project the water mask to radar grid and masks out the invalid
        pixels

    threshold: float
        Threshold to identify invalid pixels.
        If 'mask_type' is 'coherence' pixels with coherence below threshold
        are considered invalid
        If 'mask_type' is 'median_filter' pixels with median absolute
        deviation (MAD) above this threshold are considered outliers
    filter_size: int
        Size of median filter for median absolute deviation
        outlier identification method
    filling_method: str
        Algorithm to fill invalid pixels. 'distance_interpolator'
        applies distance weighted interpolation from Chen et al., 2015
    distance: int
        Distance metric for interpolation. For distance interpolator in
        Chen et al [1]_ is distance is intended as radius

    Returns
    -------
    filt_igram: numpy.ndarray
        Wrapped interferogram with outlier pixel being filtered
        out and replaced with pixels computed by the selected
        'filling_method'. The magnitude of the input wrapped interferogram
        is discarded.

    References
    ----------
    .. [1] J. Chen, H. A. Zebker,and R. Knight, ""A persistent scatterer interpolation
       for retrieving accurate ground deformation over InSAR-decorrelated
       Agricultural fields", Geoph. Res. Lett., 42(21), 9294-9301, (2015).
    '''

    # Extract some preprocess options
    error_channel = journal.error('unwrap.run.preprocess_wrapped_igram')

    # Create mask of invalid pixels
    invalid_mask = np.full(igram.shape, dtype=bool, fill_value=False)

    # Identify invalid pixels and store them in a mask.
    # Criteria to identify invalid pixels:
    # 1-1) Based on user-provided mask
    # 1-2) Based on water mask
    if mask is not None:
        invalid_mask[mask == 1] = True
    # 2) Based on InSAR correlation values
    elif mask_type == 'coherence':
        invalid_mask[coherence < threshold] = True
    # 3) Based on median absolute deviation (MAD)
    elif mask_type == 'median_filter':
        igram_pha = np.angle(igram)
        mad = median_absolute_deviation(igram_pha, filter_size)
        invalid_mask[mad > threshold] = True
    # Not a valid algorithm to mask pixels
    else:
        err_str = f"{mask_type} is an invalid selection for mask_type"
        error_channel.log(err_str)
        raise ValueError(err_str)

    if filling_enabled:
        # Fill invalid interferogram pixels using user-defined algorithm
        # Distance-based interpolator Chen et al. _[1]
        if filling_method == 'distance_interpolator':
            phase_filt = distance_interpolator(np.angle(igram), distance,
                                            invalid_mask)
        else:
            err_str = f"{filling_method} is an invalid selection for filling_method"
            error_channel.log(err_str)
            raise ValueError(err_str)
    else:
        igram[invalid_mask==1] = 0
        phase_filt = np.angle(igram)
    # Go to complex value
    igram_filt = np.exp(1j * phase_filt)

    return igram_filt


def distance_interpolator(arr, radius, invalid_mask):
    '''
    Interpolate pixels based on distance from valid pixels
    following Chen et al [1]_.

    Parameters
    ----------
    arr: numpy.ndarray
        Array containing invalid pixel locations to fill
    radius: int
        Radius of the sampling/filling window
    invalid_mask: numpy.ndarray
        Boolean mask identifying invalid pixels (True:invalid)

    Returns
    -------
    fill_arr: numpy.ndarray
        Array with interpolated values at invalid pixel locations

    References
    __________
    .. [1] J. Chen, H. A. Zebker,and R. Knight, ""A persistent scatterer interpolation
       for retrieving accurate ground deformation over InSAR-decorrelated
       Agricultural fields", Geoph. Res. Lett., 42(21), 9294-9301, (2015).
    '''
    arr_filt = np.copy(arr)

    # Get center locations
    x_cent, y_cent = np.where(invalid_mask == True)

    # Find the coordinates of valid pixels
    x, y = np.where(invalid_mask == False)

    for xc, yc in zip(x_cent, y_cent):
        # Compute distance between center pixel and valid pixels
        ps_dist = np.sqrt((x - xc) ** 2 + (y - yc) ** 2)
        # Compute weights based on distance and selected radius
        w = np.exp(-ps_dist ** 2 / 2 * radius)
        # Compute Eq. 2 of Chen at al [1]_
        weighted_arr = arr_filt[x, y].flatten() * w
        arr_filt[xc, yc] = np.nansum(weighted_arr) / np.nansum(w)

    return arr_filt


def median_absolute_deviation(arr, filter_size):
    '''
    Compute the median absolute deviation (MAD) of `arr`
    defined as median(abs(arr - median(arr))

    Parameters
    ----------
    arr: numpy.ndarray
        Array for which to compute MAD
    filter_size: int
        Size of median filter, in pixels

    Returns
    -------
    mad: numpy.ndarray
        Median absolute deviation of `arr`
    '''
    med = np.abs(arr - median_filter(arr, [filter_size, filter_size]))
    mad = median_filter(med, [filter_size, filter_size])
    return mad


def _gdal_type_to_np_type_str(gd_type):
    '''
    Convenience function to convert GDAL data type to numpy data type string
    '''
    gdal_type_to_np_dict = {1: "int8",
                            2: "uint16",
                            3: "int16",
                            4: "uint32",
                            5: "int32",
                            6: "float32",
                            7: "float64",
                            10: "complex64",
                            11: "complex128",}
    return gdal_type_to_np_dict[gd_type]


def _get_gdal_raster_shape_type(raster_path):
    '''
    Convenience function to get shape and numpy data type of GDAL-openable
    raster
    '''
    data_raster = gdal.Open(raster_path)

    data_shape = [data_raster.RasterYSize, data_raster.RasterXSize]

    data_band = data_raster.GetRasterBand(1)
    data_type = data_band.DataType
    np_data_type = _gdal_type_to_np_type_str(data_type)

    return data_shape, np_data_type


def _transform_bbox_epsg(bbox, src_epsg, dst_epsg):
    """
    Transform bbox from source EPSG to destination EPSG.

    Parameters
    ----------
    bbox : list or tuple
        [xmin, ymin, xmax, ymax]
    src_epsg : int
        EPSG of input bbox
    dst_epsg : int
        EPSG of output bbox

    Returns
    -------
    list
        Transformed bbox [xmin, ymin, xmax, ymax]
    """
    xmin, ymin, xmax, ymax = bbox

    src_srs = osr.SpatialReference()
    src_srs.ImportFromEPSG(int(src_epsg))
    src_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

    dst_srs = osr.SpatialReference()
    dst_srs.ImportFromEPSG(int(dst_epsg))
    dst_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

    transformer = osr.CoordinateTransformation(src_srs, dst_srs)

    corners_lonlat = [
        (xmin, ymin),
        (xmin, ymax),
        (xmax, ymin),
        (xmax, ymax),
    ]

    transformed = [transformer.TransformPoint(x, y) for x, y in corners_lonlat]

    xs = [p[0] for p in transformed]
    ys = [p[1] for p in transformed]

    return [min(xs), min(ys), max(xs), max(ys)]


def _read_gdal_with_bbox(input_raster, bbox, bbox_epsg=4326):
    """
    Read only the raster subset intersecting the input bbox.

    Parameters
    ----------
    input_raster : gdal.Dataset
        Input GDAL raster
    bbox : list[float]
        [xmin, ymin, xmax, ymax]
    bbox_epsg : int
        EPSG of bbox coordinates

    Returns
    -------
    arr : numpy.ndarray
        Raster subset array
    raster_info : list[float]
        [block_x0, block_y0, block_dx, block_dy]
    """
    gt = input_raster.GetGeoTransform()
    proj = input_raster.GetProjection()
    band = input_raster.GetRasterBand(1)

    if band is None:
        raise RuntimeError("Failed to access raster band.")

    if gt is None:
        raise RuntimeError("Raster geotransform is missing.")

    # north-up only, same practical assumption as most GeoTIFF use cases here
    if gt[2] != 0 or gt[4] != 0:
        raise NotImplementedError(
            "_read_gdal_with_bbox currently supports only north-up rasters."
        )

    # get raster EPSG directly here, without introducing a helper
    raster_srs = osr.SpatialReference()
    raster_srs.ImportFromWkt(proj)

    try:
        raster_srs.AutoIdentifyEPSG()
    except Exception:
        pass

    raster_epsg = raster_srs.GetAuthorityCode(None)
    if raster_epsg is None:
        raise RuntimeError("Could not determine raster EPSG from projection.")
    raster_epsg = int(raster_epsg)

    # transform bbox to raster CRS only if needed
    xmin, ymin, xmax, ymax = bbox
    if bbox_epsg != raster_epsg:
        src_srs = osr.SpatialReference()
        src_srs.ImportFromEPSG(int(bbox_epsg))
        dst_srs = osr.SpatialReference()
        dst_srs.ImportFromEPSG(int(raster_epsg))

        try:
            src_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
            dst_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        except Exception:
            pass

        tx = osr.CoordinateTransformation(src_srs, dst_srs)

        corners = [
            tx.TransformPoint(float(xmin), float(ymin))[:2],
            tx.TransformPoint(float(xmin), float(ymax))[:2],
            tx.TransformPoint(float(xmax), float(ymin))[:2],
            tx.TransformPoint(float(xmax), float(ymax))[:2],
        ]
        xs = [p[0] for p in corners]
        ys = [p[1] for p in corners]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)

    x0 = gt[0]
    dx = gt[1]
    y0 = gt[3]
    dy = gt[5]  # usually negative

    raster_width = input_raster.RasterXSize
    raster_height = input_raster.RasterYSize

    raster_xmin = min(x0, x0 + raster_width * dx)
    raster_xmax = max(x0, x0 + raster_width * dx)
    raster_ymin = min(y0, y0 + raster_height * dy)
    raster_ymax = max(y0, y0 + raster_height * dy)

    # intersect requested bbox with raster bounds
    xmin_i = max(xmin, raster_xmin)
    xmax_i = min(xmax, raster_xmax)
    ymin_i = max(ymin, raster_ymin)
    ymax_i = min(ymax, raster_ymax)

    if xmin_i >= xmax_i or ymin_i >= ymax_i:
        raise ValueError("Input bbox does not overlap raster.")

    # convert bbox to pixel window
    col0 = int(np.floor((xmin_i - x0) / dx))
    col1 = int(np.ceil((xmax_i - x0) / dx))

    row_a = int(np.floor((ymin_i - y0) / dy))
    row_b = int(np.ceil((ymax_i - y0) / dy))
    row_c = int(np.floor((ymax_i - y0) / dy))
    row_d = int(np.ceil((ymin_i - y0) / dy))
    row0 = min(row_a, row_b, row_c, row_d)
    row1 = max(row_a, row_b, row_c, row_d)

    # optional small padding for nearest-neighbor safety
    col0 -= 1
    row0 -= 1
    col1 += 1
    row1 += 1

    # clip to raster
    col0 = max(0, col0)
    row0 = max(0, row0)
    col1 = min(raster_width, col1)
    row1 = min(raster_height, row1)

    win_x = col1 - col0
    win_y = row1 - row0

    if win_x <= 0 or win_y <= 0:
        raise ValueError("Computed raster window is empty.")

    arr = band.ReadAsArray(col0, row0, win_x, win_y)
    if arr is None:
        raise RuntimeError("Failed to read raster window.")

    block_x0 = x0 + col0 * dx
    block_y0 = y0 + row0 * dy

    return arr, [block_x0, block_y0, dx, dy]


def _find_rdr2geo_paths(scratch_path, freq):
    """
    Find x.rdr and y.rdr files for the given frequency inside scratch_path.

    The function searches recursively because the files may exist in several
    possible directories such as:

        scratch/rdr2geo/freqA/
        scratch/ionosphere/main_diff_ms_band/rdr2geo/freqB/
        scratch/ionosphere/main_side_band/rdr2geo/freqB/

    Parameters
    ----------
    scratch_path : pathlib.Path
    freq : str

    Returns
    -------
    dict
        {"x": path_to_x_rdr, "y": path_to_y_rdr}

    Raises
    ------
    FileNotFoundError
        If the rdr2geo files cannot be located.
    """

    scratch_path = pathlib.Path(scratch_path)

    candidates = list(
        scratch_path.glob(f"**/rdr2geo/freq{freq}/x.rdr")
    )

    if not candidates:
        raise FileNotFoundError(
            f"Could not find any x.rdr under {scratch_path} "
            f"for frequency {freq}."
        )

    # choose the shallowest path (closest to scratch root)
    candidates.sort(key=lambda p: len(p.parts))
    x_path = candidates[0]

    y_path = x_path.parent / "y.rdr"

    if not y_path.exists():
        raise FileNotFoundError(
            f"Found {x_path} but corresponding y.rdr does not exist."
        )

    return {"x": str(x_path), "y": str(y_path)}


def project_map_to_radar(
    cfg,
    input_data_path,
    freq,
    out_block_rows=512,
    out_block_cols=512,
    output_memmap_path=None,
):
    """
    Project map coordinate image to radar grid using block-wise processing.

    Parameters
    ----------
    cfg : dict
        Input runconfig file.
    input_data_path : str
        Input file path for map coordinate image.
    freq : str
        Frequency to be projected.
    out_block_rows : int
        Number of output decimated rows to process per block.
    out_block_cols : int
        Number of output decimated cols to process per block.
    output_memmap_path : str or None
        If given, output is stored in a memmap on disk instead of RAM.

    Returns
    -------
    rdr_data : numpy.ndarray or np.memmap
        Projected image in radar grid.
    """
    scratch_path = pathlib.Path(cfg["product_path_group"]["scratch_path"])

    az_looks = cfg["processing"]["crossmul"]["azimuth_looks"]
    rg_looks = cfg["processing"]["crossmul"]["range_looks"]
    unw_az_looks = cfg["processing"]["phase_unwrap"]["azimuth_looks"]
    unw_rg_looks = cfg["processing"]["phase_unwrap"]["range_looks"]

    if unw_az_looks != 1:
        az_looks = unw_az_looks
    if unw_rg_looks != 1:
        rg_looks = unw_rg_looks

    topo_paths = _find_rdr2geo_paths(scratch_path, freq)

    x_ds = gdal.Open(topo_paths["x"], gdal.GA_ReadOnly)
    y_ds = gdal.Open(topo_paths["y"], gdal.GA_ReadOnly)
    if x_ds is None or y_ds is None:
        raise RuntimeError("Failed to open rdr2geo x/y rasters.")

    x_band = x_ds.GetRasterBand(1)
    y_band = y_ds.GetRasterBand(1)

    full_rows = y_ds.RasterYSize
    full_cols = y_ds.RasterXSize

    out_rows = full_rows // az_looks
    out_cols = full_cols // rg_looks

    # center pixel positions of each multilook block
    slice_az_start = az_looks // 2
    slice_rg_start = rg_looks // 2

    _, output_dtype = _get_gdal_raster_shape_type(input_data_path)

    geo_ds = gdal.Open(input_data_path, gdal.GA_ReadOnly)
    if geo_ds is None:
        raise RuntimeError(f"Failed to open input map raster: {input_data_path}")

    if output_memmap_path is not None:
        output_arrays = np.memmap(
            output_memmap_path,
            dtype=output_dtype,
            mode="w+",
            shape=(out_rows, out_cols),
        )
        output_arrays[:] = 0
    else:
        output_arrays = np.zeros((out_rows, out_cols), dtype=output_dtype)

    for out_r0 in range(0, out_rows, out_block_rows):
        out_r1 = min(out_r0 + out_block_rows, out_rows)

        src_r0 = slice_az_start + out_r0 * az_looks
        src_r1 = slice_az_start + (out_r1 - 1) * az_looks + 1

        for out_c0 in range(0, out_cols, out_block_cols):
            out_c1 = min(out_c0 + out_block_cols, out_cols)

            src_c0 = slice_rg_start + out_c0 * rg_looks
            src_c1 = slice_rg_start + (out_c1 - 1) * rg_looks + 1

            win_x = src_c1 - src_c0
            win_y = src_r1 - src_r0
            if win_x <= 0 or win_y <= 0:
                continue

            x_block_full = x_band.ReadAsArray(src_c0, src_r0, win_x, win_y)
            y_block_full = y_band.ReadAsArray(src_c0, src_r0, win_x, win_y)

            if x_block_full is None or y_block_full is None:
                raise RuntimeError(
                    f"Failed to read rdr2geo block: "
                    f"xoff={src_c0}, yoff={src_r0}, xsize={win_x}, ysize={win_y}"
                )

            # center-pixel decimation
            x_block = x_block_full[::az_looks, ::rg_looks]
            y_block = y_block_full[::az_looks, ::rg_looks]
            del x_block_full, y_block_full

            # shape safety
            expected_shape = (out_r1 - out_r0, out_c1 - out_c0)
            if x_block.shape != expected_shape or y_block.shape != expected_shape:
                raise RuntimeError(
                    f"Unexpected decimated block shape. "
                    f"Expected={expected_shape}, x={x_block.shape}, y={y_block.shape}"
                )

            valid_mask = np.isfinite(x_block) & np.isfinite(y_block)
            if not np.any(valid_mask):
                continue

            bbox = [
                float(np.nanmin(x_block[valid_mask])),
                float(np.nanmin(y_block[valid_mask])),
                float(np.nanmax(x_block[valid_mask])),
                float(np.nanmax(y_block[valid_mask])),
            ]

            try:
                input_arr_block, [block_x0, block_y0, block_dx, block_dy] = (
                    _read_gdal_with_bbox(
                        geo_ds,
                        bbox,
                        bbox_epsg=4326,
                    )
                )
            except ValueError:
                # no overlap with map raster
                continue

            # scipy coordinates are (row, col)
            row_coords = (y_block - block_y0) / block_dy
            col_coords = (x_block - block_x0) / block_dx

            # for north-up raster, block_dy is negative, and above formula is still correct
            # because GDAL row relation uses the same dy sign.

            # Fill invalid coordinates to something harmless.
            # They will be reset after sampling.
            safe_row_coords = np.where(valid_mask, row_coords, 0.0)
            safe_col_coords = np.where(valid_mask, col_coords, 0.0)

            sampled = np.empty(expected_shape, dtype=output_dtype)

            map_coordinates(
                input_arr_block,
                [safe_row_coords, safe_col_coords],
                output=sampled,
                mode="nearest",
                order=0,
                cval=0,
                prefilter=False,
            )

            # reset invalid radar coords to zero
            if np.issubdtype(sampled.dtype, np.floating):
                sampled[~valid_mask] = np.nan
            else:
                sampled[~valid_mask] = 0

            output_arrays[out_r0:out_r1, out_c0:out_c1] = sampled

            del x_block, y_block, input_arr_block, sampled
            del row_coords, col_coords, safe_row_coords, safe_col_coords, valid_mask

    return output_arrays


def interpret_subswath_mask(subswath_mask, nodata=255):
    """
    Interprets a subswath mask integer by decoding its digits into boolean
    flags indicating reference validity, secondary validity, and water
    presence.

    Parameters
    ----------
    subswath_mask : numpy.array
        Each digit represents a specific flag:
        - Units digit (1s place): Secondary subswath mask
            Non-zero indicates valid; zero indicates invalid.
        - Tens digit (10s place): Reference subswath mask
            Non-zero indicates valid; zero indicates invalid.
        - Hundreds digit (100s place): Water presence flag.
            Non-zero indicates presence of water; zero indicates absence.
    nodata : int, default 255

    Returns
    -------
    reference_valid : bool
        True if the reference is valid (tens digit is non-zero),
        False otherwise.
    secondary_valid : bool
        True if the secondary is valid (units digit is non-zero),
        False otherwise.
    water : bool
        True if water is present (hundreds digit is non-zero),
        False otherwise.
    """
    arr = np.asarray(subswath_mask)

    nd = (arr == nodata)

    secondary_valid = subswath_mask % 10 != 0
    reference_valid = (subswath_mask // 10) % 10 != 0
    water = (subswath_mask // 100) % 10 != 0

    secondary_valid = np.where(nd, False, secondary_valid)
    reference_valid = np.where(nd, False, reference_valid)
    water = np.where(nd, False, water)

    return reference_valid, secondary_valid, water
