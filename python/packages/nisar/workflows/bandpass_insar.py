#!/usr/bin/env python3
import copy
import os
import pathlib
import time

import h5py
import isce3
import journal
import numpy as np
from scipy.fft import next_fast_len

from isce3.io import HDF5OptimizedReader
from isce3.splitspectrum import splitspectrum
from nisar.h5 import cp_h5_meta_data
from nisar.products.insar.product_paths import CommonPaths
from nisar.products.readers import RSLC
from nisar.workflows.bandpass_insar_runconfig import BandpassRunConfig
from nisar.workflows.ionosphere_runconfig import _get_rslc_h5_freq_pols
from nisar.workflows.yaml_argparse import YamlArgparse


def update_or_create(dst_h5, ds_path, value):
    """
    Create or update a dataset, creating parent groups if necessary.
    """
    # Ensure parent group exists
    if "/" in ds_path:
        grp_path, _ = ds_path.rsplit("/", 1)
        dst_h5.require_group(grp_path)

    if ds_path in dst_h5:
        # Update the dataset
        dst_h5[ds_path][...] = value
    else:
        # Create the dataset
        dst_h5.create_dataset(ds_path, data=value)


def update_list_of_frequencies(h5_path, freqs):
    """
    freqs: list like ['A', 'B']
    """
    ds_path = 'science/LSAR/identification/listOfFrequencies'
    data_bytes = [f.encode('utf-8') for f in freqs]

    with h5py.File(h5_path, "a") as f:
        # Remove existing dataset if it exists (shape is changing)
        if ds_path in f:
            del f[ds_path]

        # Create as HDF5 string dataset (elements will be bytes-like)
        dt = h5py.string_dtype(encoding='utf-8')
        f.create_dataset(ds_path, data=data_bytes, dtype=dt)


def read_rslc_swath_info(h5_path, freq):
    """
    Read selected RSLC swath items for the given frequency.

    Parameters
    ----------
    h5_path : str
        Path to the RSLC HDF5 file.
    frequency : str
        'A' or 'B'

    Returns
    -------
    dict
        Dictionary containing dataset values.
    """

    # Items we want to read
    items_to_read = [
        "acquiredCenterFrequency",
        "acquiredRangeBandwidth",
        "listOfPolarizations",
        "nominalAcquisitionPRF",
        "numberOfSubSwaths",
        "processedAzimuthBandwidth",
        "processedCenterFrequency",
        "processedRangeBandwidth",
        "sceneCenterAlongTrackSpacing",
        "sceneCenterGroundRangeSpacing",
        "slantRange",
        "slantRangeSpacing",
        "validSamplesSubSwath1",
        "validSamplesSubSwath2",
        "validSamplesSubSwath3",
    ]

    group_path = f"/science/LSAR/RSLC/swaths/frequency{freq}/"
    results = {}

    with h5py.File(h5_path, "r") as f:
        grp = f[group_path]
        for name in items_to_read:
            results[name] = grp[name][()] if name in grp else None

    return results


def _prepare_freq_pols_for_bandpass(cfg, ref_hdf5, sec_hdf5, info_channel):
    """
    Build a copy of list_of_frequencies (freq_pols_bp) with ionosphere logic
    applied. Returns (freq_pols_bp, iono_bool, iono_method, iono_freq_pols).
    """
    freq_pols_cfg = cfg["processing"]["input_subset"]["list_of_frequencies"]
    freq_pols_bp = copy.deepcopy(freq_pols_cfg)

    iono_cfg = cfg["processing"]["ionosphere_phase_correction"]
    iono_freq_pols = iono_cfg["list_of_frequencies"]
    iono_method = iono_cfg["spectral_diversity"]
    iono_bool = iono_cfg["enabled"]

    if not (iono_bool and iono_method in ["main_diff_ms_band", "main_side_band"]):
        return freq_pols_bp, iono_bool, iono_method, iono_freq_pols

    ref_B_pols = _get_rslc_h5_freq_pols(ref_hdf5, "B")
    sec_B_pols = _get_rslc_h5_freq_pols(sec_hdf5, "B")
    set_ref_B = set(ref_B_pols) if ref_B_pols else set()
    set_sec_B = set(sec_B_pols) if sec_B_pols else set()

    if not set_ref_B and not set_sec_B:
        info_channel.log(
            "Ionosphere method requires frequency B, but neither reference "
            "nor secondary SLC contains frequency B. Bandpass will be run "
            "for frequency A only."
        )
        return freq_pols_bp, iono_bool, iono_method, iono_freq_pols

    # At least one has B
    if set_ref_B and set_sec_B:
        common_B = set_ref_B & set_sec_B
        if common_B:
            chosen = sorted(common_B)
            freq_pols_bp["B"] = chosen
            info_channel.log(
                "Frequency B present in reference and secondary. Using common "
                f"B polarizations: {chosen}"
            )
        else:
            chosen = sorted(set_ref_B)
            freq_pols_bp["B"] = chosen
            info_channel.log(
                "Frequency B present in both reference and secondary, but with "
                "no common polarizations. Using reference B polarizations: "
                f"{chosen}"
            )
    else:
        # when one of SLC does not have frequency B,
        # bandpass main band (80MHz) to get side band
        side = "reference" if set_ref_B else "secondary"
        b_source = set_ref_B if set_ref_B else set_sec_B
        pols_a = set(freq_pols_bp.get("A", []))
        chosen = sorted(pols_a & b_source) or sorted(b_source)
        freq_pols_bp["B"] = chosen
        info_channel.log(
            f"Frequency B present only in {side} SLC. Using B polarizations: "
            f"{chosen}. Bandpass will be used to synthesize/align frequency B "
            "for the other SLC."
        )

    return freq_pols_bp, iono_bool, iono_method, iono_freq_pols


def run(cfg: dict):
    '''
    run bandpass
    '''
    # pull parameters from cfg
    ref_hdf5 = cfg['input_file_group']['reference_rslc_file']
    sec_hdf5 = cfg['input_file_group']['secondary_rslc_file']
    blocksize = cfg['processing']['bandpass']['lines_per_block']
    window_function = cfg['processing']['bandpass']['window_function']
    window_shape = cfg['processing']['bandpass']['window_shape']
    fft_size_cfg = cfg["processing"]["bandpass"]["range_fft_size"]
    scratch_path = pathlib.Path(cfg['product_path_group']['scratch_path'])

    info_channel = journal.info("bandpass_insar.run")
    info_channel.log("starting bandpass_insar")

    (freq_pols_bp,
     iono_bool,
     iono_method,
     iono_freq_pols) = _prepare_freq_pols_for_bandpass(
        cfg, ref_hdf5, sec_hdf5, info_channel)

    # init parameters shared by frequency A and B
    ref_slc = RSLC(hdf5file=ref_hdf5)
    sec_slc = RSLC(hdf5file=sec_hdf5)

    t_all = time.time()

    # check if bandpass is necessary
    bandpass_modes = splitspectrum.check_range_bandwidth_overlap(
        ref_slc=ref_slc,
        sec_slc=sec_slc,
        pols=freq_pols_bp)

    if not bandpass_modes:
        info_channel.log("No bandpass required. Exiting bandpass_insar.")
        return

    # Sanity-check that all freqs target the same SLC
    targets = set(bandpass_modes.values())
    if len(targets) != 1:
        raise RuntimeError(
            f"Inconsistent bandpass targets per frequency: {bandpass_modes}"
        )
    bandpass_target = targets.pop()

    # check if user provided path to raster(s) is a file or directory
    bandpass_slc_path = pathlib.Path(f"{scratch_path}/bandpass/")
    bandpass_slc_path.mkdir(parents=True, exist_ok=True)

    ref_slc_output = f"{bandpass_slc_path}/ref_slc_bandpassed.h5"
    sec_slc_output = f"{bandpass_slc_path}/sec_slc_bandpassed.h5"

    if bandpass_target == 'ref':
        target_hdf5 = ref_hdf5
        target_slc = ref_slc
        base_slc = sec_slc

        # update reference SLC path
        cfg['input_file_group']['reference_rslc_file'] = ref_slc_output
        target_output = ref_slc_output

    elif bandpass_target == 'sec':
        target_hdf5 = sec_hdf5
        target_slc = sec_slc
        base_slc = ref_slc

        # update secondary SLC path
        cfg['input_file_group']['secondary_rslc_file'] = sec_slc_output
        target_output = sec_slc_output

    if os.path.exists(target_output):
        os.remove(target_output)

    swath_path = ref_slc.SwathPath  # e.g. "/science/LSAR/RSLC/swaths"

    with HDF5OptimizedReader(name=target_hdf5, mode='r',
                             libver='latest', swmr=True) as src_h5, \
         HDF5OptimizedReader(name=target_output, mode='w') as dst_h5:

        # Copy HDF 5 file to be bandpassed
        cp_h5_meta_data(src_h5, dst_h5, f'{CommonPaths.RootPath}')

        # set of the freq and polarization
        target_freq_pol = target_slc.polarizations

        for target_freq in target_freq_pol.keys():
            for target_pol in target_freq_pol[target_freq]:
                if target_freq in freq_pols_bp.keys():
                    if target_pol not in freq_pols_bp[target_freq]:
                        delete_pol_path = (
                            f"{swath_path}/frequency{target_freq}/{target_pol}"
                        )
                        if delete_pol_path in dst_h5:
                            del dst_h5[delete_pol_path]
        new_frequencylist = bandpass_modes.keys()
        update_list_of_frequencies(target_output, new_frequencylist)

    with HDF5OptimizedReader(name=target_hdf5, mode='r',
                             libver='latest', swmr=True) as src_h5, \
         HDF5OptimizedReader(name=target_output, mode='a') as dst_h5:
        # freq: [A, B], target : 'ref' or 'sec'
        for freq, target in bandpass_modes.items():
            pol_list = freq_pols_bp[freq]

            # meta data extraction
            base_meta_data = splitspectrum.BandpassMetaData.load_from_slc(
                slc_product=base_slc,
                freq=freq)
            target_meta_data = splitspectrum.BandpassMetaData.load_from_slc(
                slc_product=target_slc,
                freq='A')

            sampling_bandwidth_ratio = \
                base_meta_data.rg_sample_freq / base_meta_data.rg_bandwidth

            info_channel.log("base RSLC:")
            info_channel.log(
                f"    bandwidth : {base_meta_data.rg_bandwidth}")
            info_channel.log(
                f"    sampling_frequency : {base_meta_data.rg_sample_freq}")
            info_channel.log("target RSLC:")
            info_channel.log(
                f"    bandwidth : {target_meta_data.rg_bandwidth}")
            info_channel.log(
                f"    sampling_frequency : {target_meta_data.rg_sample_freq}")
            info_channel.log(
                f"sampling_frequency / bandwidth : {sampling_bandwidth_ratio}")

            bandwidth_half = 0.5 * base_meta_data.rg_bandwidth
            low_frequency_base = \
                base_meta_data.center_freq - bandwidth_half
            high_frequency_base = \
                base_meta_data.center_freq + bandwidth_half

            # Initialize bandpass instance
            # Specify meta parameters of SLC to be bandpassed
            bandpass = splitspectrum.SplitSpectrum(
                rg_sample_freq=target_meta_data.rg_sample_freq,
                rg_bandwidth=target_meta_data.rg_bandwidth,
                center_frequency=target_meta_data.center_freq,
                slant_range=target_meta_data.slant_range,
                freq=freq,
                sampling_bandwidth_ratio=sampling_bandwidth_ratio)
            swath_path = ref_slc.SwathPath
            dest_freq_path = f"{swath_path}/frequency{freq}"
            dest_freq_a_path = f"{swath_path}/frequencyA"

            # Copy HDF 5 file to be bandpassed
            # cp_h5_meta_data(src_h5, dst_h5, f'{CommonPaths.RootPath}')

            for pol in pol_list:

                target_raster_str = \
                    f'HDF5:{target_hdf5}:/{target_slc.slcPath("A", pol)}'
                target_slc_raster = isce3.io.Raster(target_raster_str)
                rows = target_slc_raster.length
                cols = target_slc_raster.width
                nblocks = int(np.ceil(rows / blocksize))

                if fft_size_cfg is None:
                    fft_size = next_fast_len(cols)
                else:
                    fft_size = fft_size_cfg

                reader = target_slc.getSlcDatasetAsNativeComplex('A', pol)
                dest_pol_path = f"{dest_freq_path}/{pol}"

                for block in range(0, nblocks):
                    print("-- bandpass block: ", block)
                    row_start = block * blocksize
                    if (row_start + blocksize > rows):
                        block_rows_data = rows - row_start
                    else:
                        block_rows_data = blocksize

                    # Read SLC from HDF5
                    target_slc_image = reader[
                        row_start:row_start + block_rows_data,
                        :]
                    # Specify low and high frequency to be passed (bandpass)
                    # and the center frequency to be basebanded (demodulation)
                    bandpass_slc, bandpass_meta = \
                        bandpass.bandpass_shift_spectrum(
                            slc_raster=target_slc_image,
                            low_frequency=low_frequency_base,
                            high_frequency=high_frequency_base,
                            new_center_frequency=base_meta_data.center_freq,
                            fft_size=fft_size,
                            window_shape=window_shape,
                            window_function=window_function,
                            resampling=True
                            )

                    if block == 0:
                        if dest_pol_path in dst_h5:
                            del dst_h5[dest_pol_path]
                        # Initialize the raster with updated shape in HDF5
                        dst_h5.create_dataset(
                            dest_pol_path,
                            [rows, np.shape(bandpass_slc)[1]],
                            np.complex64, chunks=(128, 128))
                    # Write bandpassed SLC to HDF5
                    dst_h5[dest_pol_path].write_direct(
                        bandpass_slc,
                        dest_sel=np.s_[row_start:row_start + block_rows_data,
                                       :])

                dst_h5[dest_pol_path].attrs['description'] = \
                    f"Bandpass SLC image ({pol})"
                dst_h5[dest_pol_path].attrs['units'] = ""

            # update meta information for bandpass SLC

            update_or_create(
                dst_h5,
                f"{dest_freq_path}/processedCenterFrequency",
                bandpass_meta['center_frequency'])
            update_or_create(
                dst_h5,
                f"{dest_freq_path}/slantRangeSpacing",
                bandpass_meta['range_spacing'])
            update_or_create(
                dst_h5,
                f"{dest_freq_path}/processedRangeBandwidth",
                base_meta_data.rg_bandwidth)
            update_or_create(
                dst_h5,
                f"{dest_freq_path}/processedAzimuthBandwidth",
                base_meta_data.rg_bandwidth)

            bandpass_ratio = (
                target_meta_data.rg_pxl_spacing / bandpass_meta['range_spacing']
            )
            subswath_num_path = f"{dest_freq_path}/numberOfSubSwaths"
            subswath_num_a_path = f"{dest_freq_a_path}/numberOfSubSwaths"

            if subswath_num_path in src_h5:
                subswath_number = src_h5[subswath_num_path][()]
                for swath_count in range(subswath_number):
                    # Update the validateSamplesSubswaths
                    valid_sample_path = \
                        f"{dest_freq_path}/validSamplesSubSwath{swath_count+1}"
                    valid_samples = src_h5[valid_sample_path][()]
                    valid_samples_bandpass = \
                        np.array(valid_samples * bandpass_ratio, dtype='int')

                    update_or_create(
                        dst_h5,
                        valid_sample_path,
                        valid_samples_bandpass.astype(int))

            elif subswath_num_a_path in src_h5 and subswath_num_path not in src_h5:
                bandpass_ratio_a = (
                    target_meta_data.rg_pxl_spacing / bandpass_meta['range_spacing']
                )
                subswath_number = src_h5[subswath_num_a_path][()]
                update_or_create(
                    dst_h5,
                    subswath_num_path,
                    subswath_number)
                for swath_count in range(subswath_number):
                    # Update the validateSamplesSubswaths
                    valid_sample_path = \
                        f"{dest_freq_a_path}/validSamplesSubSwath{swath_count+1}"
                    valid_sample_target_path = \
                        f"{dest_freq_path}/validSamplesSubSwath{swath_count+1}"
                    valid_samples = src_h5[valid_sample_path][()]
                    valid_samples_bandpass = \
                        np.array(valid_samples * bandpass_ratio_a, dtype='int')
                    update_or_create(
                        dst_h5,
                        valid_sample_target_path,
                        valid_samples_bandpass.astype(int))

            # update slant range for bandpassed SLC
            slant_path = f"{dest_freq_path}/slantRange"
            if slant_path in dst_h5:
                del dst_h5[slant_path]
            dst_h5.create_dataset(slant_path, data=bandpass_meta['slant_range'])

            # Create missing information under the "swath"
            if freq != 'A':
                swath_meta_A = read_rslc_swath_info(target_hdf5, 'A')
                for name, value in swath_meta_A.items():
                    if value is None:
                        continue
                    dest_ds_path = f"{dest_freq_path}/{name}"
                    if dest_ds_path in dst_h5:
                        continue
                    update_or_create(dst_h5, dest_ds_path, value)
                param_freq_a_path = f"{ref_slc.ProcessingInformationPath}/parameters/frequencyA"
                param_freq_b_path = f"{ref_slc.ProcessingInformationPath}/parameters/frequencyB"
                if param_freq_b_path in dst_h5:
                    del dst_h5[param_freq_b_path]

                # Copy entire group
                src_h5.copy(param_freq_a_path, dst_h5, name=param_freq_b_path)

    t_all_elapsed = time.time() - t_all
    print('total processing time: ', t_all_elapsed, ' sec')
    info_channel.log(
        f"successfully ran bandpass_insar in {t_all_elapsed:.3f} seconds")


if __name__ == "__main__":
    '''
    run bandpass from command line
    '''
    # load command line args
    bandpass_parser = YamlArgparse()
    args = bandpass_parser.parse()
    # get a runconfig dict from command line args
    bandpass_runconfig = BandpassRunConfig(args)
    # run bandpass
    run(bandpass_runconfig.cfg)
