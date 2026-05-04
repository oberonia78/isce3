
import itertools
import journal
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.ndimage import label as nd_label
from scipy.ndimage import (binary_erosion,
                           find_objects,
                           binary_dilation)
from scipy.sparse import csgraph as csg
from scipy.spatial import cKDTree
from typing import Tuple, Dict, Any, List


def bridge_unwrapped_phase(unw_phase: np.ndarray,
                           radius: int,
                           min_num_pixel: int,
                           erosion_size: int,
                           ramp_type: str,
                           deramp_max_num_sample: int,
                           phase_jump_constant=2.0 * np.pi,
                           bridge_method: str = "mst",
                           max_bridge_distance: Optional[float] = 2000, #None,
                           max_num_neighbor: int = 10,
                           residual_ratio_threshold: float = 0.35,
                           min_vote_confidence: float = 0.5,
                           min_num_votes: int = 1) -> np.ndarray:
    """Bridge disconnected unwrapped phase regions by estimating and removing
    their relative phase offsets, which is interger of 2 pi.

    In unwrapping, isolated components may be unwrapped independently,
    producing integer cycle jumps between them. This function finds
    nearby components (within `radius` pixels), computes the median phase
    difference at their nearest boundary pixels, shifts one region to align it,
    and merges the labels—effectively “bridging” the jump.
    Finally, it filters out speckle (small islands) and smooths
    mask edges via erosion.

    Parameters
    ----------
    unw_phase : numpy.ndarray
        2D array representing the unwrapped phase image.
    radius : int
        The maximum radius used when bridging disconnected regions.
    min_num_pixel : int
        The minimum number of pixel of connected components to retain during processing.
    erosion_size : int
        The size of the structuring element used for erosion during labeling.
    ramp_type : str, optional
        Type of ramp to be estimated. Default is 'linear'.
        Possible options -
        'linear', 'quadratic', 'linear_range', 'linear_azimuth'
        'quadratic_range', 'quadratic_azimuth'
    deramp_max_num_sample : float, optional
        Maximum number of pixel samples, above which uniform sampling is
        applied to reduce sample size. Default is 1e6.    Returns
    -------
    bridge_unw : numpy.ndarray
        2D array with the bridged unwrapped phase image.
    """
    channel = journal.info("isce3.unwrap.bridge_phase.bridge_unwrapped_phase")
    channel.log("Starting bridge_unwrapped_phase")
    channel.log(f"phase_jump_constant: {phase_jump_constant}")

    runw_img_bool = unw_phase != 0
    label_img, num_cluster = nd_label(runw_img_bool, structure=np.ones((3, 3)))
    channel.log(f"Bridge algorithm : {num_cluster} clusters")

    if num_cluster <= 1:
        channel.log("Bridge algorithm is not applied since all components are connected.")
        return unw_phase

    channel.log(
        f"Bridge parameters: radius={radius}, "
        f"min_num_pixel={min_num_pixel}, "
        f"erosion_size={erosion_size}, "
        f"ramp_type={ramp_type}, "
        f"deramp_max_num_sample={deramp_max_num_sample}"
    )

    channel.log(f"   radius: {radius}   min_num_pixel: {min_num_pixel}  erosion_size: {erosion_size} ")
    cc = bridgeConnectComponent(conncomp=label_img)
    cc.label(min_num_pixel=min_num_pixel, erosion_size=erosion_size)

    channel.log(f"Number of labels after cleanup: {cc.num_label}")
    channel.log(f"Reference label: {cc.label_ref}")

    if cc.label_ref is not None:
        if bridge_method == "mst":
            cc.find_mst_bridge()
            bridge_unw = cc.unwrap_conn_comp(
                unw_phase,
                radius=radius,
                ramp_type=ramp_type,
                max_num_sample=deramp_max_num_sample,
                phase_jump_constant=phase_jump_constant
            )

        elif bridge_method == "voting":
            bridge_unw = cc.unwrap_conn_comp_with_voting(
                unw_phase,
                radius=radius,
                ramp_type=ramp_type,
                max_num_sample=deramp_max_num_sample,
                phase_jump_constant=phase_jump_constant,
                max_bridge_distance=max_bridge_distance,
                max_num_neighbor=max_num_neighbor,
                residual_ratio_threshold=residual_ratio_threshold,
                min_vote_confidence=min_vote_confidence,
                min_num_votes=min_num_votes,
            )

        else:
            raise ValueError(f"Unsupported bridge_method: {bridge_method}")

        channel.log("Finished unwrap_conn_comp")
        channel.log(f"Output non-zero pixels: {np.count_nonzero(bridge_unw)}")

    else:

        channel.log("Bridge algorithm is skipped because no valid reference label was found.")
        bridge_unw = unw_phase
    channel.log("Finished bridge_unwrapped_phase")

    return bridge_unw


def label_boundary(
        label_img: np.ndarray,
        num_label: int,
        erosion_size: int = 5,
        ) -> Tuple[np.ndarray, int, np.ndarray]:
    """
    Label the boundary of the labeled array.

    This function labels the boundaries of connected components in a labeled
    array. It optionally applies morphological erosion to the labeled array
    before finding and labeling the boundaries.

    Parameters
    ----------
    label_img : numpy.ndarray
        2D array of labeled regions.
    num_label : int
        The number of labels in the labeled array.
    erosion_size : int, optional
        The size of the structuring element used for morphological erosion.
        Default is 5.

    Returns
    -------
    label_img : numpy.ndarray
        2D array with relabeled regions after erosion and boundary labeling.
    num_label : int
        The updated number of labels after relabeling.
    label_bound : numpy.ndarray
        2D array of labeled boundaries of the regions.
    """
    channel = journal.info("isce3.unwrap.bridge_phase.label_boundary")

    if erosion_size > 0:
        erosion_yy, erosion_xx = np.ogrid[-erosion_size:erosion_size + 1,
                                          -erosion_size:erosion_size + 1]
        # circle mask
        erosion_structure = (erosion_xx ** 2 +
                             erosion_yy ** 2) <= (erosion_size**2)
        label_erosion_img = binary_erosion(
            label_img,
            structure=erosion_structure).astype(np.uint8)

        labeled_array, _ = nd_label(label_erosion_img)
        regions = find_objects(labeled_array)

        if len(regions) < num_label:
            channel.log(
                "Regions lost during morphological erosion operation:")
            erosion_labels = [label_erosion_img[region].max()
                              for region in regions]
            for i in range(1, num_label + 1):
                if i not in erosion_labels:
                    label_img[label_img == i] = 0

    else:
        label_erosion_img = label_img > 0
    label_img, num_label = nd_label(label_img, structure=np.ones((3, 3)))
    # Create a boundary map using binary dilation and subtracting the original image
    boundary_img = binary_dilation(label_erosion_img) & ~label_erosion_img
    label_bound = boundary_img.astype(np.uint8)
    label_bound *= label_erosion_img

    return label_img, num_label, label_bound


def label_conn_comp(
        mask: np.ndarray,
        min_num_pixel: float = 2.5e3,
        erosion_size: int = 5,
        ) -> Tuple[np.ndarray, int]:
    """
    Label and clean up the connected components mask.

    This function labels the connected components in a binary mask,
    removes small objects below a specified area, and optionally applies
    morphological erosion to refine the labels.

    Parameters
    ----------
    mask : numpy.ndarray
        2D binary array representing the mask of connected components.
    min_num_pixel : float, optional
        Minimum number of pixels for a region to be kept. Default is 2.5e3.
    erosion_size : int, optional
        Size of the structuring element used for morphological erosion.
        Default is 5.

    Returns
    -------
    label_img : numpy.ndarray
        2D array with labeled connected components after cleaning and erosion.
    num_label : int
        The number of labels (connected components) after cleaning and erosion.
    """
    channel = journal.info("isce3.unwrap.bridge_phase.label_conn_comp")

    # Label the connected components
    label_img, num_label = nd_label(mask, structure=np.ones((3, 3)))

    # Calculate min_num_pixel if not specified
    channel.log(f"Removing regions with area < {int(min_num_pixel)}")

    # Remove small objects
    object_slices = find_objects(label_img)
    for i, slice_ in enumerate(object_slices, start=1):
        if slice_ is not None:
            if np.sum(label_img[slice_] == i) < min_num_pixel:
                label_img[label_img == i] = 0

    # Re-label after removing small objects
    label_img, num_label = nd_label(label_img, structure=np.ones((3, 3)))

    # Apply morphological erosion if specified
    if erosion_size > 0:
        erosion_structure = np.ones((erosion_size, erosion_size), dtype=bool)
        label_erosion_img = binary_erosion(
            label_img > 0,
            structure=erosion_structure).astype(np.uint8)

        labeled_array, _ = nd_label(label_erosion_img)
        regions = find_objects(labeled_array)

        if len(regions) < num_label:
            channel.log("Regions lost during morphological erosion operation:")
            erosion_labels = [label_erosion_img[region].max()
                              for region in regions]
            for i in range(1, num_label + 1):
                if i not in erosion_labels:
                    label_img[label_img == i] = 0

        # Re-label after erosion
        label_img, num_label = nd_label(label_img, structure=np.ones((3, 3)))

    return label_img, num_label


class bridgeConnectComponent:
    """Object for bridging connected components."""

    def __init__(self, conncomp: np.ndarray):
        """Initialize the ConnectComponent object."""
        if not isinstance(conncomp, np.ndarray):
            raise ValueError("Input conncomp is not np.ndarray")
        self.conncomp = conncomp
        self.length, self.width = self.conncomp.shape

    def label(self,
              min_num_pixel: float = 2.5e3,
              erosion_size: int = 5) -> None:
        """
        Label connected components in the image and identify the reference
        label.

        This function labels connected components in the input image based on a
        minimum area threshold and performs boundary erosion. It also
        identifies the reference label as the largest connected component.

        Parameters
        ----------
        min_num_pixel : float, optional
            Minimum area threshold for connected components. Default is 2500.
        erosion_size : int, optional
            Size of the structuring element for boundary erosion. Default is 5.

        Returns
        -------
        None
        """
        channel = journal.info(
            "isce3.unwrap.bridge_phase.bridgeConnectComponent")
        self.labelImg, self.num_label = label_conn_comp(
            self.conncomp,
            min_num_pixel=min_num_pixel,
            erosion_size=erosion_size)

        if self.num_label == 1:
            channel.log(f"Bridge algorithm is not applied because only one component exists.")
        elif self.num_label == 0:
            channel.log(f"Bridge algorithm is not applied because component does not exist.")

        self.labelImg, self.num_label, self.labelBound = label_boundary(
            self.labelImg,
            self.num_label,
            erosion_size=erosion_size
            )

        regions = find_objects(self.labelImg)
        # if regions are not empty
        if regions:
            idx = np.argmax([np.sum(self.labelImg[region] == (i + 1))
                             for i, region in enumerate(regions)])
            self.label_ref = idx + 1
        # if regions are empty
        else:
            self.label_ref = None

    def get_all_bridge(self) -> Tuple[Dict[str, Any], np.ndarray]:
        """
        Compute all possible bridges between labeled areas and their distances.

        This function calculates the shortest distances between all pairs of
        labeled areas in the label image using k-d trees for efficient
        nearest-neighbor searches. It stores the connections and distances in
        a dictionary and a distance matrix.

        Parameters
        ----------
        None

        Returns
        -------
        connDict : Dict[str, Any]
            A dictionary containing the coordinates of the closest points
            between each pair of labeled areas and their distances. Keys
            are formatted as "{label1}_{label2}" with values as dictionaries
            containing:
            - label1: coordinates of the closest point in the first labeled
                      area.
            - label2: coordinates of the closest point in the second labeled
                      area.
            - "distance": the distance between these points.
        distMat : np.ndarray
            A symmetric matrix of shape (num_label, num_label) containing the
            minimum distances between each pair of labeled areas.
        """
        self.connDict = {}
        self.distMat = np.zeros((self.num_label, self.num_label),
                                dtype=np.float32)
        # if the number of labels is zero, then return empty dictionary and
        # zero array
        if self.num_label == 0:
            return self.connDict, self.distMat

        trees = [cKDTree(np.argwhere(self.labelImg == i + 1))
                 for i in range(self.num_label)]

        for i, j in itertools.combinations(range(self.num_label), 2):
            dist, idx = trees[i].query(trees[j].data)
            idx_min = np.argmin(dist)
            yxi = trees[i].data[idx[idx_min]]
            yxj = trees[j].data[idx_min]
            dist_min = dist[idx_min]
            n0, n1 = str(i + 1), str(j + 1)
            conn = {n0: yxi, n1: yxj, "distance": dist_min}
            self.connDict[f"{n0}_{n1}"] = conn
            self.distMat[i, j] = self.distMat[j, i] = dist_min

        return self.connDict, self.distMat

    def find_candidate_bridges(
        self,
        max_bridge_distance: Optional[float] = None,
        max_num_neighbor: int = 5,
    ) -> List[Dict[str, Union[int, float]]]:
        """
        Find multiple candidate bridges between connected components.

        Unlike MST, this keeps several nearby bridges per label, which allows
        voting-based offset estimation and reduces sensitivity to one bad bridge.

        Parameters
        ----------
        max_bridge_distance : float, optional
            Maximum allowed bridge distance. If None, all candidate bridges are kept
            before applying max_num_neighbor.
        max_num_neighbor : int
            Maximum number of neighbor bridges to keep per label.

        Returns
        -------
        candidate_bridges : list of dict
            Candidate bridge list.
        """
        channel = journal.info(
            "isce3.unwrap.bridge_phase.bridgeConnectComponent"
        )

        channel.log("Finding candidate bridges for voting-based correction")

        if not hasattr(self, "connDict") or not hasattr(self, "distMat"):
            self.get_all_bridge()

        all_edges = []

        for key, conn in self.connDict.items():
            label0_str, label1_str = key.split("_")
            label0 = int(label0_str)
            label1 = int(label1_str)

            distance = float(conn["distance"])

            if max_bridge_distance is not None and distance > max_bridge_distance:
                continue

            y0, x0 = conn[label0_str]
            y1, x1 = conn[label1_str]

            all_edges.append({
                "first_endpoint_x": int(x0),
                "first_endpoint_y": int(y0),
                "second_endpoint_x": int(x1),
                "second_endpoint_y": int(y1),
                "distance": distance,
                "label0": label0,
                "label1": label1,
            })

        channel.log(f"Number of candidate bridges before neighbor filtering: {len(all_edges)}")

        if max_num_neighbor is not None and max_num_neighbor > 0:
            selected_keys = set()

            for label_id in range(1, self.num_label + 1):
                connected_edges = [
                    edge for edge in all_edges
                    if edge["label0"] == label_id or edge["label1"] == label_id
                ]

                connected_edges = sorted(
                    connected_edges,
                    key=lambda edge: edge["distance"]
                )

                for edge in connected_edges[:max_num_neighbor]:
                    key = tuple(sorted((edge["label0"], edge["label1"])))
                    selected_keys.add(key)

            candidate_bridges = [
                edge for edge in all_edges
                if tuple(sorted((edge["label0"], edge["label1"]))) in selected_keys
            ]
        else:
            candidate_bridges = all_edges

        candidate_bridges = sorted(
            candidate_bridges,
            key=lambda edge: edge["distance"]
        )

        self.candidate_bridges = candidate_bridges

        channel.log(f"Number of candidate bridges after filtering: {len(candidate_bridges)}")

        return candidate_bridges

    def find_mst_bridge(self) -> List[Dict[str, Union[int, float]]]:
        """
        Search for bridges to connect all labeled areas using the minimum
        spanning tree algorithm.

        This function finds the minimum set of connections (bridges) needed to
        connect all labeled areas using the minimum spanning tree (MST)
        algorithm. If the distance matrix (`self.distMat`) is not
        available, it is computed using the `get_all_bridge` method.

        Parameters
        ----------
        None

        Returns
        -------
        bridges : List[Dict[str, Union[int, float]]]
            A list of dictionaries, each representing a bridge with the
            following keys:
            - 'first_endpoint_x': x-coordinate of the first node.
            - 'first_endpoint_y': y-coordinate of the first node.
            - 'second_endpoint_x': x-coordinate of the second node.
            - 'second_endpoint_y': y-coordinate of the second node.
            - 'label0': label of the first node.
            - 'label1': label of the second node.
            - 'distance': Euclidean distance between the two nodes.
        """
        if not hasattr(self, "distMat"):
            self.get_all_bridge()

        distMatMst = csg.minimum_spanning_tree(self.distMat)
        succs, preds = csg.breadth_first_order(
            distMatMst, i_start=self.label_ref - 1, directed=False
        )
        self.bridges = []
        for i in range(1, succs.size):
            n0 = preds[succs[i]] + 1
            n1 = succs[i] + 1
            if n0 > n1:
                nn = [str(n1), str(n0)]
            else:
                nn = [str(n0), str(n1)]
            conn = self.connDict[f"{nn[0]}_{nn[1]}"]
            first_endpoint_y, first_endpoint_x = conn[str(n0)]
            second_endpoint_y, second_endpoint_x = conn[str(n1)]
            bridge = dict()
            bridge = {
                "first_endpoint_x": int(first_endpoint_x),
                "first_endpoint_y": int(first_endpoint_y),
                "second_endpoint_x": int(second_endpoint_x),
                "second_endpoint_y": int(second_endpoint_y),
                "distance": float(
                    ((second_endpoint_x - first_endpoint_x)**2 +
                     (second_endpoint_y - first_endpoint_y)**2)**0.5),
                "label0": n0,
                "label1": n1,
            }
            self.bridges.append(bridge)
        self.num_bridge = len(self.bridges)
        return self.bridges

    def get_bridge_endpoint_aoi_mask(
        self, bridge: Dict[str, int], radius: int = 50
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the Area of Interest (AOI) mask for bridge endpoints.

        This function generates AOI masks around the endpoints of a bridge
        between two connected components in the phase data. The AOI masks are
        square regions centered at the bridge endpoints with a specified
        radius.

        Parameters
        ----------
        bridge : Dict[str, int]
            Dictionary containing the coordinates of the bridge endpoints.
            Keys should include:
            - "first_endpoint_x": x-coordinate of the first endpoint.
            - "first_endpoint_y": y-coordinate of the first endpoint.
            - "second_endpoint_x": x-coordinate of the second endpoint.
            - "second_endpoint_y": y-coordinate of the second endpoint.
        radius : int, optional
            Radius of the square AOI around each bridge endpoint.
            Default is 50.

        Returns
        -------
        aoi_mask0 : np.ndarray
            Boolean mask of the AOI around the first bridge endpoint.
        aoi_mask1 : np.ndarray
            Boolean mask of the AOI around the second bridge endpoint.
        """
        # 1) unpack the two endpoints
        first_endpoint_x = bridge["first_endpoint_x"]
        first_endpoint_y = bridge["first_endpoint_y"]
        second_endpoint_x = bridge["second_endpoint_x"]
        second_endpoint_y = bridge["second_endpoint_y"]

        # 2) compute row/col bounds for the first endpoint
        first_col_min = max(0, first_endpoint_x - radius)
        first_col_max = min(self.width, first_endpoint_x  + radius + 1)
        first_row_min = max(0, first_endpoint_y - radius)
        first_row_max = min(self.length, first_endpoint_y  + radius + 1)

        # 3) compute row/col bounds for the second endpoint
        second_col_min = max(0, second_endpoint_x - radius)
        second_col_max = min(self.width, second_endpoint_x + radius + 1)
        second_row_min = max(0, second_endpoint_y - radius)
        second_row_max = min(self.length, second_endpoint_y + radius + 1)

        # 4) build the two AOI masks
        aoi_mask_first = np.zeros(self.labelImg.shape, dtype=bool)
        aoi_mask_second = np.zeros(self.labelImg.shape, dtype=bool)

        aoi_mask_first[
            first_row_min:first_row_max,
            first_col_min:first_col_max
        ] = True

        aoi_mask_second[
            second_row_min:second_row_max,
            second_col_min:second_col_max
        ] = True

        return aoi_mask_first, aoi_mask_second

    def unwrap_conn_comp(
        self,
        unw: np.ndarray,
        radius: int = 50,
        ramp_type: Optional[str] = None,
        max_num_sample: int = 1e6,
        phase_jump_constant=2 * np.pi
    ) -> np.ndarray:
        """
        Perform bridging to unwrap connected components in the phase data.

        This function unrolls the phase data by bridging the gaps between
        connected components, optionally removing any ramp present in the data.

        Parameters
        ----------
        unw : np.ndarray
            2D array of unwrapped phase data.
        radius : int, optional
            Radius of the area of interest (AOI) around bridge endpoints for
            unwrapping. Default is 50.
        ramp_type : Optional[str], optional
            Type of ramp to be estimated and removed before unwrapping.
            If None, no ramp is removed. Default is None.

        Returns
        -------
        unw : np.ndarray
            2D array of unwrapped phase data after bridging.
        """
        channel = journal.info(
            "isce3.unwrap.bridge_phase.bridgeConnectComponent")
        radius = int(min(radius, min(self.conncomp.shape) * 0.5))
        unw = np.array(unw, dtype=np.float32)

        if ramp_type is not None:
            channel.log(f"Estimating a {ramp_type} ramp")
            ramp_mask = self.labelImg == self.label_ref
            unw, ramp = deramp(unw, ramp_mask, ramp_type, max_num_sample)

        for bridge in self.bridges:
            aoi_mask0, aoi_mask1 = self.get_bridge_endpoint_aoi_mask(
                bridge, radius=radius
            )
            label_mask0 = self.labelImg == bridge["label0"]
            label_mask1 = self.labelImg == bridge["label1"]

            sample_mask0 = np.logical_and(aoi_mask0, label_mask0)
            sample_mask1 = np.logical_and(aoi_mask1, label_mask1)

            value0 = np.nanmedian(unw[sample_mask0])
            value1 = np.nanmedian(unw[sample_mask1])

            diff_value = value1 - value0
            num_jump = int(np.round(np.abs(diff_value) / phase_jump_constant))

            if diff_value > 0:
                num_jump *= -1
            correction = phase_jump_constant * num_jump

            unw[label_mask1] += correction

        if ramp_type is not None:
            unw += ramp
        unw[self.labelImg == 0] = 0

        return unw



    def unwrap_conn_comp_with_voting(
        self,
        unw: np.ndarray,
        radius: int = 50,
        ramp_type: Optional[str] = None,
        max_num_sample: int = 1e6,
        phase_jump_constant: float = 2.0 * np.pi,
        max_bridge_distance: Optional[float] = None,
        max_num_neighbor: int = 5,
        residual_ratio_threshold: float = 0.35,
        min_vote_confidence: float = 0.5,
        min_num_votes: int = 1,
    ) -> np.ndarray:
        """
        Correct connected components using multiple candidate bridges and voting.

        This method estimates integer jump relations between many nearby label
        pairs, then determines one global integer offset per label by voting.
        Corrections are applied only once at the end, avoiding sequential
        propagation of a bad bridge correction.

        Parameters
        ----------
        unw : np.ndarray
            2D unwrapped phase array.
        radius : int
            AOI radius around bridge endpoints.
        ramp_type : str, optional
            Ramp type to remove before estimating jumps.
        max_num_sample : int
            Maximum sample size for deramp.
        phase_jump_constant : float
            Phase jump constant.
        max_bridge_distance : float, optional
            Maximum allowed bridge distance for candidate edges.
        max_num_neighbor : int
            Maximum number of nearby candidate bridges per label.
        residual_ratio_threshold : float
            Reject bridge if residual / phase_jump_constant is larger than this.
        min_vote_confidence : float
            Minimum weighted vote confidence needed to accept an offset.
        min_num_votes : int
            Minimum number of valid votes needed to accept an offset.

        Returns
        -------
        unw_out : np.ndarray
            Corrected unwrapped phase.
        """
        channel = journal.info(
            "isce3.unwrap.bridge_phase.bridgeConnectComponent"
        )

        channel.log("Starting unwrap_conn_comp_with_voting")

        original_radius = radius
        radius = int(min(radius, min(self.conncomp.shape) * 0.5))

        channel.log(f"Input radius: {original_radius}")
        channel.log(f"Effective radius: {radius}")
        channel.log(f"phase_jump_constant: {phase_jump_constant}")
        channel.log(f"max_num_neighbor: {max_num_neighbor}")
        channel.log(f"max_bridge_distance: {max_bridge_distance}")
        channel.log(f"residual_ratio_threshold: {residual_ratio_threshold}")
        channel.log(f"min_vote_confidence: {min_vote_confidence}")
        channel.log(f"min_num_votes: {min_num_votes}")

        unw_work = np.array(unw, dtype=np.float32)

        if ramp_type is not None:
            channel.log(f"Estimating and removing {ramp_type} ramp")
            ramp_mask = self.labelImg == self.label_ref

            unw_work, ramp = deramp(
                unw_work,
                ramp_mask,
                ramp_type,
                max_num_sample,
            )

            channel.log(
                f"Ramp removed. Ramp median={np.nanmedian(ramp):.4f}, "
                f"ramp min={np.nanmin(ramp):.4f}, "
                f"ramp max={np.nanmax(ramp):.4f}"
            )
        else:
            ramp = None

        candidate_bridges = self.find_candidate_bridges(
            max_bridge_distance=max_bridge_distance,
            max_num_neighbor=max_num_neighbor,
        )

        if len(candidate_bridges) == 0:
            channel.log("No candidate bridges found. Returning input.")
            return unw

        edge_relations = []

        for i, bridge in enumerate(candidate_bridges, start=1):
            label0 = int(bridge["label0"])
            label1 = int(bridge["label1"])

            aoi_mask0, aoi_mask1 = self.get_bridge_endpoint_aoi_mask(
                bridge,
                radius=radius,
            )

            label_mask0 = self.labelImg == label0
            label_mask1 = self.labelImg == label1

            sample_mask0 = np.logical_and(aoi_mask0, label_mask0)
            sample_mask1 = np.logical_and(aoi_mask1, label_mask1)

            num_sample0 = np.count_nonzero(sample_mask0)
            num_sample1 = np.count_nonzero(sample_mask1)

            if num_sample0 == 0 or num_sample1 == 0:
                # channel.log(
                #     f"Candidate bridge {i} skipped: "
                #     f"label {label0} - label {label1}, "
                #     f"samples=({num_sample0}, {num_sample1})"
                # )
                continue

            value0 = np.nanmedian(unw_work[sample_mask0])
            value1 = np.nanmedian(unw_work[sample_mask1])

            if not np.isfinite(value0) or not np.isfinite(value1):
                channel.log(
                    f"Candidate bridge {i} skipped because median is not finite."
                )
                continue

            num_jump_01, correction, residual = estimate_integer_jump(
                value0,
                value1,
                phase_jump_constant,
            )

            residual_ratio = residual / phase_jump_constant

            if residual_ratio > residual_ratio_threshold:
                # channel.log(
                #     f"Candidate bridge {i} rejected: "
                #     f"label {label0} - label {label1}, "
                #     f"median0={value0:.4f}, median1={value1:.4f}, "
                #     f"jump={num_jump_01}, residual_ratio={residual_ratio:.3f}"
                # )
                continue

            sample_weight = min(num_sample0, num_sample1)
            distance_weight = 1.0 / (float(bridge["distance"]) + 1.0)
            residual_weight = 1.0 / (residual_ratio + 1.0e-3)

            weight = sample_weight * distance_weight * residual_weight

            edge_relations.append({
                "label0": label0,
                "label1": label1,
                "num_jump_01": num_jump_01,
                "residual_ratio": residual_ratio,
                "weight": weight,
                "distance": float(bridge["distance"]),
                "num_sample0": int(num_sample0),
                "num_sample1": int(num_sample1),
                "median0": float(value0),
                "median1": float(value1),
            })

            # channel.log(
            #     f"Candidate bridge {i} accepted: "
            #     f"label {label0} - label {label1}, "
            #     f"median0={value0:.4f}, median1={value1:.4f}, "
            #     f"jump01={num_jump_01}, "
            #     f"residual_ratio={residual_ratio:.3f}, "
            #     f"weight={weight:.3f}"
            # )

        if len(edge_relations) == 0:
            channel.log("No reliable bridge relations found. Returning input.")
            return unw

        # Voting-based global offset estimation
        #
        # offset[label] means:
        # corrected_phase = original_phase + offset[label] * phase_jump_constant
        #
        # For an edge label0 -> label1:
        # offset[label1] - offset[label0] = num_jump_01
        label_offsets = {int(self.label_ref): 0}
        unresolved_labels = set(range(1, self.num_label + 1))
        unresolved_labels.discard(int(self.label_ref))

        channel.log(f"Reference label: {self.label_ref}")
        channel.log(f"Initial unresolved labels: {sorted(unresolved_labels)}")

        max_iter = self.num_label + 5

        for iteration in range(max_iter):
            if len(unresolved_labels) == 0:
                break

            newly_resolved = {}

            for label_id in sorted(unresolved_labels):
                votes = []

                for edge in edge_relations:
                    label0 = edge["label0"]
                    label1 = edge["label1"]
                    num_jump_01 = edge["num_jump_01"]
                    weight = edge["weight"]

                    if label_id == label1 and label0 in label_offsets:
                        # offset1 = offset0 + jump01
                        vote_offset = label_offsets[label0] + num_jump_01
                        votes.append((vote_offset, weight))

                    elif label_id == label0 and label1 in label_offsets:
                        # offset0 = offset1 - jump01
                        vote_offset = label_offsets[label1] - num_jump_01
                        votes.append((vote_offset, weight))

                if len(votes) < min_num_votes:
                    continue

                best_offset, confidence = weighted_integer_vote(votes)

                if best_offset is None:
                    continue

                channel.log(
                    f"Iteration {iteration + 1}: label {label_id}, "
                    f"num_votes={len(votes)}, "
                    f"best_offset={best_offset}, "
                    f"confidence={confidence:.3f}"
                )

                if confidence >= min_vote_confidence:
                    newly_resolved[label_id] = best_offset
                else:
                    channel.log(
                        f"Label {label_id} not accepted due to low vote confidence."
                    )

            if len(newly_resolved) == 0:
                channel.log(
                    "No new labels resolved in this iteration. "
                    "Stopping voting propagation."
                )
                break

            for label_id, offset in newly_resolved.items():
                label_offsets[label_id] = offset
                unresolved_labels.discard(label_id)

        channel.log(f"Resolved label offsets: {label_offsets}")

        if len(unresolved_labels) > 0:
            channel.log(
                f"WARNING: Some labels were not resolved: {sorted(unresolved_labels)}"
            )

        # Apply all corrections once.
        unw_out = np.array(unw_work, copy=True)

        for label_id, offset in sorted(label_offsets.items()):
            if label_id == self.label_ref:
                continue

            correction = offset * phase_jump_constant
            label_mask = self.labelImg == label_id

            channel.log(
                f"Applying correction to label {label_id}: "
                f"offset={offset}, correction={correction:.4f}, "
                f"num_pixels={np.count_nonzero(label_mask)}"
            )

            unw_out[label_mask] += correction

        if ramp_type is not None:
            channel.log("Adding ramp back after voting-based correction")
            unw_out += ramp

        unw_out[self.labelImg == 0] = 0

        channel.log("Finished unwrap_conn_comp_with_voting")

        return unw_out


def weighted_integer_vote(
    votes: List[Tuple[int, float]],
) -> Tuple[Optional[int], float]:
    """
    Choose the best integer offset from weighted votes.

    Parameters
    ----------
    votes : list of tuple
        Each item is (integer_offset, weight).

    Returns
    -------
    best_offset : int or None
        Selected integer offset.
    confidence : float
        Winning vote weight divided by total vote weight.
    """
    if len(votes) == 0:
        return None, 0.0

    vote_weight = {}

    for offset, weight in votes:
        offset = int(offset)
        weight = float(weight)
        vote_weight[offset] = vote_weight.get(offset, 0.0) + weight

    total_weight = sum(vote_weight.values())

    if total_weight <= 0:
        return None, 0.0

    best_offset = max(vote_weight, key=vote_weight.get)
    confidence = vote_weight[best_offset] / total_weight

    return best_offset, confidence


def deramp(data,
           mask_in=None,
           ramp_type='linear',
           max_num_sample=1e6,
           ignore_zero_value=True):
    """Remove ramp from input data matrix based on pixel marked by
    mask. Ignore data with NaN or zero value.

    Parameters
    ----------
    data : np.ndarray
        2D or 3D array of data to be deramped. If 3D, it's in the size of
        (num_date, length, width).
    mask_in : np.ndarray, optional
        2D array mask of pixels used for ramp estimation.
    ramp_type : str, optional
        Type of ramp to be estimated. Default is 'linear'.
        Possible options -
        'linear', 'quadratic', 'linear_range', 'linear_azimuth'
        'quadratic_range', 'quadratic_azimuth'
    max_num_sample : float, optional
        Maximum number of pixel samples, above which uniform sampling is
        applied to reduce sample size. Default is 1e6.
    ignore_zero_value : bool, optional
        Ignore pixels with zero values. Default is True.
        Recommended: True for phase data and False for offset data.

    Returns
    -------
    data_out : np.ndarray
        2D or 3D array of data after deramping.
    ramp : np.ndarray
        2D or 3D array of the estimated ramp.
    """
    dshape = data.shape
    length, width = dshape[-2:]
    num_pixel = length * width

    # prepare input data
    if len(dshape) == 3:
        data = np.moveaxis(data, 0, -1)
        data = data.reshape(num_pixel, -1)
        dmean = np.mean(data, axis=-1).flatten()
    else:
        data = data.reshape(-1, 1)
        dmean = np.array(data).flatten()

    # mask
    # 1. default
    if mask_in is None:
        mask_in = np.ones((length, width), dtype=np.float32)
    mask = (mask_in != 0).flatten()
    del mask_in

    # 2. ignore pixels with NaN and/or zero data value
    mask *= ~np.isnan(dmean)
    if ignore_zero_value:
        mask *= dmean != 0.
    del dmean

    # 3. for big dataset: uniformally sample the data for ramp estimation
    mask_sum = np.sum(mask) 
    if max_num_sample and mask_sum > max_num_sample:
        step = int(np.ceil(np.sqrt(mask_sum / max_num_sample)))
        if step > 1:
            sample_flag = np.zeros((length, width), dtype=np.bool_)
            sample_flag[int(step/2)::step,
                        int(step/2)::step] = 1
            mask *= sample_flag.flatten()
            del sample_flag

    # design matrix
    xx, yy = np.meshgrid(np.arange(0, width),
                         np.arange(0, length))
    xx = np.array(xx, dtype=np.float32).reshape(-1, 1)
    yy = np.array(yy, dtype=np.float32).reshape(-1, 1)
    ones = np.ones(xx.shape, dtype=np.float32)
    if ramp_type == 'linear':
        G = np.hstack((yy, xx, ones))
    elif ramp_type == 'quadratic':
        G = np.hstack((yy**2, xx**2, yy*xx, yy, xx, ones))
    elif ramp_type == 'linear_range':
        G = np.hstack((xx, ones))
    elif ramp_type == 'linear_azimuth':
        G = np.hstack((yy, ones))
    elif ramp_type == 'quadratic_range':
        G = np.hstack((xx**2, xx, ones))
    elif ramp_type == 'quadratic_azimuth':
        G = np.hstack((yy**2, yy, ones))
    else:
        raise ValueError(f'un-recognized ramp type: {ramp_type}')

    # estimate ramp
    X = np.dot(np.linalg.pinv(G[mask, :], rcond=1e-15), data[mask, :])
    ramp = np.dot(G, X)
    ramp = np.array(ramp, dtype=data.dtype)

    # do not change pixel with original zero value
    if ignore_zero_value:
        ramp[data == 0] = 0

    data_out = data - ramp
    if len(dshape) == 3:
        ramp = np.moveaxis(ramp, -1, 0)
        data_out = np.moveaxis(data_out, -1, 0)
    ramp = ramp.reshape(dshape)
    data_out = data_out.reshape(dshape)
    return data_out, ramp


def estimate_integer_jump(
    value0: float,
    value1: float,
    phase_jump_constant: float,
) -> Tuple[int, float, float]:
    """
    Estimate the integer jump needed to align value1 to value0.

    The returned num_jump satisfies approximately:

        value1 + num_jump * phase_jump_constant ~= value0

    Parameters
    ----------
    value0 : float
        Reference median value.
    value1 : float
        Target median value to be corrected.
    phase_jump_constant : float
        Phase jump constant, e.g., 2*pi, pi, or another user-defined value.

    Returns
    -------
    num_jump : int
        Integer jump applied to value1.
    correction : float
        Phase correction value.
    residual : float
        Absolute residual after correction.
    """
    diff_value = value1 - value0

    num_jump = -int(np.round(diff_value / phase_jump_constant))
    correction = phase_jump_constant * num_jump
    residual = abs((value1 + correction) - value0)

    return num_jump, correction, residual
