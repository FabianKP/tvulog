
import numpy as np
from typing import List, Union, Sequence

from ..laplacian_of_gaussians import laplacian_of_gaussians
from ..laplacian_of_gaussians import best_blob_first, stack_to_blobs
from ..gaussian_blob import GaussianBlob, compute_overlap
from ..gaussian_scale_space import scale_normalized_laplacian
from ._compute_blanket import compute_blanket
from ._ulog_result import ULoGResult


class ULoG:
    """
    Performs uncertainty-aware blob blob_detection with automatic scale selection.
    """
    def __init__(self, sigmas: Sequence[float], width_to_height: float = 1.):
        """
        Parameters
        ----------
        sigmas
            The list of standard deviations.
        width_to_height
            Ratio of blob width to blob height.
        """
        self._sigmas = sigmas
        self._width_to_height = width_to_height

    def compute_blankets(self, lower_bound: np.array, upper_bound: np.ndarray) -> np.array:
        """
        Solve the ULoG-minimization problem in the given scale-space tube.

        Parameters
        ----------
        lower_bound : shape (K, M, N)
            The lower bound of the scale-space tube.
        upper_bound : shape (K, M, N)
            The upper bound of the scale-space tube.

        Returns
        -------
        minimizer : shape (K, M, N)
            The minimizer in the scale-space tube.
        """
        # Compute minimizer argmin_f ||Lap(f)||_2^2 s.t. lower <= f <= upper.
        blanket_list = []
        for lower, upper in zip(lower_bound, upper_bound):
            # Compute minimizer at scale t.
            blanket = compute_blanket(lower, upper, self._width_to_height)
            blanket_list.append(blanket)
            # Return minimizer stack as array.
        minimizer = np.array(blanket_list)
        return minimizer

    def extract_blobs(self, minimizer: np.ndarray, reference: np.array, rthresh1: float = 0.05, rthresh2: float = 0.1,
                      overlap1: float = 0.5, overlap2: float = 0.5, exclude_max_scale: bool = False) -> ULoGResult:
        """
        Performs feature matching to determine which features in the MAP estimate are significant with respect to the
        posterior distribution defined by the given model.
        - A feature not present in ``ansatz`` cannot be significant.
        - Also, the size and position of the possible significant features are determined by ``ansatz``.
        - Hence, we only have to determine which features are significant, and the resolution of the signficant
            features.

        Parameters
        ----------
        minimizer : shape (K, M, N)
            The minimizer of the ULoG minimization problem.
        reference : shape (M, N)
            The reference image for which significant features need to be determined, e.g. the MAP estimate.
        rthresh1
            The relative threshold for feature strength that a blob has to satisfy in order to count as detected.
        rthresh2
            A "significant" blob must have strength equal to more than the factor `rthresh2` of the strength
            of the corresponding MAP blob.
        overlap1
            The maximum allowed overlap for blobs in the same image.
        overlap2 : The relative overlap that is used in the matching of the blobs in the reference image to the
            minimizer-blobs.
        exclude_max_scale : bool
            If True, blobs cannot be detected at the maximal scale.
        """
        # Apply scale-normalized Laplacian to minimizer stack.
        blanket_laplacian_stack = scale_normalized_laplacian(ssr=minimizer, sigmas=self._sigmas,
                                                             width_to_height=self._width_to_height)
        # Compute minimizer-blobs.
        blanket_blobs = stack_to_blobs(scale_stack=minimizer, log_stack=blanket_laplacian_stack, sigmas=self._sigmas,
                                       width_to_height=self._width_to_height, rthresh=rthresh2,
                                       max_overlap=overlap1, exclude_max_scale=exclude_max_scale)
        # Identify features in reference image.
        log_result = laplacian_of_gaussians(image=reference, sigmas=self._sigmas,
                                            width_to_height=self._width_to_height, max_overlap=overlap1,
                                            rthresh=rthresh1,  exclude_max_scale=exclude_max_scale)
        reference_blobs = log_result.blobs
        # Perform mathcing between significant blobs and reference blobs.
        mapped_pairs, n_mapped = self._match_blobs(reference_blobs=reference_blobs, blanket_blobs=blanket_blobs,
                                                   overlap=overlap2)
        # Return ULoGResult-object.
        result = ULoGResult(image=reference, mapped_pairs=mapped_pairs)
        return result

    def _match_blobs(self, reference_blobs: List[GaussianBlob], blanket_blobs: List[GaussianBlob], overlap: float):
        """
        Given a set of reference blobs and a set of minimizer-blobs, we look for matches.
        """
        blanket_blobs_sorted = best_blob_first(blanket_blobs)
        mapped_pairs = []
        # Iterate over the reference blobs.
        n_mapped = 0
        for blob in reference_blobs:
            # Find the feature matching the map_feature
            matching_blob = self._find_blob(blob, blanket_blobs_sorted, overlap)
            # Check that blobs really have the right overlap
            if matching_blob is not None:
                overl = compute_overlap(blob, matching_blob)
                n_mapped += 1
                assert overl >= overlap
            # Add corresponding pair to "mapped_pairs".
            mapped_pairs.append(tuple([blob, matching_blob]))

        return mapped_pairs, n_mapped

    @staticmethod
    def _find_blob(blob: GaussianBlob, blobs: List[GaussianBlob], overlap: float) -> Union[GaussianBlob, None]:
        """
        Find a blob in a given collection of blobs.
        A blob is mapped if the overlap to another blob is more than a given threshold.
        If there is more than one matching blob, then the FIRST MATCH is selected.
        If the relative overlap is 1 for more than one blob in ``blobs``, the first matching feature is selected.
        """
        # Iterate over blobs.
        found = None
        for candidate in blobs:
            candidate_overlap = compute_overlap(blob, candidate)
            enough_overlap = (candidate_overlap >= overlap)
            if enough_overlap:
                found = candidate
                break
        return found
