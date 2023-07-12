
from dataclasses import dataclass
import numpy as np
from typing import Sequence, Tuple

from ._blob_set import BlobSet
from ._mode_set import ModeSet
from ._extract_modes import extract_modes
from ._set_projection import scale_projection


@dataclass
class ExtractionResult:
    """
    Container class for the result of TV-ULoG.
    """
    mode_sets: Sequence[ModeSet]                    # The mode sets (in scale space).
    blob_sets: Sequence[Tuple[BlobSet, BlobSet]]    # The projections of the mode sets (in signal domain).


class ExtractorVisualizer1D:
    """
    Extracts and visualizes the blob regions from the minimizer of the TV-ULoG optimization problem for 1D-signals.
    Why is this a class and not a function?
    """
    def __init__(self, sigmas: Sequence[float]):
        """
        Parameters
        ----------
        sigmas
            The standard deviation of the scale-space representation.
        """
        self._sigmas = sigmas

    def extract_blobs(self, normlap: np.ndarray, rthresh: float = 0.5, maxima_thresh: float = 0.01) -> ExtractionResult:
        """
        Given a TV-ULoG minimizer, extracts the blob regions using the procedure described in the paper.

        Parameters
        ----------
        normlap: shape (k, n)
            The scale-normalized Laplacian of the minimizer.
        rthresh
            Relative threshold for plateau-extraction.
        maxima_thresh
            Threshold for maxima detection. A local maximum is only detected if the intensity is at least
            `maxima_tresh * 100`% of the global maximum of `-normlap`.

        Returns
        -------
        result
            Object of type `ExtractionResult`.
        """
        # Extract signal shape.
        signal_shape = normlap[0].shape
        # Detect modes in **negative** scale-normalized Laplacian.
        neg_normlap_blanket = (-normlap).clip(min=0.)
        mode_sets = extract_modes(arr=neg_normlap_blanket, rthresh=rthresh, maxima_thresh=maxima_thresh, eps_rel=1e-3)
        blob_set_tuples = [scale_projection(mode_set=mode, shape=signal_shape, sigmas=self._sigmas) for mode in
                           mode_sets]
        result = ExtractionResult(mode_sets=mode_sets, blob_sets=blob_set_tuples)
        return result


class ExtractorVisualizer2D:
    """
    Extracts and visualizes the blob regions from the minimizer of the TV-ULoG optimization problem for 1D-signals.
    Why is this a class and not a function?
    """
    def __init__(self, sigmas: Sequence[float], width_to_height: float):
        """
        Parameters
        ----------
        sigmas
            The standard deviations for the scale-space representation.
        width_to_height
            Width-to-height ratio for the desired blobs.
        """
        self._sigmas = sigmas
        self._width_to_height = width_to_height

    def extract_blobs(self, normlap: np.ndarray, rthresh: float = 0.5, maxima_thresh: float = 0.01) -> ExtractionResult:
        """
        Given a TV-ULoG minimizer, extracts the detected blob regions using the procedure described in the paper.

        Parameters
        ----------
        normlap: shape (k, n)
            The scale-normalized Laplacian of the minimizer.
        rthresh
            Relative threshold for plateau-extraction.
        maxima_thresh
            Threshold for maxima detection. A local maximum is only detected if the intensity is at least
            `maxima_tresh * 100`% of the global maximum of `-normlap`.

        Returns
        -------
        result
            Object of type `ExtractionResult`.
        """
        signal_shape = normlap[0].shape
        # Detect modes in **negative** scale-normalized Laplacian.
        neg_normlap_blanket = (-normlap).clip(min=0.)
        mode_sets = extract_modes(arr=neg_normlap_blanket, rthresh=rthresh, maxima_thresh=maxima_thresh, eps_rel=1e-3)
        blob_set_tuples = [scale_projection(mode_set=mode, shape=signal_shape, sigmas=self._sigmas,
                                            width_to_height=self._width_to_height) for mode in mode_sets]
        result = ExtractionResult(blob_sets=blob_set_tuples, mode_sets=mode_sets)
        return result
