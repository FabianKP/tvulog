import numpy as np


class ForwardOperator:
    """
    Abstract base class of a forward operator, i.e. something that maps stellar distribution functions to spectra.
    """
    m_f: int              # Number of metallicity bins.
    n_f: int              # Number of age bins.
    dim_theta: int        # Dimension of theta_v
    dim_y: int            # Number of data points (with mask applied).
    dim_y_unmasked: int   # Number of data points (without mask). Should be the same as `mask.size`.

    def fwd(self, f: np.ndarray) -> np.ndarray:
        """
        Evaluates the forward operator on a given stellar distribution `f`.

        Parameters
        ----------
        f : shape (mn, )
            The flattened stellar distribution.

        Returns
        -------
        y : shape (k, )
        """
        raise NotImplementedError

    @property
    def mat(self) -> np.ndarray:
        """
        Returns the matrix representation of the forward operator.

        Returns
        -------
        arr : shape (k, mn)
        """
        raise NotImplementedError

    def fwd_unmasked(self, f: np.ndarray) -> np.ndarray:
        """
        Evaluates the forward operator on a given stellar distribution `f`, but also the masked values are returned
        (i. e. the ones that are otherwise excluded because there is too much noise).

        Parameters
        ----------
        f : shape (mn, )
            The flattened stellar distribution.
        Returns
        -------
        y : shape (k2,)
        """
        raise NotImplementedError

    @property
    def mat_unmasked(self) -> np.ndarray:
        """
        Returns the matrix representation of the unmasked forward operator, i.e. corresponding to `self.fwd_unmasked`.

        Returns
        -------
        arr : shape (k2, mn)
        """
        raise NotImplementedError
