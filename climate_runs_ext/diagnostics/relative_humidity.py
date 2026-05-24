"""Relative humidity diagnostic process.

Ported from ``climate_runs/utils/utils_methods.py``: ``RelativeHumidity``.

Computes relative humidity as a diagnostic from the model state
(``Tatm``, ``q``) using the extended saturation-humidity function
from ``climlab_stardust_extension``.

Usage
-----
::

    from climate_runs_ext.diagnostics.relative_humidity import RelativeHumidity

    rh_proc = RelativeHumidity(state=model.state, timestep=model.timestep)
    print(rh_proc.rh)          # current relative humidity field
    print(rh_proc.rh_func(state))  # standalone call
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import numpy as np

from climlab.process.diagnostic import DiagnosticProcess
from climlab_stardust_extension.utils.thermo import qsat_extended


# ---------------------------------------------------------------------------
# RelativeHumidity diagnostic
# ---------------------------------------------------------------------------

class RelativeHumidity(DiagnosticProcess):
    """Diagnostic process that computes relative humidity from state.

    RH is computed as ``q / q_sat(T, p)`` where ``q_sat`` is obtained
    from the extended Clausius-Clapeyron formula.

    Parameters
    ----------
    state : AttrDict
        Must contain ``'Tatm'`` and ``'q'``.
    do_era5 : bool, optional
        Use ERA5-compatible Teten's formula (default False).
    do_simplified : bool, optional
        Use simplified denominator formula (default False).
    small : float, optional
        Regularisation parameter for numerical stability (default 0).
    **kwargs
        Passed to ``DiagnosticProcess.__init__``.

    Diagnostics
    -----------
    relative_humidity : ndarray
        Relative humidity field (dimensionless, typically 0–1).
    """

    def __init__(self, **kwargs):
        # --- extract qsat keyword arguments before calling super ----------
        qsat_kw_names = ('do_simplified', 'small', 'do_era5')
        qsat_params = {
            k: kwargs.pop(k) for k in qsat_kw_names if k in kwargs
        }
        super(RelativeHumidity, self).__init__(**kwargs)
        self.qsat_params = qsat_params

        # --- initial RH computation ---------------------------------------
        self._rh = self.rh_func(self.state, **self.qsat_params)
        self._compute_rh()
        self.add_diagnostic('relative_humidity', self._rh)

    # --- internal computation ---------------------------------------------

    def _compute_rh(self):
        """Recompute RH from the current model state."""
        self._rh[:] = self.rh_func(self.state, **self.qsat_params)

    def _compute(self):
        """Called by climlab on each diagnostic step."""
        self._compute_rh()
        return {}

    # --- public property --------------------------------------------------

    @property
    def rh(self):
        """Current relative humidity field (recomputed on access)."""
        self._compute_rh()
        return self._rh

    # --- standalone function ----------------------------------------------

    @staticmethod
    def rh_func(state, **kwargs):
        """Compute relative humidity from a state dict.

        Parameters
        ----------
        state : AttrDict
            Must have ``'Tatm'`` and ``'q'`` keys.
        **kwargs
            Forwarded to ``qsat_extended`` (``do_era5``, ``small``,
            ``do_simplified``).

        Returns
        -------
        ndarray
            Relative humidity (dimensionless).
        """
        lev = state['Tatm'].domain.lev.points
        # For 2D (lat, lev) states, broadcast lev to (1, nlev)
        if len(state['Tatm'].shape) > 1:
            lev = lev[np.newaxis, :]
        qsaturation = qsat_extended(state['Tatm'], lev, **kwargs)
        return state['q'] / qsaturation
