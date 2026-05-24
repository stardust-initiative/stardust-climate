"""Prescribed (fixed) oceanic heat uptake process.

Ported from ``climate_runs/utils/utils_methods.py``:
``FixedOceanicHeatUptake``.

Applies a latitude-dependent ocean heat uptake (W/m²) as a cooling
tendency on the surface temperature.  The spatial pattern is fixed at
initialisation and does not evolve with the model state.

Usage
-----
::

    from climate_runs_ext.diagnostics.fixed_oceanic_heat_uptake import (
        FixedOceanicHeatUptake,
    )

    ohu = FixedOceanicHeatUptake(oha_lat=oha_profile, state=model.state)
    model.add_subprocess('OHU', ohu)
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import numpy as np

from climlab import TimeDependentProcess


# ---------------------------------------------------------------------------
# FixedOceanicHeatUptake
# ---------------------------------------------------------------------------

class FixedOceanicHeatUptake(TimeDependentProcess):
    """Prescribed ocean heat uptake as a surface temperature tendency.

    The tendency is computed as::

        dTs/dt = -oha_lat / heat_capacity

    where *oha_lat* is the prescribed heat uptake (W/m²) and
    *heat_capacity* comes from the surface domain (J/m²/K).

    Parameters
    ----------
    oha_lat : ndarray, shape (nlat,)
        Latitude-dependent ocean heat uptake (W/m²).  Positive values
        mean the ocean *absorbs* heat (cools the surface).
    **kwargs
        Passed to ``TimeDependentProcess.__init__``.  Must include
        ``state`` with a ``'Ts'`` variable.

    Diagnostics
    -----------
    ohu : ndarray
        The current ocean heat uptake field (W/m²).

    Properties
    ----------
    oha_lat : ndarray
        Settable; updates the prescribed heat-uptake field in-place.
    """

    def __init__(self, oha_lat, **kwargs):
        super(FixedOceanicHeatUptake, self).__init__(**kwargs)

        # --- validate inputs ----------------------------------------------
        assert len(oha_lat) == len(self.lat), (
            f"oha_lat length ({len(oha_lat)}) must match "
            f"number of latitudes ({len(self.lat)})"
        )
        assert hasattr(self.state, 'Ts'), (
            "state must contain a 'Ts' variable"
        )

        # --- broadcast oha_lat to match Ts shape --------------------------
        # Ts may be (nlat,) or (nlat, 1); ensure oha_lat has the same shape
        while len(oha_lat.shape) < len(self.state['Ts'].shape):
            oha_lat = oha_lat[..., np.newaxis]
        self._oha_lat = oha_lat + 0.0 * self.Ts

        # --- diagnostic field ---------------------------------------------
        self.ohu = oha_lat + 0.0 * self.Ts
        self.add_diagnostic('ohu', self.ohu)

    # --- tendency computation ---------------------------------------------

    def _compute(self):
        """Compute surface temperature tendency from prescribed OHU."""
        self.ohu[:] = self.oha_lat
        tendencies = {}
        tendencies['Ts'] = -self.oha_lat / self.Ts.domain.heat_capacity
        return tendencies

    # --- oha_lat property (settable) --------------------------------------

    @property
    def oha_lat(self):
        """Prescribed ocean heat uptake field (W/m²)."""
        return self._oha_lat

    @oha_lat.setter
    def oha_lat(self, value):
        """Update the OHU pattern in-place."""
        self._oha_lat[:] = value
