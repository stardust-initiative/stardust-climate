"""State construction helpers.

Ported from ``climate_runs/utils/utils_methods.py``:

* ``lev_grid_construct``   -- build a non-uniform pressure-level grid
* ``create_state``         -- build a climlab state from numpy arrays
* ``create_state_for_conv`` -- same, but with humidity for convection models

Usage
-----
::

    from climate_runs_ext.utils.state_helpers import (
        lev_grid_construct, create_state, create_state_for_conv,
    )
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import numpy as np
from scipy import interpolate

import climlab
from climlab.domain import field, domain
from climlab.utils.attrdict import AttrDict


# ---------------------------------------------------------------------------
# Pressure-level grid
# ---------------------------------------------------------------------------

def lev_grid_construct(n, dp0, dps, p0=0.0, ps=1000.0):
    """Build a non-uniform pressure-level grid using PCHIP interpolation.

    The grid is denser near the TOA (layer thickness ≈ *dp0*) and surface
    (≈ *dps*), with smooth stretching in between.

    Parameters
    ----------
    n   : int     Number of levels.
    dp0 : float   Layer thickness near the top of the atmosphere (hPa).
    dps : float   Layer thickness near the surface (hPa).
    p0  : float   Pressure at the top boundary (hPa, default 0).
    ps  : float   Pressure at the surface boundary (hPa, default 1000).

    Returns
    -------
    ndarray, shape (n,)
        Pressure at level midpoints (hPa), monotonically increasing
        from TOA to surface.
    """
    # Four control points: top edge, first interior, last interior, bottom edge
    n_sampling = np.array([0, 1, n - 1, n])
    p_bound_sampling = np.array([p0, p0 + dp0, ps - dps, ps])

    # Monotone cubic interpolation to get all n+1 boundary pressures
    f = interpolate.PchipInterpolator(n_sampling, p_bound_sampling)
    p_bound = f(np.arange(n + 1))

    # Level midpoints
    p = 0.5 * (p_bound[:-1] + p_bound[1:])
    return p


# ---------------------------------------------------------------------------
# State construction
# ---------------------------------------------------------------------------

def create_state(Tatm0, Ts0, lev, lat, water_depth=10.0):
    """Create a climlab state dict from numpy arrays.

    Builds the appropriate domain (single-column or zonal-mean) from
    the *lev* and *lat* grids and wraps the temperature arrays as
    ``climlab.domain.field.Field`` objects.

    Parameters
    ----------
    Tatm0 : ndarray       Atmospheric temperature profile(s).
    Ts0   : ndarray       Surface temperature(s).
    lev   : ndarray       Pressure levels (hPa).
    lat   : ndarray       Latitude grid (degrees).
    water_depth : float   Slab-ocean depth in metres (default 10).

    Returns
    -------
    AttrDict  with keys ``'Ts'`` and ``'Tatm'``.
    """
    num_lat = len(lat)
    num_lev = len(lev)

    # --- choose the right domain type ------------------------------------
    if num_lat == 1:
        sfc, atm = domain.single_column(
            water_depth=water_depth, num_lev=num_lev, lev=lev,
        )
    else:
        sfc, atm = domain.zonal_mean_column(
            water_depth=water_depth, num_lev=num_lev, lev=lev,
            num_lat=num_lat, lat=lat,
        )

    # --- wrap arrays as climlab Fields ------------------------------------
    Ts   = field.Field(Ts0,   domain=sfc)
    Tatm = field.Field(Tatm0, domain=atm)

    state = AttrDict()
    state['Ts']   = Ts
    state['Tatm'] = Tatm
    return state


def create_state_for_conv(lev, lat, Ts, Tatm, q, water_depth=10.0):
    """Create a climlab state with humidity for convection models.

    Builds on ``climlab.column_state`` and injects ``q`` as a state
    variable.  This is the state layout expected by convective
    parameterisation schemes (SBM, LSC) and moist dynamics.

    Parameters
    ----------
    lev   : ndarray   Pressure levels (hPa).
    lat   : ndarray   Latitude grid (degrees).
    Ts    : ndarray   Surface temperature(s).
    Tatm  : ndarray   Atmospheric temperature(s).
    q     : ndarray   Specific humidity.
    water_depth : float   Slab-ocean depth in metres (default 10).

    Returns
    -------
    AttrDict  with keys ``'Ts'``, ``'Tatm'``, and ``'q'``.
    """
    full_state = climlab.column_state(
        lev=lev, lat=lat, water_depth=water_depth,
    )

    # Assign values — handle the (nlat,1) vs (nlat,) shape for Ts
    full_state['Tatm'][:] = Tatm
    if full_state['Ts'].ndim == 2 and np.ndim(Ts) == 1:
        full_state['Ts'][:] = np.atleast_2d(Ts).T
    else:
        full_state['Ts'][:] = Ts
    full_state['q'] = q

    return full_state
