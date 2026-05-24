"""Radiation diagnostic helpers.

Ported from ``climate_runs/utils/diagnostic_methods.py`` and
``climate_runs/utils/utils_methods.py``:

* ``get_inst_diag``          -- instantaneous radiative forcing diagnostic
* ``switch_rad_cld_fb_param`` -- set parameters on radiation + SW/LW

Usage
-----
::

    from climate_runs_ext.diagnostics.radiation_helpers import (
        get_inst_diag, switch_rad_cld_fb_param,
    )
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

from copy import deepcopy


# ---------------------------------------------------------------------------
# switch_rad_cld_fb_param
# ---------------------------------------------------------------------------

def switch_rad_cld_fb_param(rad, **kwargs):
    """Set parameter values on a radiation process and its SW/LW children.

    For each ``key=value`` pair in *kwargs*, the attribute is set on *rad*
    and on ``rad.subprocess['SW']`` and ``rad.subprocess['LW']`` (if
    the attribute exists on those objects).

    Parameters
    ----------
    rad : climlab Process
        The RRTMG radiation process (or any radiation wrapper that has
        ``'SW'`` and ``'LW'`` subprocesses).
    **kwargs
        Attribute name / value pairs to set.
    """
    for r in [rad, rad.subprocess['SW'], rad.subprocess['LW']]:
        r_dict = r.__dict__
        for k, v in kwargs.items():
            if k in r_dict:
                r_dict[k] = v


# ---------------------------------------------------------------------------
# get_inst_diag
# ---------------------------------------------------------------------------

def get_inst_diag(rad0, rad_ref0, rad_param_change_dict=None,
                  do_copy_ref=True):
    """Compute instantaneous radiative forcing by differencing two radiation
    processes.

    Deep-copies both radiation processes, optionally copies cloud fields from
    the reference to the perturbed process, then computes diagnostics on each
    and returns the difference (perturbed - reference) as an xarray Dataset.

    Parameters
    ----------
    rad0 : climlab Process
        The perturbed radiation process (e.g. with aerosol layer or 2xCO2).
    rad_ref0 : climlab Process
        The reference radiation process (baseline).
    rad_param_change_dict : dict or None
        Optional parameter overrides to apply via
        ``switch_rad_cld_fb_param`` on both copies before computing
        diagnostics (e.g. ``{'n_rrtmg_repeat': 100}``).
    do_copy_ref : bool
        If *True* (default), copy cloud fields and state from reference
        to perturbed so the only difference is the forcing agent.

    Returns
    -------
    xarray.Dataset
        ``diag(perturbed) - diag(reference)``.  Useful diagnostics include
        ``'ASR'`` (absorbed shortwave) and ``'OLR'`` (outgoing longwave).
        The instantaneous RF is ``diag['ASR'] - diag['OLR']``.
    """
    if rad_param_change_dict is None:
        rad_param_change_dict = {}

    rad = deepcopy(rad0)
    rad_ref = deepcopy(rad_ref0)

    rad_ref_dict = rad_ref.__dict__
    rad_dict = rad.__dict__

    # Copy cloud fields from reference → perturbed
    if do_copy_ref:
        for k in ['cldfrac', 'clwp', 'ciwp', 'r_ice', 'r_liq']:
            if k in rad_dict and k in rad_ref_dict:
                rad_dict[k][:] = rad_ref_dict[k]

    # Apply parameter overrides
    if rad_param_change_dict:
        switch_rad_cld_fb_param(rad_ref, **rad_param_change_dict)
        switch_rad_cld_fb_param(rad, **rad_param_change_dict)

    # Compute reference diagnostics
    rad_ref.compute_diagnostics()

    # Copy state from reference → perturbed (so only forcing agent differs)
    if do_copy_ref:
        for k in ['specific_humidity', 'Ts', 'Tatm']:
            if k in rad_dict and k in rad_ref_dict:
                rad_dict[k][:] = rad_ref_dict[k][:]

    # Compute perturbed diagnostics
    rad.compute_diagnostics()

    # Return difference
    diag_ref = rad_ref.to_xarray(diagnostics=True, timeave=False)
    diag = rad.to_xarray(diagnostics=True, timeave=False) - diag_ref
    return diag
