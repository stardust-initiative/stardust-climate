"""Model control helpers -- fix / unfix surface temperature, add bounds.

* ``fix_Ts``         / ``unfix_Ts``         -- pin or release surface temperature
* ``fix_q``          / ``unfix_q``          -- pin or release specific humidity
* ``fix_Tatm_trop``  / ``unfix_Tatm_trop``  -- pin or release tropospheric Tatm
* ``add_bounds``                             -- attach value-clipping Limiters

Implemented via climlab's ``Limiter`` adjustment subprocess, which clips
state variables to ``[min, max]`` after every timestep.

Usage
-----
::

    from climate_runs_ext.utils.model_control import fix_Ts, unfix_Ts, add_bounds
    fix_Ts(model)            # pin Ts to its current value
    unfix_Ts(model)          # let Ts evolve again
    add_bounds(model, {'q': (0.0, np.inf), 'Tatm': (150.0, np.inf)})
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import numpy as np
from climlab.process.limiter import Limiter


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Subprocess name used by fix_Ts / unfix_Ts -- kept as a module-level
# constant so that both functions always agree on the name.
_FIXED_TS_SUBPROCESS_NAME = 'FixedTs'
_FIXED_Q_SUBPROCESS_NAME = 'FixedQ'
_FIXED_TATM_TROP_SUBPROCESS_NAME = 'FixedTatmTrop'


# ---------------------------------------------------------------------------
# Surface temperature pinning
# ---------------------------------------------------------------------------

def _remove_generic_bound_limiter(model, var_name):
    """Delete the ``limiter_<var>`` subprocess (if any) from ``add_bounds``.

    When pinning a state variable via ``fix_Ts`` / ``fix_q`` / ``fix_Tatm_trop``,
    any pre-existing ``limiter_<var>`` (e.g. from ``add_bounds``) must be
    removed.  Otherwise climlab sums the adjustments from both Limiters,
    producing inconsistent state — particularly when the pinned snapshot
    lies outside the generic bounds (e.g. stratospheric q below a 5e-6
    floor).
    """
    subproc_name = f'limiter_{var_name}'
    if subproc_name in model.subprocess:
        del model.subprocess[subproc_name]


def fix_Ts(model):
    """Pin surface temperature to its current value.

    Adds a ``Limiter`` subprocess named ``'FixedTs'`` that clips ``Ts`` to
    ``[current_value, current_value]`` after every timestep.  Because both
    the minimum and maximum bounds equal the current value, the effect is
    that ``Ts`` cannot change.

    Any pre-existing ``limiter_Ts`` subprocess (from ``add_bounds``) is
    removed so that climlab does not sum conflicting adjustments.

    Parameters
    ----------
    model : climlab Process
        The coupled climate model.
    """
    # Guard: do nothing if already fixed
    if _FIXED_TS_SUBPROCESS_NAME in model.subprocess:
        return

    _remove_generic_bound_limiter(model, 'Ts')

    Ts = model.state['Ts']
    pinned = np.array(Ts).copy()

    lim = Limiter(
        state={'Ts': Ts},
        bounds={'Ts': {'minimum': pinned, 'maximum': pinned.copy()}},
        timestep=model.timestep,
    )
    model.add_subprocess(_FIXED_TS_SUBPROCESS_NAME, lim)


def unfix_Ts(model):
    """Allow surface temperature to evolve freely again.

    Removes the ``'FixedTs'`` ``Limiter`` subprocess that was added by
    ``fix_Ts``.  Safe to call even if Ts is not currently fixed.

    Parameters
    ----------
    model : climlab Process
        The coupled climate model.
    """
    if _FIXED_TS_SUBPROCESS_NAME in model.subprocess:
        del model.subprocess[_FIXED_TS_SUBPROCESS_NAME]


# ---------------------------------------------------------------------------
# Humidity pinning
# ---------------------------------------------------------------------------

def fix_q(model):
    """Pin specific humidity to its current value at every level.

    Adds a ``Limiter`` subprocess named ``'FixedQ'`` that clips ``q`` to
    ``[current_value, current_value]`` after every timestep.  Used to
    suppress water-vapor feedback in stratosphere-adjusted RF calculations,
    where humidity and tropospheric temperature are held fixed while only
    the stratosphere relaxes radiatively.

    Parameters
    ----------
    model : climlab Process
        The coupled climate model.  Must expose ``q`` in ``model.state``.
    """
    if _FIXED_Q_SUBPROCESS_NAME in model.subprocess:
        return

    _remove_generic_bound_limiter(model, 'q')

    q = model.state['q']
    pinned = np.array(q).copy()

    lim = Limiter(
        state={'q': q},
        bounds={'q': {'minimum': pinned, 'maximum': pinned.copy()}},
        timestep=model.timestep,
    )
    model.add_subprocess(_FIXED_Q_SUBPROCESS_NAME, lim)


def unfix_q(model):
    """Allow specific humidity to evolve freely again.

    Removes the ``'FixedQ'`` ``Limiter`` added by ``fix_q``.  Safe to call
    if ``q`` is not currently fixed.
    """
    if _FIXED_Q_SUBPROCESS_NAME in model.subprocess:
        del model.subprocess[_FIXED_Q_SUBPROCESS_NAME]


# ---------------------------------------------------------------------------
# Tropospheric Tatm pinning
# ---------------------------------------------------------------------------

def fix_Tatm_trop(model, p_trop_hPa=180.0, strato_Tmin=180.0):
    """Pin tropospheric atmospheric temperature to its current profile.

    Adds a ``Limiter`` subprocess named ``'FixedTatmTrop'`` that clips
    ``Tatm`` at all levels with ``lev >= p_trop_nearest`` to the current
    value (pinned), where ``p_trop_nearest`` is the model level nearest
    to *p_trop_hPa*.  Stratospheric levels (``lev < p_trop_nearest``) are
    left effectively free to relax radiatively, subject only to a
    numerical safety floor ``strato_Tmin`` (default 180 K) that prevents
    runaway cold excursions that would break ``qsat`` evaluations.

    Any pre-existing ``limiter_Tatm`` subprocess (from ``add_bounds``) is
    removed so that climlab does not sum conflicting adjustments — the
    combined troposphere-pin + stratospheric-floor logic replaces it.

    This is the core primitive for stratosphere-adjusted RF (SARF)
    experiments.  Combine with ``fix_Ts`` and ``fix_q`` to hold the entire
    surface + troposphere + water vapor field fixed.

    Parameters
    ----------
    model : climlab Process
        The coupled climate model.  Must expose ``Tatm`` in ``model.state``
        with an attached pressure-level domain.
    p_trop_hPa : float, default 180.0
        Target tropopause pressure.  The nearest model level is selected
        so the snapshot stays on the native grid.
    strato_Tmin : float, default 180.0
        Numerical safety floor for stratospheric temperature.  Prevents
        unphysical cold pools that would cause ``qsat`` overflow /
        divide-by-zero in coupled subprocesses.  Set to ``-np.inf`` to
        disable.
    """
    if _FIXED_TATM_TROP_SUBPROCESS_NAME in model.subprocess:
        return

    _remove_generic_bound_limiter(model, 'Tatm')

    Tatm = model.state['Tatm']
    lev = Tatm.domain.lev.points
    idx = int(np.argmin(np.abs(lev - p_trop_hPa)))
    p_trop_nearest = lev[idx]

    snap = np.array(Tatm).copy()
    # True where we pin (troposphere + surface-side), False in stratosphere.
    trop_mask = (lev >= p_trop_nearest)

    minimum = np.where(trop_mask, snap, strato_Tmin)
    maximum = np.where(trop_mask, snap, np.inf)

    lim = Limiter(
        state={'Tatm': Tatm},
        bounds={'Tatm': {'minimum': minimum, 'maximum': maximum}},
        timestep=model.timestep,
    )
    model.add_subprocess(_FIXED_TATM_TROP_SUBPROCESS_NAME, lim)


def unfix_Tatm_trop(model):
    """Allow tropospheric Tatm to evolve freely again.

    Removes the ``'FixedTatmTrop'`` ``Limiter`` added by ``fix_Tatm_trop``.
    Safe to call if not currently fixed.
    """
    if _FIXED_TATM_TROP_SUBPROCESS_NAME in model.subprocess:
        del model.subprocess[_FIXED_TATM_TROP_SUBPROCESS_NAME]


# ---------------------------------------------------------------------------
# General value bounds
# ---------------------------------------------------------------------------

def add_bounds(model, bound_dict):
    """Add value-clipping Limiter subprocesses from a bound_dict.

    Each entry in *bound_dict* creates a Limiter subprocess named
    ``'limiter_<var_name>'``.  If a subprocess with that name already
    exists it is replaced.

    Parameters
    ----------
    model : climlab Process
        The coupled climate model.
    bound_dict : dict
        ``{var_name: (lower, upper), ...}`` where either bound may be
        ``None`` (treated as ``-inf`` / ``+inf``).

    Examples
    --------
    >>> add_bounds(model, {
    ...     'q':    (0.0,   np.inf),   # humidity must be non-negative
    ...     'Tatm': (150.0, np.inf),   # temperature floor at 150 K
    ...     'Ts':   (150.0, np.inf),   # same for surface
    ... })
    """
    for var_name, (lo, hi) in bound_dict.items():
        subproc_name = f'limiter_{var_name}'

        # Remove any existing limiter for this variable
        if subproc_name in model.subprocess:
            del model.subprocess[subproc_name]

        state_var = model.state[var_name]
        minimum = -np.inf if lo is None else lo
        maximum =  np.inf if hi is None else hi

        lim = Limiter(
            state={var_name: state_var},
            bounds={var_name: {'minimum': minimum, 'maximum': maximum}},
            timestep=model.timestep,
        )
        model.add_subprocess(subproc_name, lim)
