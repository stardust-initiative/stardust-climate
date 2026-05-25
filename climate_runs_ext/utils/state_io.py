"""State I/O -- save / load model state to/from npz and xarray.

Ported from ``climate_runs/utils/utils_methods.py``:

* ``model_to_npz``            -- serialise state + diagnostics + time-averages
* ``update_model_from_xr``    -- overwrite state from an xarray Dataset
* ``update_model_from_file``  -- overwrite state from npz / dict / model
* ``iteratively_update_internal`` -- walk the process tree and propagate
  reference state

Usage
-----
::

    from climate_runs_ext.utils.state_io import (
        model_to_npz, update_model_from_xr, update_model_from_file,
        iteratively_update_internal,
    )
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import sys
import numpy as np
from copy import deepcopy

from climlab import TimeDependentProcess
from climlab.utils import walk


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

def model_to_npz(model, filename='', skip_compute=False, **kwargs):
    """Serialize model state, diagnostics and time-averages.

    Three namespaces are used for the keys:

    * ``STATE_<name>``    -- state variables (Ts, Tatm, q, ...)
    * ``AXES_<name>``     -- coordinate axes (lev, lat)
    * ``DIAG_<name>``     -- diagnostic fields
    * ``TIMEAVE_<name>``  -- time-averaged fields

    Parameters
    ----------
    model : climlab Process
        The climate model to serialise.
    filename : str
        If non-empty, write to this ``.npz`` path.  Otherwise return a
        plain Python dict.
    skip_compute : bool
        If *False* (default), call ``compute_diagnostics()`` first so
        that diagnostic arrays are up to date.
    **kwargs
        Extra key-value pairs to include in the saved file (useful for
        metadata like commit hashes).

    Returns
    -------
    dict or None
        The combined dict when *filename* is empty; ``None`` otherwise.
    """
    if not skip_compute:
        model.compute_diagnostics()

    combined_dict = {}

    # --- coordinate axes --------------------------------------------------
    for k_state in model.state.keys():
        for k_axes, axes_obj in model.state[k_state].domain.axes.items():
            combined_dict[f'AXES_{k_axes}'] = axes_obj.points

    # --- state variables --------------------------------------------------
    for k_state in model.state.keys():
        combined_dict[f'STATE_{k_state}'] = np.array(model.state[k_state])

    # --- diagnostic fields ------------------------------------------------
    for k_diag, val_diag in model.diagnostics.items():
        combined_dict[f'DIAG_{k_diag}'] = val_diag

    # --- time-averaged fields ---------------------------------------------
    if hasattr(model, 'timeave'):
        for k_diag, val_diag in model.timeave.items():
            combined_dict[f'TIMEAVE_{k_diag}'] = val_diag

    # --- save or return ---------------------------------------------------
    if filename == '':
        return combined_dict
    else:
        np.savez(filename, **combined_dict, **kwargs)


# ---------------------------------------------------------------------------
# Loading from xarray
# ---------------------------------------------------------------------------

def update_model_from_xr(model, xr_obj, do_compute=False):
    """Overwrite model state and time-averages from an xarray Dataset.

    Each key in ``model.state`` is looked up in *xr_obj* and the values
    are copied in-place.  The ``timeave`` dict is also rebuilt from the
    Dataset so that time-averaged diagnostics are consistent.

    Parameters
    ----------
    model : climlab Process
        Target model (modified in-place).
    xr_obj : xarray.Dataset
        Dataset whose variables match the model's state and diagnostic keys.
    do_compute : bool
        If *True*, call ``compute_diagnostics()`` after loading.
    """
    # --- overwrite state variables ----------------------------------------
    state = model.state
    for k in state.keys():
        state[k][:] = xr_obj[k].values

    # --- rebuild time-averaged fields -------------------------------------
    # Use the union of diagnostics + state as the "shape template".  If the
    # xr_obj is missing some of the model's diagnostics (e.g. loading a
    # netCDF saved by an older/predecessor fork with a slightly different
    # diagnostic list), we just skip them rather than failing — later
    # compute_diagnostics() will repopulate the instantaneous values.
    diag = {}
    diag.update(model.diagnostics)
    diag.update(model.state)
    if hasattr(model, 'timeave'):
        new_timeave = {}
        for k, v in diag.items():
            if k in xr_obj:
                new_timeave[k] = xr_obj[k].values + 0.0 * v
        model.timeave = new_timeave

    if do_compute:
        model.compute_diagnostics()


# ---------------------------------------------------------------------------
# Loading from npz / dict / model
# ---------------------------------------------------------------------------

def update_model_from_file(model, **kwargs):
    """Load model state from an npz file, dict, or another model.

    Exactly one of the following keyword arguments must be provided:

    * ``model_source`` -- another climlab model to deep-copy from.
    * ``files_dict``   -- a dict as returned by ``model_to_npz``.
    * ``file_path``    -- path to a ``.npz`` file.

    After loading, subprocesses that implement ``update_internal_fields``
    are notified of the new reference state (only when ``model_source``
    is used).

    Parameters
    ----------
    model : climlab Process
        Target model (modified in-place).
    do_compute_diag : bool
        Call ``compute_diagnostics`` before loading (default True).
    load_timeave : bool
        After loading, overwrite state with time-averages (default False).

    Returns
    -------
    dict
        Any keys from the source that did not match the STATE/DIAG/TIMEAVE
        namespace prefixes (e.g. user metadata).
    """
    use_ref_state = False
    do_compute_diag = kwargs.get('do_compute_diag', True)
    load_timeave = kwargs.get('load_timeave', False)

    # --- determine the data source ----------------------------------------
    if 'model_source' in kwargs:
        model_source = deepcopy(kwargs['model_source'])
        ref_state = model_source.state
        # If the source has time-averages, fold them into the ref state
        if hasattr(model_source, 'timeave'):
            for key in ref_state.keys():
                if key in model_source.timeave:
                    ref_state[key][:] = model_source.timeave[key][:]
        files_dict = model_to_npz(model_source, **kwargs)
        use_ref_state = True
    elif 'files_dict' in kwargs:
        files_dict = kwargs['files_dict']
    elif 'file_path' in kwargs:
        files_dict = dict(np.load(kwargs['file_path']))
    else:
        sys.exit(
            'Provide an input source: model_source, files_dict, or file_path'
        )

    if do_compute_diag:
        model.compute_diagnostics()

    # --- apply the loaded data to the model -------------------------------
    model_dict = model.__dict__
    output_dict = {}

    for k, v in files_dict.items():
        i0 = k.find('_')
        if i0 < 0:
            output_dict[k] = v
            continue
        prefix = k[:i0]
        name = k[i0 + 1:]

        if prefix == 'STATE':
            # Verify that axis grids match the saved file
            for k_axes, axes in model.state[name].domain.axes.items():
                if len(axes.points) > 1:
                    assert np.all(
                        axes.points == files_dict[f'AXES_{k_axes}']
                    ), f'{k_axes} does not match saved file'
            model_dict[name][:] = v

        elif prefix == 'DIAG':
            if name in model.diagnostics:
                model.diagnostics[name][:] = v

        elif prefix == 'TIMEAVE':
            if name in model_dict:
                if not hasattr(model, 'timeave'):
                    model.timeave = {}
                model.timeave[name] = v + 0.0 * model_dict[name]

        elif prefix != 'AXES':
            # Keys that don't match any namespace → return to caller
            output_dict[k] = v

    # --- propagate the reference state to subprocesses --------------------
    if use_ref_state:
        iteratively_update_internal(model, ref_state)

    # --- optionally overwrite state with time-averages --------------------
    if load_timeave and hasattr(model, 'timeave'):
        for k in model.state.keys():
            if k in model.timeave:
                model.state[k][:] = model.timeave[k][:]

    return output_dict


# ---------------------------------------------------------------------------
# Process-tree walker
# ---------------------------------------------------------------------------

def iteratively_update_internal(model, ref_state):
    """Walk the process tree and call ``update_internal_fields`` where present.

    This is the mechanism by which diagnostic processes (e.g. CloudFeedback)
    learn about a new reference state.  Any subprocess that exposes an
    ``update_internal_fields(ref_state=...)`` method will be called.

    Parameters
    ----------
    model : climlab Process
        Root of the process hierarchy.
    ref_state : dict or AttrDict
        Reference state to propagate.
    """
    for _, proc, _ in walk.walk_processes(model):
        if hasattr(proc, 'update_internal_fields'):
            proc.update_internal_fields(ref_state=ref_state)
