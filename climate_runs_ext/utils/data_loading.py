"""Data loading from GitHub repositories via pooch caching.

Ported from ``climate_runs/utils/utils_methods.py``:

* ``load_xr_from_repo`` -- download and open a NetCDF / numpy file from
  the climate database GitHub repository.
* ``era5_grid``          -- return the standard ERA5 lat/lev grids.

The download is handled by ``climlab_stardust_extension``'s
``load_repo_table`` (which uses ``pooch`` under the hood) so that repeated
calls hit the local cache.

Usage
-----
::

    from climate_runs_ext import load_project_config
    from climate_runs_ext.utils.data_loading import load_xr_from_repo, era5_grid

    cfg = load_project_config()
    ds  = load_xr_from_repo('Monthly_Zonal_Cloud_Cover_2008_2017', cfg)
    lev, lat = era5_grid(cfg)
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import numpy as np
import xarray as xr

from climlab_stardust_extension.utils.file_handling import load_repo_table


# ---------------------------------------------------------------------------
# GitHub-based data loading
# ---------------------------------------------------------------------------

# Legacy climate-database file_key -> stardust_2d_inputs registry key. Datasets
# in this map are served by the input-data engine's content-addressed store;
# every other file_key falls back to the legacy GitHub/pooch path until chunk 2
# of the migration registers it. Keys are the extension-less file_key strings
# passed by the model's loaders.
_REGISTRY_KEY_MAP = {
    'Monthly_Zonal_Cloud_Cover_2008_2017': 'monthly_zonal_cloud_cover',
    'Monthly_Zonal_Cloud_content_2008_2017': 'monthly_zonal_cloud_content',
    'Monthly_Zonal_o3_2008_2017': 'monthly_zonal_o3',
    'Monthly_Zonal_Variables_2008_2017': 'monthly_zonal_variables',
    'Monthly_Zonal_Srf_Params_2008_2017': 'monthly_zonal_srf_params',
    'Monthly_Zonal_Toa_and_Surf_balance_2008': 'monthly_zonal_surface_sw_balance',
    'diffusion_ml_TEMdecomp_raw_2008_2017': 'eddy_diffusivity',
    'tropopause_ERA5_zonal_mean_2008_2017': 'tropopause',
    # Private superset — loaded by the full-physics / ERF / cloud-feedback /
    # transport-parameter paths (not the minimal SARF). Served from the private
    # GCS store; marked public:false in the registry so they never ship publicly.
    'Monthly_Zonal_RH_2008_2017': 'monthly_zonal_rh',
    'Monthly_Zonal_SURFACE_FLUXES_2008_2017': 'monthly_zonal_surface_fluxes',
    'Monthly_Zonal_balance_2008_2017': 'monthly_zonal_balance',
    'Monthly_Zonal_Srf_Pres_2008_2017': 'monthly_zonal_srf_pres',
    'Mean_meridional_moist_diffusion': 'mean_meridional_moist_diffusion',
    './mse_files/Transport_2008_2017': 'mse_transport',
    './mse_files/era5_q_h_2008_2017': 'mse_q_h',
    'cloud_sensitivity_files/cloud_param_historic_sensitivity_ver2': 'cloud_param_sensitivity_monthly',
    'cloud_sensitivity_files/cloud_param_historic_sensitivity_ver2_MultiYear': 'cloud_param_sensitivity_multiyear',
}


def load_xr_from_repo(file_key, config, file_type='xarray'):
    """Load a dataset from the climate database GitHub repository.

    The file is downloaded (and cached locally via pooch) from the
    ``climate_database_files`` repository whose URL is specified in
    *config*.

    Parameters
    ----------
    file_key : str
        Relative path within the repository **without** extension.
        Example: ``'Monthly_Zonal_Cloud_Cover_2008_2017'``.
    config : dict
        Project configuration as returned by ``load_project_config()``.
        Must contain ``climate_database_files_http``,
        ``climate_database_token``, and ``proj_name``.
    file_type : str
        ``'xarray'``  → append ``.nc`` and return ``xarray.Dataset``
        ``'numpy'``   → append ``.npz`` and return ``numpy.NpzFile``

    Returns
    -------
    xarray.Dataset  or  numpy.NpzFile
    """
    assert file_type in ('xarray', 'numpy'), (
        f"file_type={file_type!r} not supported (use 'xarray' or 'numpy')"
    )

    # New path (release-workflow standard, Phase C): datasets registered in the
    # stardust_2d_inputs engine are served from its content-addressed store
    # (local / GCS / Zenodo, chosen by the engine's own config) and resolved
    # through the pinned transport-paper release, hash-verified on fetch. The
    # engine config is located via STARDUST_2D_INPUTS_CONFIG (see the engine's
    # config.example.json). Only xarray datasets are migrated so far; .npz
    # inputs and any file_key not yet in _REGISTRY_KEY_MAP use the legacy
    # GitHub/pooch path below, which is removed once chunk 2 grows the map.
    registry_key = _REGISTRY_KEY_MAP.get(file_key)
    if registry_key is not None and file_type == 'xarray':
        from stardust_2d_inputs.core.loader import load as _engine_load
        return _engine_load(registry_key)

    # Legacy path (transitional): download (or retrieve from the pooch cache)
    # from the climate_database_files GitHub repo.
    extension = '.nc' if file_type == 'xarray' else '.npz'
    file_path = file_key + extension
    local_file, _ = load_repo_table(
        config['climate_database_files_http'],
        file_path,
        config['climate_database_token'],
        proj_name=config['proj_name'],
    )
    if file_type == 'xarray':
        return xr.open_dataset(local_file)
    else:
        return np.load(local_file)


# ---------------------------------------------------------------------------
# ERA5 grid helpers
# ---------------------------------------------------------------------------

def era5_grid(config):
    """Return the ERA5 pressure-level and latitude grids.

    Loads the ``Monthly_Zonal_Cloud_Cover_2008_2017`` dataset and extracts
    the coordinate arrays.

    Parameters
    ----------
    config : dict
        Project configuration (from ``load_project_config()``).

    Returns
    -------
    lev : ndarray   ERA5 pressure levels (hPa), ascending.
    lat : ndarray   ERA5 latitudes (degrees), ascending (south → north).
    """
    fields = load_xr_from_repo(
        'Monthly_Zonal_Cloud_Cover_2008_2017', config,
    )
    lat = fields.latitude[::-1].to_numpy()
    lev = fields.level.to_numpy().astype('float32')
    return lev, lat
