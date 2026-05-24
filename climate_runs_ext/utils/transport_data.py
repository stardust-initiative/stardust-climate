r"""Loader for the 2-D aerosol-transport driver fields.

`get_atm_data` reads the precomputed TEM-decomposed eddy diffusivities and
residual winds, the ERA5 zonal-mean tropopause, and the monthly zonal-mean
temperature from the climate database, floors the diffusivities, and reduces
them in time for use by the 2-D transport solver.

The preprocessing that *produces* these driver datasets lives in the
stardust_2d_inputs input-data engine (``generators.transport.diffusion_decomp``
and ``generators.transport.tropopause``).
"""
import numpy as np
import xarray as xr
from scipy import interpolate
from climlab import constants as const

from climate_runs_ext.utils.data_loading import load_xr_from_repo


def get_atm_data(config, time_type=1, snr_limit=1.0, data_year=-1):
    r"""Load and assemble the 2-D transport driver fields.

    Parameters
    ----------
    config : dict
        Project configuration (from ``load_project_config``), used to
        resolve files in the climate database.
    time_type : int
        ``0`` -> annual mean; ``1`` -> monthly climatology; ``2`` -> full
        time series.
    snr_limit : float
        Kzz is replaced by its floor where its signal-to-noise ratio
        exceeds this value.
    data_year : int
        With ``time_type < 2``: ``-1`` averages over all years, otherwise
        the given calendar year is selected.

    Returns
    -------
    vlev, vlat, kzz, kyy, kyz, tropopause
        Residual vertical and meridional winds, the three eddy-diffusivity
        components (Kzz, Kyy, Kyz), and the tropopause pressure.
    """
    ds = load_xr_from_repo('diffusion_ml_TEMdecomp_raw_2008_2017', config)
    tropopause = load_xr_from_repo(
        'tropopause_ERA5_zonal_mean_2008_2017', config,
    )['tropopause_pressure']
    ds = ds.sortby('latitude')

    kyynew = ds['K_phi_phi']
    kzznew = ds['K_pp']
    kyznew = ds['K_phi_p'] * 0.0

    # Floor Kzz / Kyy, using the air density derived from the monthly-mean T.
    fields = load_xr_from_repo('Monthly_Zonal_Variables_2008_2017', config)
    fields = fields.sortby('latitude')
    temperature = fields.T.mean(dim='month')
    interp_temperature = interpolate.RectBivariateSpline(
        fields.latitude, fields.level, temperature.T, kx=1, ky=1,
    )
    temperature_for_kzz = interp_temperature(
        kzznew['latitude'], kzznew['level'],
    ).T
    rho_for_kzz = (kzznew['level'].data[:, None] * 1e2
                   / temperature_for_kzz / const.Rd)
    kzzmin = 0.01
    kyymin = 1e4
    kzzminmat = kzzmin * (rho_for_kzz * const.g) ** 2.0
    kzzminmat_da = xr.DataArray(
        kzzminmat, dims=['level', 'latitude'],
        coords={'level': kzznew['level'], 'latitude': kzznew['latitude']},
    )
    n_years = kzznew.shape[0] // ds['K_pp_snr'].shape[0]
    condition = np.tile(ds['K_pp_snr'].values <= snr_limit, (n_years, 1, 1))
    kzznew = kzznew.where(condition, kzzminmat_da)
    kzznew = kzznew.where(kzznew >= kzzminmat, kzzminmat)
    kyynew = kyynew.where(kyynew >= kyymin, kyymin)

    if time_type < 2:
        if data_year == -1:
            kzznew = kzznew.groupby('time.month').mean(dim='time')
            kyynew = kyynew.groupby('time.month').mean(dim='time')
            kyznew = kyznew.groupby('time.month').mean(dim='time')
            ds = ds.groupby('time.month').mean(dim='time')
            tropopause = tropopause.mean(dim='year')
        else:
            kzznew = kzznew.sel(time=ds.time.dt.year == data_year)
            kyynew = kyynew.sel(time=ds.time.dt.year == data_year)
            kyznew = kyznew.sel(time=ds.time.dt.year == data_year)
            ds = ds.sel(time=ds.time.dt.year == data_year)
            tropopause = tropopause.sel(year=data_year)
    if time_type < 1:
        ds = ds.mean(dim='month')
        kzznew = kzznew.mean(dim='month')
        kyynew = kyynew.mean(dim='month')
        kyznew = kyznew.mean(dim='month')
        tropopause = tropopause.mean(dim='month')

    vlat = ((ds['v_mean'] + ds['v_star_TEM'])
            * np.cos(np.deg2rad(ds['latitude']))).fillna(0)
    vlev = (ds['w_mean'] + ds['w_star_TEM']).fillna(0)
    ds.close()

    kyynew = kyynew.where(kyynew >= 0.0, 0.0)
    kzznew = kzznew.where(kzznew >= 0.0, 0.0)

    return vlev, vlat, kzznew, kyynew, kyznew, tropopause
