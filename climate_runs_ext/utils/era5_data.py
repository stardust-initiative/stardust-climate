"""ERA5 reanalysis data loading and interpolation functions.

Ported from ``climate_runs/utils/utils_methods.py``.

Every function that loads data from the repository takes an explicit
``config`` parameter (the dict returned by ``load_project_config()``).
Internally they all call ``load_xr_from_repo`` from ``data_loading.py``.

Usage
-----
::

    from climate_runs_ext import load_project_config
    from climate_runs_ext.utils.era5_data import (
        Seasonal_insolation, Surface_albedo, Ozone,
        Relative_Humidity_Profile, Cloud_Cover, get_era5_mycloud,
        oceanic_heat_uptake, get_surface_flux_drag_coe,
    )

    cfg = load_project_config()
    albedo = Surface_albedo(lat, months, cfg)
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import numpy as np
from scipy import interpolate
from scipy.signal import savgol_filter as sf
from climlab import constants as const
from climlab.solar.insolation import daily_insolation

from climate_runs_ext.utils.data_loading import load_xr_from_repo


# ---------------------------------------------------------------------------
# Season type mapping
# ---------------------------------------------------------------------------

class SeasonTypes:
    """Season name <-> month-index mapping."""
    DJF = 'DJF'
    MAM = 'MAM'
    JJA = 'JJA'
    SON = 'SON'
    Annual = 'Annual'
    Jan = 'Jan'
    Feb = 'Feb'
    Mar = 'Mar'
    Apr = 'Apr'
    May = 'May'
    Jun = 'Jun'
    Jul = 'Jul'
    Aug = 'Aug'
    Sep = 'Sep'
    Oct = 'Oct'
    Nov = 'Nov'
    Dec = 'Dec'
    months_dict = {
        'DJF': [11, 0, 1], 'MAM': [2, 3, 4],
        'JJA': [5, 6, 7], 'SON': [8, 9, 10],
        'Annual': list(range(0, 12)),
        'Jan': [0], 'Feb': [1], 'Mar': [2], 'Apr': [3],
        'May': [4], 'Jun': [5], 'Jul': [6], 'Aug': [7],
        'Sep': [8], 'Oct': [9], 'Nov': [10], 'Dec': [11],
    }
    days_in_month_dict = {
        'Jan': 31, 'Feb': 28, 'Mar': 31, 'Apr': 30, 'May': 31, 'Jun': 30,
        'Jul': 31, 'Aug': 31, 'Sep': 30, 'Oct': 31, 'Nov': 30, 'Dec': 31,
    }
    month_str_dict = {
        0: 'Jan', 1: 'Feb', 2: 'Mar', 3: 'Apr', 4: 'May', 5: 'Jun',
        6: 'Jul', 7: 'Aug', 8: 'Sep', 9: 'Oct', 10: 'Nov', 11: 'Dec',
    }


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def lat_avg(field, lat):
    """Cosine-weighted latitude average along axis 0."""
    return np.average(field, weights=np.cos(np.deg2rad(lat)), axis=0)


def Smooth(field, window_length=10, polyorder=1, axis=0):
    """Savitzky-Golay smoothing wrapper."""
    return sf(np.squeeze(field), window_length=window_length,
              polyorder=polyorder, axis=axis)


# ---------------------------------------------------------------------------
# Insolation / albedo / ocean heat
# ---------------------------------------------------------------------------

def Seasonal_insolation(lat, seas):
    """Annual or seasonal mean insolation via climlab daily_insolation."""
    if seas == SeasonTypes.Annual:
        days = np.linspace(0, const.days_per_year, 365)
    else:
        days = []
        for i_month in SeasonTypes.months_dict[seas]:
            days += list(np.mod(
                np.arange(i_month * const.days_per_month,
                           (i_month + 1) * const.days_per_month),
                const.days_per_year,
            ))
        days = np.array(days)
    insolation = np.mean(daily_insolation(lat, days), axis=1)
    return insolation


def Surface_albedo(lat, months, config):
    """ERA5 surface albedo interpolated to *lat*.

    Loads monthly zonal TOA and surface balance (2008) and computes
    albedo = sw_up / sw_down.
    """
    fields = load_xr_from_repo('Monthly_Zonal_Toa_and_Surf_balance_2008', config)
    fill_nan = 1e-10
    sw_down = fields.msdwswrf[months, :]
    sw_net = fields.msnswrf[months, :]
    sw_down_avg = sw_down.mean(dim='month')
    sw_up_avg = (sw_down - sw_net).mean(dim='month')
    albedo = sw_up_avg / sw_down_avg
    val_albedo = np.where(np.isnan(albedo), fill_nan, albedo)
    lat_albedo = np.array(albedo.latitude.values)
    ind = np.argsort(lat_albedo)
    res = interpolate.interp1d(lat_albedo[ind], val_albedo[ind], kind='linear')
    return res(lat)


def oceanic_heat_uptake(lat, months, config):
    """ERA5 ocean heat absorption interpolated to *lat* (W/m^2)."""
    fields = load_xr_from_repo('Monthly_Zonal_balance_2008_2017', config)
    oha = fields.oha[months, :].mean(dim='month') / 3600.0
    res = interpolate.interp1d(oha.latitude, oha.values, kind='linear')
    return res(lat)


# ---------------------------------------------------------------------------
# Clouds
# ---------------------------------------------------------------------------

def Cloud_Cover(new_lev, new_lat, months, config):
    """ERA5 cloud fraction interpolated to model grid."""
    fields = load_xr_from_repo('Monthly_Zonal_Cloud_Cover_2008_2017', config)
    cc = fields.cloud_cover[months, :, ::-1].mean(dim='time')
    lat_mat, lev_mat = np.meshgrid(new_lat, new_lev, indexing='ij')
    coords = np.column_stack([lat_mat.ravel(), lev_mat.ravel()])
    interp = interpolate.RegularGridInterpolator(
        (cc.latitude, cc.level), cc.values.T,
        bounds_error=False, fill_value=0.0, method='linear',
    )
    res = interp(coords).reshape(lat_mat.shape)
    return np.where(res > 0.0, res, 0.0)


def get_cloud_water_path(domain, months, config, lat0=0):
    """ERA5 CLWP / CIWP interpolated to model grid (g/m^2)."""
    lev = domain.lev.points
    lat = domain.lat.points if hasattr(domain, 'lat') else np.array([lat0])
    dp_g = 1e5 * domain.lev.delta / const.g
    fields = load_xr_from_repo('Monthly_Zonal_Cloud_content_2008_2017', config)
    ciwp_era5 = fields.ciwc[months, :, :].mean(dim='month')
    clwp_era5 = fields.clwc[months, :, :].mean(dim='month')
    lat_mat, lev_mat = np.meshgrid(lat, lev, indexing='ij')
    coords = np.column_stack([lat_mat.ravel(), lev_mat.ravel()])
    interp_ci = interpolate.RegularGridInterpolator(
        (ciwp_era5.latitude, ciwp_era5.level), ciwp_era5.values.T,
        bounds_error=False, fill_value=0.0, method='linear',
    )
    interp_cl = interpolate.RegularGridInterpolator(
        (clwp_era5.latitude, clwp_era5.level), clwp_era5.values.T,
        bounds_error=False, fill_value=0.0, method='linear',
    )
    ciwp = interp_ci(coords).reshape(lat_mat.shape)
    clwp = interp_cl(coords).reshape(lat_mat.shape)
    ciwp = np.where(ciwp > 0.0, ciwp, 0.0)
    clwp = np.where(clwp > 0.0, clwp, 0.0)
    clwp *= dp_g[np.newaxis, :]
    ciwp *= dp_g[np.newaxis, :]
    return clwp, ciwp


def get_era5_mycloud(domain, months, config, **kwargs):
    """Full cloud dict from ERA5 data.

    Returns dict with keys: cldfrac, ciwp, clwp, r_ice, r_liq.
    """
    lev = domain.lev.points
    lat = domain.lat.points
    cldfrac = Cloud_Cover(lev, lat, months, config)
    clwp, ciwp = get_cloud_water_path(domain, months, config)

    nlev, nlat = len(lev), len(lat)
    r_liq = kwargs.get('r_liq', 14.0)
    r_ice = kwargs.get('r_ice', 14.0)
    if isinstance(r_liq, float):
        r_liq = r_liq * np.ones((nlat, nlev))
    if isinstance(r_ice, float):
        r_ice = r_ice * np.ones((nlat, nlev))

    return {
        'cldfrac': cldfrac, 'ciwp': ciwp, 'clwp': clwp,
        'r_ice': r_ice, 'r_liq': r_liq,
    }


# ---------------------------------------------------------------------------
# Ozone
# ---------------------------------------------------------------------------

def Ozone(new_lev, new_lat, months, config):
    """ERA5 ozone VMR interpolated to model grid."""
    fields = load_xr_from_repo('Monthly_Zonal_o3_2008_2017', config).o3
    oz = fields[months, :, :].mean(dim='time')[:, ::-1] * 28.97 / 48
    lat_mat, lev_mat = np.meshgrid(new_lat, new_lev, indexing='ij')
    coords = np.column_stack([lat_mat.ravel(), lev_mat.ravel()])
    interp = interpolate.RegularGridInterpolator(
        (oz.latitude, oz.level), oz.values.T,
        bounds_error=False, fill_value=None, method='linear',
    )
    res = interp(coords).reshape(lat_mat.shape)
    return np.where(res > 0.0, res, 0.0)


# ---------------------------------------------------------------------------
# Humidity
# ---------------------------------------------------------------------------

def era5_annual_initial_state(domain, months, config, **kwargs):
    """ERA5 zonal/time-averaged T, q, Ts on the model grid.

    Loads the monthly-zonal ERA5 climatology (2008-2017) and produces
    initial-condition arrays matching ``domain``.  Used as ``Tinit_dict``
    for ``model_generator`` / ``get_rce_sbm_model_annual_avg`` so the model
    starts from ERA5 annual-mean temperature and humidity rather than
    climlab's default idealized profile.

    Parameters
    ----------
    domain : climlab Domain
        Target atmospheric domain (``state.Tatm.domain``) — provides
        ``lev.points`` (hPa) and ``lat.points`` (degrees).
    months : list of int
        Month indices (0-11) to average over.  Annual mean uses
        ``SeasonTypes.months_dict['Annual']``.
    config : dict
        Project configuration from ``load_project_config()``.
    **kwargs
        ``ts_variable`` : name of surface temperature variable in the
        surface-params file (default ``'ts'``).

    Returns
    -------
    dict
        ``{'Tatm': (nlat, nlev), 'q': (nlat, nlev), 'Ts': (nlat,)}``.
        ``Tatm`` is atmospheric temperature in K; ``q`` is specific
        humidity in kg/kg; ``Ts`` is surface skin temperature in K.

    Notes
    -----
    * ERA5 native pressure levels span 1-1000 hPa.  Model levels above
      1 hPa are filled by extrapolation (``fill_value=None`` in
      ``RegularGridInterpolator``) — effectively isothermal extrapolation
      of the topmost ERA5 level.
    * Negative humidities (from cubic extrapolation into the upper
      stratosphere) are clipped to a small positive floor.
    """
    new_lev = domain.lev.points
    new_lat = domain.lat.points

    # --- T, q on pressure levels ---
    fields = load_xr_from_repo('Monthly_Zonal_Variables_2008_2017', config)
    T_era5 = fields.T[months, :, :].mean(dim='month')  # (level, latitude)
    q_era5 = fields.q[months, :, :].mean(dim='month')

    lat_era5 = T_era5.latitude.values
    lev_era5 = T_era5.level.values
    # Ensure ascending latitude axis for the interpolator
    if np.diff(lat_era5).mean() < 0:
        lat_era5 = lat_era5[::-1]
        T_vals = T_era5.values[:, ::-1]   # (level, latitude)
        q_vals = q_era5.values[:, ::-1]
    else:
        T_vals = T_era5.values
        q_vals = q_era5.values

    lat_mat, lev_mat = np.meshgrid(new_lat, new_lev, indexing='ij')
    coords = np.column_stack([lat_mat.ravel(), lev_mat.ravel()])

    interp_T = interpolate.RegularGridInterpolator(
        (lat_era5, lev_era5), T_vals.T,     # (latitude, level)
        bounds_error=False, fill_value=None, method='linear',
    )
    interp_q = interpolate.RegularGridInterpolator(
        (lat_era5, lev_era5), q_vals.T,
        bounds_error=False, fill_value=None, method='linear',
    )
    Tatm = interp_T(coords).reshape(lat_mat.shape)
    q = interp_q(coords).reshape(lat_mat.shape)
    q = np.where(q > 0.0, q, 5e-7)

    # --- Ts from surface params ---
    ts_variable = kwargs.get('ts_variable', 'ts')
    srf = surface_params_era5(config, months=months)
    ts_da = srf[ts_variable]
    lat_srf = np.array(ts_da.latitude.values)
    ts_vals = np.array(ts_da.values)
    if np.diff(lat_srf).mean() < 0:
        lat_srf = lat_srf[::-1]
        ts_vals = ts_vals[::-1]
    Ts = np.interp(new_lat, lat_srf, ts_vals)
    # Match climlab Ts layout (nlat, 1) to avoid broadcasting to (nlat, nlat)
    # when merged into an existing climlab state.
    Ts = Ts.reshape(-1, 1)

    return {'Tatm': Tatm, 'q': q, 'Ts': Ts}


def Relative_Humidity_Profile(new_lev, new_lat, months, config, **kwargs):
    """ERA5 relative humidity profile interpolated to model grid (fraction)."""
    fields = load_xr_from_repo('Monthly_Zonal_RH_2008_2017', config)
    if 'years' in kwargs:
        fields = fields.sel(year=kwargs['years'])
    fields = fields.mean('year')
    rh = fields.transpose('month', 'level', 'latitude').RH[months, :, :].mean(dim='month')
    rh = 0.01 * rh  # % -> fraction
    if np.diff(rh.latitude.values).mean() < 0:
        rh = rh[:, ::-1]
    lat_mat, lev_mat = np.meshgrid(new_lat, new_lev, indexing='ij')
    coords = np.column_stack([lat_mat.ravel(), lev_mat.ravel()])
    interp = interpolate.RegularGridInterpolator(
        (rh.latitude, rh.level), rh.values.T,
        bounds_error=False, fill_value=0.0, method='linear',
    )
    return interp(coords).reshape(lat_mat.shape)


# ---------------------------------------------------------------------------
# Surface parameters and fluxes
# ---------------------------------------------------------------------------

def surface_pressure_era5(config, **kwargs):
    """ERA5 zonal-mean surface pressure."""
    fields = load_xr_from_repo('Monthly_Zonal_Srf_Pres_2008_2017', config)
    if 'years' in kwargs:
        fields = fields.sel(valid_time=fields.valid_time.dt.year.isin(kwargs['years']))
    if 'months' in kwargs:
        fields = fields.sel(
            valid_time=fields.valid_time.dt.month.isin([m + 1 for m in kwargs['months']]),
        ).mean('valid_time')
    else:
        fields = fields.mean('valid_time')
    if 'lat' in kwargs:
        val_interp = interpolate.interp1d(fields.latitude, fields.sp)(kwargs['lat'])
        return val_interp
    else:
        return fields.sp


def surface_params_era5(config, **kwargs):
    """ERA5 surface temperature, dew point, etc."""
    fields = load_xr_from_repo('Monthly_Zonal_Srf_Params_2008_2017', config)
    if 'years' in kwargs:
        fields = fields.sel(valid_time=fields.valid_time.dt.year.isin(kwargs['years']))
    if 'months' in kwargs:
        fields = fields.sel(
            valid_time=fields.valid_time.dt.month.isin([m + 1 for m in kwargs['months']]),
        ).mean('valid_time')
    else:
        fields = fields.mean('valid_time')
    if 'lat' in kwargs:
        params_dict = {
            da: interpolate.interp1d(fields.latitude, fields[da])(kwargs['lat'])
            for da in fields.data_vars
        }
        return params_dict
    else:
        return fields


def get_surface_fluxes(config, **kwargs):
    """ERA5 latent and sensible heat fluxes (W/m^2)."""
    fields = load_xr_from_repo('Monthly_Zonal_SURFACE_FLUXES_2008_2017', config)
    if 'years' in kwargs:
        fields = fields.sel(valid_time=fields.valid_time.dt.year.isin(kwargs['years']))
    if 'months' in kwargs:
        fields = fields.sel(
            valid_time=fields.valid_time.dt.month.isin([m + 1 for m in kwargs['months']]),
        ).mean('valid_time')
    else:
        fields = fields.mean('valid_time')
    if 'lat' in kwargs:
        lhf = interpolate.interp1d(fields.latitude, fields.slhf, kind='linear')(kwargs['lat'])
        shf = interpolate.interp1d(fields.latitude, fields.sshf, kind='linear')(kwargs['lat'])
    else:
        lhf = fields.slhf
        shf = fields.sshf
    return lhf, shf


def get_surface_flux_drag_coe(config, **kwargs):
    """Drag coefficients (Cd) for LHF and SHF fitted from ERA5 data.

    Returns (Cd_lhf, Cd_shf, lat) with shape (..., 1) for broadcasting.
    """
    def get_coe(y, x, n=0, xmin=0.0):
        y1 = y * x
        x1 = x ** 2 + xmin ** 2
        nx = len(x)
        if n > 0:
            coe = np.zeros_like(x)
            for i in range(nx):
                ind = np.arange(max([i - n, 0]), min([i + n + 1, nx - 1]), dtype=int)
                coe[i] = float(np.mean(y1[ind] * x1[ind]) / np.mean(x1[ind] ** 2))
            return coe
        else:
            return y1 / x1

    lat_win_lhf = kwargs.get('lat_win_lhf', (-70.0, 90.0))
    lat_win_shf = kwargs.get('lat_win_shf', (-63.0, 52.0))
    assert len(lat_win_lhf) == 2, 'lat_win_lhf should be of length 2'
    assert len(lat_win_shf) == 2, 'lat_win_shf should be of length 2'

    override_q = kwargs.get('override_q', False)
    lhf, shf = get_surface_fluxes(config, **kwargs)
    surface_param_dict = surface_params_era5(config, **kwargs)
    ps = kwargs.get('ps', 1000.0)

    if override_q:
        from climlab.utils import thermo
        do_simplified_dict = {'do_simplified': kwargs['do_simplified']} if 'do_simplified' in kwargs else {}
        small_dict = {'small': kwargs['small']} if 'small' in kwargs else {}
        do_era5_dict = {'do_era5': kwargs['do_era5']} if 'do_era5' in kwargs else {}
        params = {**do_simplified_dict, **small_dict, **do_era5_dict}
        qs = thermo.qsat(surface_param_dict['ts'], ps, **params)
        q2m = thermo.qsat(surface_param_dict['d2m'], ps, **params)
    else:
        qs = surface_param_dict['qs']
        q2m = surface_param_dict['q2m']
    dq = qs - q2m
    dT = surface_param_dict['ts'] - surface_param_dict['t2m']
    U = kwargs.get('U', 5.0)
    do_smooth = kwargs.get('do_smooth', True)
    rho = kwargs.get('rho', ps * const.mb_to_Pa / const.Rd / surface_param_dict['ts'].values)

    if 'new_lat' in kwargs:
        lat = kwargs['new_lat']
    else:
        lat = lhf.latitude.values[::-1]

    window_length = int(np.abs(np.mean(np.diff(lat))) / np.abs(np.mean(np.diff(dq.latitude.values))))
    window_length = max(window_length, 3)  # ensure minimum valid window
    if window_length % 2 == 0:
        window_length += 1  # must be odd for savgol

    n = kwargs.get('n', 0)
    dT_min = kwargs.get('dT_min', 1e-2)
    dq_min = kwargs.get('dq_min', 1e-4)

    if do_smooth:
        Cd_shf = get_coe(Smooth(shf.values, window_length=window_length),
                         Smooth(dT.values, window_length=window_length),
                         n=n, xmin=dT_min) / rho / U / const.cp
        Cd_lhf = get_coe(Smooth(lhf.values, window_length=window_length),
                         Smooth(dq.values, window_length=window_length),
                         n=n, xmin=dq_min) / rho / U / const.Lhvap
    else:
        Cd_shf = get_coe(shf.values, dT.values, n=n, xmin=dT_min) / rho / U / const.cp
        Cd_lhf = get_coe(lhf.values, dq.values, n=n, xmin=dq_min) / rho / U / const.Lhvap

    Cd_shf_min = kwargs.get('Cd_shf_min', 1e-4)
    Cd_lhf_min = kwargs.get('Cd_lhf_min', 1e-5)
    Cd_shf_max = kwargs.get('Cd_shf_max', 1e-2)
    Cd_lhf_max = kwargs.get('Cd_lhf_max', 1e-2)

    Cd_shf = Cd_shf[::-1]
    Cd_lhf = Cd_lhf[::-1]
    if Cd_shf_min is not None:
        Cd_shf = np.where(Cd_shf > Cd_shf_min, Cd_shf, Cd_shf_min)
    if Cd_lhf_min is not None:
        Cd_lhf = np.where(Cd_lhf > Cd_lhf_min, Cd_lhf, Cd_lhf_min)
    if Cd_shf_max is not None:
        Cd_shf = np.where(Cd_shf < Cd_shf_max, Cd_shf, Cd_shf_max)
    if Cd_lhf_max is not None:
        Cd_lhf = np.where(Cd_lhf < Cd_lhf_max, Cd_lhf, Cd_lhf_max)

    lat_era5 = lhf.latitude.values[::-1]
    if 'new_lat' in kwargs:
        Cd_lhf = np.interp(lat, lat_era5, Cd_lhf)
        Cd_shf = np.interp(lat, lat_era5, Cd_shf)
    Cd_lhf = np.where(lat > lat_win_lhf[0], Cd_lhf, np.interp(lat_win_lhf[0], lat, Cd_lhf))
    Cd_lhf = np.where(lat < lat_win_lhf[1], Cd_lhf, np.interp(lat_win_lhf[1], lat, Cd_lhf))
    Cd_shf = np.where(lat > lat_win_shf[0], Cd_shf, np.interp(lat_win_shf[0], lat, Cd_shf))
    Cd_shf = np.where(lat < lat_win_shf[1], Cd_shf, np.interp(lat_win_shf[1], lat, Cd_shf))

    return Cd_lhf[:, np.newaxis], Cd_shf[:, np.newaxis], lat


# ---------------------------------------------------------------------------
# Moisture diffusivity
# ---------------------------------------------------------------------------

def meridional_Kq(new_lat, config, Kmin=1e3, excluded_win_list=None):
    """ERA5-derived meridional moisture diffusivity."""
    if excluded_win_list is None:
        excluded_win_list = [(0.0, 8.0)]
    fields = load_xr_from_repo('Mean_meridional_moist_diffusion', config)
    lat = fields.Kq_avg.latitude.values
    Kq = fields.Kq_avg.values
    ind = np.where(Kq > Kmin)[0]
    for excluded_win in excluded_win_list:
        ind1 = np.where((lat[ind] > excluded_win[1]) | (lat[ind] < excluded_win[0]))[0]
        ind = ind[ind1]
    lat = lat[ind]
    Kq = Kq[ind]
    ind = np.argsort(lat)
    from scipy.interpolate import CubicSpline
    return CubicSpline(lat[ind], Kq[ind])(new_lat)
