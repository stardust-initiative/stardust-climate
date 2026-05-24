"""Transport parameter computation (2D advection-diffusion).

Ported from ``climate_runs/utils/utils_methods.py``:

* ``get_mse_input``              -- ERA5 MSE transport diagnostics
* ``regularize_wz``              -- vertical velocity regularisation
* ``correct_vw_field_poisson_solver`` -- Poisson solver for mass-conservation
* ``correct_vw_field``           -- higher-level velocity-field correction
* ``edge_preserving_smoothing``  -- gradient-aware smoothing (scikit-image)
* ``merge_maps``                 -- blend two fields using a mask
* ``get_transport_param``        -- main entry point

Usage
-----
::

    from climate_runs_ext.utils.transport_params import get_transport_param
    tp = get_transport_param(domain, config, months=months)
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import numpy as np
import xarray as xr
from scipy import interpolate
from scipy.linalg import solve_banded
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

from climlab import constants as const

from climate_runs_ext.utils.data_loading import load_xr_from_repo
from climate_runs_ext.utils.era5_data import (
    Smooth, surface_pressure_era5,
)


# ---------------------------------------------------------------------------
# MSE input loading
# ---------------------------------------------------------------------------

def get_mse_input(months, config, **kwargs):
    """Load and compute ERA5 MSE transport diagnostics.

    Returns an xarray Dataset with fields: vbar, wbar, qbar, Tbar, GPbar,
    hbar, K_mer_se, K_ver_se, K_mer_q, K_ver_q, Sbar_h, Sbar_q, wqtag.
    """
    small = kwargs.get('small', 1e-2)
    lev_win_regularization = kwargs.get('lev_win_regularization', (400, 900))
    lat_win_regularization = kwargs.get('lat_win_regularization', (-60, 60))
    fields = load_xr_from_repo('./mse_files/era5_q_h_2008_2017', config)
    if 'year' in kwargs:
        fields = fields.sel(year=kwargs['year'], drop=True)
    else:
        fields = fields.mean('year')
    month_list = [m + 1 for m in months]
    fields = fields.sel(month=month_list).mean('month')

    lat_rad = np.deg2rad(fields.latitude)
    coslat = np.cos(lat_rad)
    deg2rad = np.pi / 180.0

    GPbar = (fields.hbar - const.cp * (1 - fields.qbar) * fields.Tbar
             - const.Lhvap * fields.qbar)

    vqtag = fields.vqbar - fields.vbar * fields.qbar
    wqtag = fields.wqbar - fields.wbar * fields.qbar
    vttag = fields.vTbar - fields.vbar * fields.Tbar
    wttag = fields.wTbar - fields.wbar * fields.Tbar

    se_bar = fields.hbar - GPbar
    vsetag = const.cp * vttag + const.Lhvap * vqtag
    wsetag = const.cp * wttag + const.Lhvap * wqtag

    grad_mer_q = fields.qbar.differentiate('latitude', edge_order=2) / deg2rad / const.a
    grad_ver_q = fields.qbar.differentiate('level', edge_order=2) / 100
    grad_mer_se = se_bar.differentiate('latitude', edge_order=2) / deg2rad / const.a
    grad_ver_se = se_bar.differentiate('level', edge_order=2) / 100

    mask = np.where(
        (np.abs(fields.level - 0.5 * (lev_win_regularization[0] + lev_win_regularization[1]))
         <= 0.5 * (lev_win_regularization[1] - lev_win_regularization[0]))
        & (np.abs(fields.latitude - 0.5 * (lat_win_regularization[0] + lat_win_regularization[1]))
           <= 0.5 * (lat_win_regularization[1] - lat_win_regularization[0])),
        1.0, 0.0,
    )
    K_mer_q = -vqtag * grad_mer_q / (grad_mer_q ** 2 + small * (mask * grad_mer_q ** 2).max())
    K_ver_q = -wqtag * grad_ver_q / (grad_ver_q ** 2 + small * (mask * grad_ver_q ** 2).max())
    K_mer_se = -vsetag * grad_mer_se / (grad_mer_se ** 2 + small * (mask * grad_mer_se ** 2).max())
    K_ver_se = -wsetag * grad_ver_se / (grad_ver_se ** 2 + small * (mask * grad_ver_se ** 2).max())

    flux_mer_q, flux_ver_q = fields.vqbar * coslat, fields.wqbar
    flux_mer_h, flux_ver_h = fields.vhbar * coslat, fields.whbar

    div_mer_q = flux_mer_q.differentiate('latitude', edge_order=2) * 360.0 / (2 * np.pi * const.a * coslat)
    div_ver_q = flux_ver_q.differentiate('level', edge_order=2) / 100.0
    div_mer_h = flux_mer_h.differentiate('latitude', edge_order=2) * 360.0 / (2 * np.pi * const.a * coslat)
    div_ver_h = flux_ver_h.differentiate('level', edge_order=2) / 100.0

    Sbar_q = fields.dq_dt + div_mer_q + div_ver_q
    Sbar_h = fields.dh_dt + div_mer_h + div_ver_h

    out = xr.Dataset(
        {
            'vbar': fields.vbar[:, ::-1],
            'wbar': fields.wbar[:, ::-1],
            'qbar': fields.qbar[:, ::-1],
            'Tbar': fields.Tbar[:, ::-1],
            'GPbar': GPbar[:, ::-1],
            'hbar': fields.hbar[:, ::-1],
            'K_mer_se': K_mer_se[:, ::-1],
            'K_ver_se': K_ver_se[:, ::-1],
            'K_mer_q': K_mer_q[:, ::-1],
            'K_ver_q': K_ver_q[:, ::-1],
            'Sbar_h': Sbar_h[:, ::-1],
            'Sbar_q': Sbar_q[:, ::-1],
            'wqtag': wqtag[:, ::-1],
        },
        coords={'level': fields.level, 'latitude': fields.latitude[::-1]},
    )
    return out


# ---------------------------------------------------------------------------
# Regularisation and velocity correction
# ---------------------------------------------------------------------------

def regularize_wz(wz_raw, z, **kwargs):
    """Vertical velocity regularisation via tridiagonal solve."""
    nz = len(z) - 1
    ny = wz_raw.shape[0]
    assert wz_raw.shape[1] == nz + 1, 'wz_raw.shape[1] should be len(z)+1'
    eta_coe = kwargs.get('eta_coe', 0.0) + 0 * np.ones((ny, nz + 1))
    gamma_coe = kwargs.get('gamma_coe', 1.0) + 0 * np.ones((ny, nz))
    assert eta_coe.shape == (ny, nz + 1)
    assert gamma_coe.shape == (ny, nz)

    wz = np.zeros_like(wz_raw)
    dz = np.diff(z)
    ind = np.arange(1, nz)
    for i in range(ny):
        mat = np.zeros((3, nz - 1))
        d = (dz[ind] + dz[ind - 1]) * wz_raw[i, ind]
        mat[1, :] = (dz[ind] + dz[ind - 1]) * (1 + eta_coe[i, ind])
        c = 2 * gamma_coe[i, ind] / dz[ind]
        a = 2 * gamma_coe[i, ind - 1] / dz[ind - 1]
        mat[1, :] += (c + a)
        mat[0, 1:] = -c[:-1]
        mat[2, :-1] = -a[1:]
        wz[i, 1:-1] = solve_banded((1, 1), mat, d)
    return wz


def correct_vw_field_poisson_solver(v_raw, w_raw, y, z,
                                     w_weight=1.0, do_smooth=True,
                                     window_length=10, do_plot=False):
    """Poisson solver for divergence-free velocity field correction.

    Finds phi such that v_new = dphi/dz, w_new = -dphi/dy
    and the field (v_new, w_new) is divergence-free.
    """
    n_y = len(y) - 1
    n_z = len(z) - 1
    v = Smooth(v_raw, window_length=window_length) if do_smooth else v_raw.copy()
    w = Smooth(w_raw, window_length=window_length) if do_smooth else w_raw.copy()
    assert v.shape == (n_y + 1, n_z)
    assert w.shape == (n_y, n_z + 1)
    w_weight_mat = w_weight + np.zeros((n_y, n_z))

    # Apply boundary conditions
    v[0, :] = 0.0
    v[-1, :] = 0.0
    w[:, 0] = 0.0
    w[:, -1] = 0.0
    v_mid = 0.5 * (v[1:, :] + v[:-1, :])
    w_mid = 0.5 * (w[:, 1:] + w[:, :-1])
    v_ext_z = np.concatenate(
        (v_mid[:, 0, np.newaxis],
         0.5 * (v_mid[:, 1:] + v_mid[:, :-1]),
         v_mid[:, -1, np.newaxis]),
        axis=1,
    )
    w_ext_y = np.concatenate(
        (w_mid[0, np.newaxis, :],
         0.5 * (w_mid[1:, :] + w_mid[:-1, :]),
         w_mid[-1, np.newaxis, :]),
        axis=0,
    )
    Q = (np.diff(v_ext_z, axis=1) / np.diff(z)[np.newaxis, :]
         - w_weight_mat * np.diff(w_ext_y, axis=0) / np.diff(y)[:, np.newaxis])

    # Build sparse matrix
    ind_func = lambda i, j: i * (n_z + 1) + j
    row_ind = []
    col_ind = []
    data = []
    r = np.zeros(((n_y + 1) * (n_z + 1),))

    ind_r = 0
    for i in range(1, n_y):
        for j in range(1, n_z):
            a2 = float(2 * w_weight_mat[i, j] / (y[i + 1] - y[i - 1]) / (y[i + 1] - y[i]))
            a1 = float(2 * w_weight_mat[i, j] / (y[i + 1] - y[i - 1]) / (y[i] - y[i - 1]))
            b2 = float(2 / (z[j + 1] - z[j - 1]) / (z[j + 1] - z[j]))
            b1 = float(2 / (z[j + 1] - z[j - 1]) / (z[j] - z[j - 1]))

            row_ind.append(ind_r)
            col_ind.append(ind_func(i, j))
            data.append(-(a1 + a2 + b1 + b2))
            row_ind.append(ind_r)
            col_ind.append(ind_func(i + 1, j))
            data.append(a2)
            row_ind.append(ind_r)
            col_ind.append(ind_func(i - 1, j))
            data.append(a1)
            row_ind.append(ind_r)
            col_ind.append(ind_func(i, j + 1))
            data.append(b2)
            row_ind.append(ind_r)
            col_ind.append(ind_func(i, j - 1))
            data.append(b1)

            r[ind_r] = Q[i - 1, j - 1]
            ind_r += 1

    # Neumann BC: phi=0 on frame
    for j in range(n_z + 1):
        row_ind.append(ind_r)
        col_ind.append(ind_func(0, j))
        data.append(1.0)
        ind_r += 1
        row_ind.append(ind_r)
        col_ind.append(ind_func(n_y, j))
        data.append(1.0)
        ind_r += 1
    for i in range(1, n_y):
        row_ind.append(ind_r)
        col_ind.append(ind_func(i, 0))
        data.append(1.0)
        ind_r += 1
        row_ind.append(ind_r)
        col_ind.append(ind_func(i, n_z))
        data.append(1.0)
        ind_r += 1

    mat = csr_matrix(
        (data, (row_ind, col_ind)),
        shape=((n_y + 1) * (n_z + 1), (n_y + 1) * (n_z + 1)),
    )
    phi = spsolve(mat, r).reshape((n_y + 1, n_z + 1))

    vnew = np.diff(phi, axis=1) / np.diff(z)[np.newaxis, :]
    wnew = -np.diff(phi, axis=0) / np.diff(y)[:, np.newaxis]

    return vnew, wnew


def correct_vw_field(vy_raw, wp_raw, lat, lat_bound, lev, lev_bound,
                     config, **kwargs):
    """Higher-level velocity field correction with sigma-coordinate transform."""
    sp_bound = surface_pressure_era5(config, lat=lat_bound, **kwargs)
    do_sp_lat_max = kwargs.get('do_sp_lat_max', True)
    sp_lat_max = kwargs.get('sp_lat_max', -50.0)
    if do_sp_lat_max:
        sp_bound = np.where(lat_bound > sp_lat_max, sp_bound.max(), sp_bound)

    y_bound = const.a * np.sin(lat_bound * np.pi / 180)
    p_bound = 1e2 * lev_bound

    w_weight = kwargs.get('w_weight', 1.0e5)
    do_smooth = kwargs.get('do_smooth', False)
    do_plot = kwargs.get('do_plot', False)
    window_length = kwargs.get('window_length', 10)
    n_sigma = kwargs.get('n_sigma', 200)
    sp = np.interp(lat, lat_bound, sp_bound)

    wsig_raw = -wp_raw / sp[:, np.newaxis]
    sigma_vect = np.linspace(0, 1.0, n_sigma + 1)
    sigma_mid = 0.5 * (sigma_vect[1:] + sigma_vect[:-1])
    vy_raw_interp = np.zeros((len(lat) + 1, n_sigma))

    p_mat, _ = np.meshgrid(p_bound, lat)
    p_mat_bound, _ = np.meshgrid(p_bound, lat_bound)
    sigma_lev = np.where(p_mat <= sp[:, np.newaxis],
                         (1 - p_mat / sp[:, np.newaxis]), 0.0)
    sigma_lev_bound = np.where(p_mat_bound <= sp_bound[:, np.newaxis],
                               (1 - p_mat_bound / sp_bound[:, np.newaxis]), 0.0)

    do_pole_cutoff = kwargs.get('do_pole_cutoff', True)
    lat_maxima = kwargs.get('lat_maxima', [-60.0, 180.0])
    lat_fall_scale = kwargs.get('lat_fall_scale', 2.0)
    if do_pole_cutoff:
        ex1_bound = np.exp((lat_bound - lat_maxima[0]) / lat_fall_scale)
        ex1 = np.exp((lat + lat_maxima[0]) / lat_fall_scale)
        ex2_bound = np.exp((lat_maxima[1] - lat_bound) / lat_fall_scale)
        ex2 = np.exp((lat_maxima[1] - lat) / lat_fall_scale)
        fac_poly_cutoff_bound = ((ex1_bound / (1 + ex1_bound))
                                 * (ex2_bound / (1 + ex2_bound)))[:, np.newaxis]
        fac_poly_cutoff = ((ex1 / (1 + ex1))
                           * (ex2 / (1 + ex2)))[:, np.newaxis]
    else:
        fac_poly_cutoff_bound = np.ones((len(lat) + 1, 1))
        fac_poly_cutoff = np.ones((len(lat), 1))

    for j in range(len(lat) + 1):
        vy_raw_interp[j, :] = np.interp(
            sigma_mid,
            0.5 * (sigma_lev_bound[j, 1:] + sigma_lev_bound[j, :-1])[::-1],
            (fac_poly_cutoff_bound * vy_raw)[j, ::-1],
        )
    wsig_raw_interp = np.zeros((len(lat), n_sigma + 1))
    for j in range(len(lat)):
        wsig_raw_interp[j, :] = np.interp(
            sigma_vect, sigma_lev[j, :][::-1], wsig_raw[j, ::-1],
        )
    wsig_raw_interp = np.where(np.isnan(wsig_raw_interp), 0.0, wsig_raw_interp)
    vy_raw_interp = np.where(np.isnan(vy_raw_interp), 0.0, vy_raw_interp)

    gamma_coe_large_scale = kwargs.get('gamma_coe_large_scale', 1.0e1)
    gamma_coe_small_scale = kwargs.get('gamma_coe_small_scale', 0.0)
    gamma_coe = gamma_coe_large_scale - (gamma_coe_large_scale - gamma_coe_small_scale) * fac_poly_cutoff
    wsig_interp_regularized = regularize_wz(wsig_raw_interp, sigma_vect, gamma_coe=gamma_coe, eta_coe=0.0)

    do_skip_correction = kwargs.get('do_skip_correction', False)
    if do_skip_correction:
        vy_interp, wsig_interp = vy_raw_interp, wsig_interp_regularized
    else:
        vy_interp, wsig_interp = correct_vw_field_poisson_solver(
            vy_raw_interp, wsig_interp_regularized, y_bound, sigma_vect,
            do_smooth=do_smooth, w_weight=w_weight, do_plot=do_plot,
            window_length=window_length,
        )

    # Project back to pressure axis
    vy = np.zeros_like(vy_raw)
    for j in range(len(lat) + 1):
        vy[j, :] = np.interp(
            0.5 * (sigma_lev_bound[j, 1:] + sigma_lev_bound[j, :-1]),
            sigma_mid, vy_interp[j, :],
        )
    vy = np.where(np.isnan(vy), 0.0, vy)
    wsig = np.zeros_like(wp_raw)
    for j in range(len(lat)):
        wsig[j, :] = np.interp(sigma_lev[j, :], sigma_vect, wsig_interp[j, :])
    wp = -sp[:, np.newaxis] * np.where(np.isnan(wsig), 0.0, wsig)
    wp = np.where(p_mat > sp[:, np.newaxis], 0.0, wp)
    vy = np.where(0.5 * (p_mat_bound[:, 1:] + p_mat_bound[:, :-1]) > sp_bound[:, np.newaxis], 0.0, vy)

    return vy, wp


# ---------------------------------------------------------------------------
# Edge-preserving smoothing
# ---------------------------------------------------------------------------

def edge_preserving_smoothing(u, grad_thresh_percentile=99, min_area=100, **kwargs):
    """Detect high-gradient regions and smooth the rest via inpainting.

    Requires ``scikit-image``.

    Returns (mask, u_fixed) where mask marks the edge regions.
    """
    from skimage import measure, morphology
    from scipy.ndimage import binary_fill_holes
    from skimage.restoration import inpaint_biharmonic

    mask_and = kwargs.get('mask_and', np.ones(u.shape, dtype=bool))
    mask_or = kwargs.get('mask_or', np.zeros(u.shape, dtype=bool))

    grad_x = np.gradient(u, axis=0)
    grad_y = np.gradient(u, axis=1)
    grad_mag = np.hypot(grad_x, grad_y)

    threshold = np.percentile(grad_mag, grad_thresh_percentile)
    edges = grad_mag > threshold

    labeled = measure.label(edges)
    regions = measure.regionprops(labeled)

    mask = np.zeros(u.shape, dtype=bool)
    for region in regions:
        if region.area >= min_area:
            filled = binary_fill_holes(labeled == region.label)
            mask |= filled

    mask = morphology.binary_closing(mask, morphology.disk(2))
    mask |= mask_or
    mask &= mask_and

    u_fixed = inpaint_biharmonic(u, mask)
    return mask, u_fixed


# ---------------------------------------------------------------------------
# Map merging
# ---------------------------------------------------------------------------

def merge_maps(map_a, map_b, mask_bad_a, blur_radius=5):
    """Blend two maps using a smooth confidence mask.

    Where *mask_bad_a* is True, *map_b* is used; elsewhere *map_a*.
    A Gaussian-blurred transition provides smooth blending.
    """
    from scipy.ndimage import gaussian_filter
    mask_conf = (~mask_bad_a).astype(float)
    mask_smooth = gaussian_filter(mask_conf, sigma=blur_radius)
    mask_smooth /= mask_smooth.max()
    merged = mask_smooth * map_a + (1 - mask_smooth) * map_b
    return merged, mask_smooth


# ---------------------------------------------------------------------------
# Main transport parameter entry point
# ---------------------------------------------------------------------------

def get_transport_param(domain, config, **kwargs):
    """Compute 2D transport parameters (velocity, diffusivities).

    Returns dict with keys: 'U' (vy, wp), 'Kq' (kyy, kzz, kyz),
    'Kh' (kyy, kzz, kyz), 'geopotential'.
    """
    months = kwargs.get('months', np.arange(12))
    kpp_min = kwargs.get('kpp_min', 0.05)
    kyy_min = kwargs.get('kyy_min', 1000.0)
    if 'years' in kwargs:
        year_dict = {'year': kwargs['years']}
    else:
        year_dict = {}
    small = kwargs.get('small', 1e-3)

    lat = domain.lat.points
    lev = domain.lev.points
    lev_bound = domain.lev.bounds
    lat_bound = domain.lat.bounds

    mse_input = get_mse_input(months, config, **year_dict, small=small)

    era5_lev = mse_input['level']
    era5_lat = mse_input['latitude']
    era5_vbar = mse_input['vbar']
    era5_wbar = mse_input['wbar']
    era5_kyy_se = mse_input['K_mer_se']
    era5_kpp_se = mse_input['K_ver_se']
    era5_kyy_q = mse_input['K_mer_q']
    era5_kpp_q = mse_input['K_ver_q']
    era5_gp = mse_input['GPbar']

    vlat_raw = interpolate.RectBivariateSpline(
        era5_lat, era5_lev, era5_vbar.T, kx=1, ky=1,
    )(lat_bound, lev)
    wp_raw = interpolate.RectBivariateSpline(
        era5_lat, era5_lev, era5_wbar.T, kx=1, ky=1,
    )(lat, lev_bound)

    vy_raw = vlat_raw * np.cos(lat_bound[:, np.newaxis] * np.pi / 180)

    do_u_correction = kwargs.get('do_u_correction', True)
    do_uraw_smooth = kwargs.get('do_uraw_smooth', False)
    if do_u_correction:
        vy, wp = correct_vw_field(
            vy_raw, wp_raw, lat, lat_bound, lev, lev_bound,
            config, do_smooth=do_uraw_smooth, **kwargs,
        )
    else:
        vy, wp = vy_raw, wp_raw

    era5_kyy_se = era5_kyy_se.where(era5_kyy_se > 0.0, 0.0)
    era5_kpp_se = era5_kpp_se.where(era5_kpp_se > 0.0, 0.0)
    era5_kyy_q = era5_kyy_q.where(era5_kyy_q > 0.0, 0.0)
    era5_kpp_q = era5_kpp_q.where(era5_kpp_q > 0.0, 0.0)

    klatlat_se = interpolate.RectBivariateSpline(
        era5_lat, era5_lev, era5_kyy_se.T, kx=1, ky=1,
    )(lat_bound, lev)
    klevlev_se = interpolate.RectBivariateSpline(
        era5_lat, era5_lev, era5_kpp_se.T, kx=1, ky=1,
    )(lat, lev_bound)
    klevlat_se = np.zeros((len(lat_bound), len(lev_bound)))

    klatlat_q = interpolate.RectBivariateSpline(
        era5_lat, era5_lev, era5_kyy_q.T, kx=1, ky=1,
    )(lat_bound, lev)
    klevlev_q = interpolate.RectBivariateSpline(
        era5_lat, era5_lev, era5_kpp_q.T, kx=1, ky=1,
    )(lat, lev_bound)
    klevlat_q = np.zeros((len(lat_bound), len(lev_bound)))

    do_edge_preserving_smoothing = kwargs.get('do_edge_preserving_smoothing', True)
    if do_edge_preserving_smoothing:
        grad_thresh_percentile = kwargs.get('grad_thresh_percentile', 70.0)
        _, klatlat_se = edge_preserving_smoothing(klatlat_se, grad_thresh_percentile=grad_thresh_percentile)
        _, klevlev_se = edge_preserving_smoothing(klevlev_se, grad_thresh_percentile=grad_thresh_percentile)
        _, klatlat_q = edge_preserving_smoothing(klatlat_q, grad_thresh_percentile=grad_thresh_percentile)
        _, klevlev_q = edge_preserving_smoothing(klevlev_q, grad_thresh_percentile=grad_thresh_percentile)

    klatlat_se = np.where(klatlat_se > kyy_min, klatlat_se, kyy_min)
    klatlat_q = np.where(klatlat_q > kyy_min, klatlat_q, kyy_min)
    klevlev_se = np.where(klevlev_se > kpp_min, klevlev_se, kpp_min)
    klevlev_q = np.where(klevlev_q > kpp_min, klevlev_q, kpp_min)

    geopotential = interpolate.RectBivariateSpline(
        era5_lat, era5_lev, era5_gp.T, kx=1, ky=1,
    )(lat_bound, lev_bound)

    do_kyy_modification = kwargs.get('do_kyy_modification', True)
    if do_kyy_modification:
        kyy_mod_param_dict = {
            'fac_threshold': 2e-4, 'lat_min': -75.0, 'lat_max': 75.0,
            'lev_min': 50.0, 'blur_radius': 5,
        }
        kyy_mod_param_dict_input = kwargs.get('kyy_mod_param_dict', {})
        for k, v in kyy_mod_param_dict_input.items():
            kyy_mod_param_dict[k] = v

        ds_holton = load_xr_from_repo('./mse_files/Transport_2008_2017', config)
        if 'year' in year_dict:
            ds_holton = ds_holton.sel(year=year_dict['year'], drop=True)
        else:
            ds_holton = ds_holton.mean('year')
        month_list = [m + 1 for m in months]
        ds_holton = ds_holton.sel(month=month_list).mean('month')

        klatlat_se_holton = interpolate.RectBivariateSpline(
            ds_holton.latitude.values[::-1], ds_holton.level.values,
            ds_holton.D_phi_phi.values[:, ::-1].T, kx=1, ky=1,
        )(lat_bound, lev)

        fac_era5 = (const.Lhvap * mse_input['qbar']
                     / (const.Lhvap * mse_input['qbar']
                        + const.cp * (1 - mse_input['qbar']) * mse_input['Tbar']))
        fac_interp = interpolate.RectBivariateSpline(
            era5_lat, era5_lev, fac_era5.T, kx=1, ky=1,
        )(lat_bound, lev)

        mask_bad = np.where(fac_interp <= kyy_mod_param_dict['fac_threshold'], True, False)
        lev_mat, lat_bound_mat = np.meshgrid(lev, lat_bound)
        mask_bad &= np.where(
            (lat_bound_mat >= kyy_mod_param_dict['lat_min'])
            & (lat_bound_mat <= kyy_mod_param_dict['lat_max']),
            True, False,
        )
        mask_bad &= np.where(lev_mat >= kyy_mod_param_dict['lev_min'], True, False)

        klatlat_se_merged, _ = merge_maps(
            klatlat_se, klatlat_se_holton, mask_bad,
            blur_radius=kyy_mod_param_dict['blur_radius'],
        )
        klatlat_se = klatlat_se_merged
        klatlat_se = np.where(klatlat_se > kyy_min, klatlat_se, kyy_min)

    transport_param_dict = {
        'U': (vy, wp),
        'Kq': (klatlat_q, klevlev_q, klevlat_q),
        'Kh': (klatlat_se, klevlev_se, klevlat_se),
        'geopotential': geopotential,
    }
    return transport_param_dict
