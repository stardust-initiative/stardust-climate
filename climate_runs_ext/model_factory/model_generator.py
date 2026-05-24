"""High-level model generator -- wraps model_builder with defaults.

Ported from ``climate_runs/reference_model/reference_model_generator.py``:
``model_generator()`` (lines 16-221).

Usage
-----
::

    from climate_runs_ext import load_project_config
    from climate_runs_ext.model_factory.model_generator import model_generator

    cfg = load_project_config()
    model = model_generator(config=cfg, lat=lat, lev=lev)
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import numpy as np
from scipy.interpolate import CubicSpline

import climlab
from climlab import constants as const
from climlab import TimeDependentProcess

from climate_runs_ext.utils.data_loading import load_xr_from_repo, era5_grid
from climate_runs_ext.utils.era5_data import (
    SeasonTypes, Ozone, Relative_Humidity_Profile,
    get_surface_flux_drag_coe, meridional_Kq,
)
from climate_runs_ext.utils.state_helpers import lev_grid_construct
from climate_runs_ext.utils.state_io import (
    update_model_from_file, iteratively_update_internal,
)
from climate_runs_ext.model_factory.model_builder import (
    get_rce_sbm_model_annual_avg,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def K_func(lat, param_dict):
    """CubicSpline interpolant for MSE diffusivity (1D fallback)."""
    return CubicSpline(param_dict['K_func_lat'], param_dict['K_func_K'])(lat)


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def model_generator(config, **kwargs):
    """Build a radiative-convective equilibrium (RCE) climate model.

    This is the high-level factory that:
    1. Sets the lat/lev grid (ERA5 default or custom).
    2. Computes GHG concentrations (with ERA5 O3 by default).
    3. Computes surface drag coefficients from ERA5 fluxes.
    4. Computes 2D transport parameters if requested.
    5. Loads the LSC sink function from the repository.
    6. Builds the model via ``get_rce_sbm_model_annual_avg``.
    7. Optionally loads state from a file or existing model.

    Parameters
    ----------
    config : dict
        Project configuration from ``load_project_config()``.
    **kwargs
        See inline comments for the many configuration knobs.

    Returns
    -------
    climlab.TimeDependentProcess
        The fully coupled climate model.
    """
    season_str = kwargs.get('season_str', 'Annual')
    months = SeasonTypes.months_dict[season_str]
    model_file_path = kwargs.get('model_file_path', None)
    model_source = kwargs.get('model_source', None)
    water_depth = kwargs.get('water_depth', 10.0)

    long_timestep = kwargs.get('long_timestep', 24 * const.seconds_per_hour)
    short_timestep = kwargs.get('short_timestep', 2 * const.seconds_per_hour)
    timestep_dict = kwargs.get('timestep_dict', {
        'Moist Diffusion': short_timestep,
        'Radiation': short_timestep,
        'Convection': short_timestep,
    })
    assert long_timestep // short_timestep == long_timestep / short_timestep, \
        'long_timestep must be an integer multiple of short_timestep'
    n_timestep_rad = int(long_timestep // short_timestep)
    n_timestep_sw = kwargs.get('n_timestep_sw', n_timestep_rad)
    n_timestep_lw = kwargs.get('n_timestep_lw', n_timestep_rad)

    cld_param_dict = {'n_rrtmg_repeat': 1, 'do_seed_permutation': False}
    do_col_by_col = kwargs.get('do_col_by_col', True)
    do_seed_permutation = kwargs.get('do_seed_permutation', cld_param_dict['do_seed_permutation'])
    n_rrtmg_repeat = kwargs.get('n_rrtmg_repeat', cld_param_dict['n_rrtmg_repeat'])
    rad_with_aero_param_dict = kwargs.get('rad_with_aero_param_dict', {})

    # --- cloud feedback ---
    do_cld_fb = kwargs.get('do_cld_fb', False)
    if do_cld_fb:
        str_rel_coord = '_RelCoord' if kwargs.get('do_rel_coord', False) else ''
        str_2nd_ord = '_ord2' if kwargs.get('do_second_order', False) else ''
        str_cc_trans = '_ccTrans' if kwargs.get('do_cc_transformation', False) else ''
        str_prod_fit = '_prodFit' if kwargs.get('do_clouds_product_fit', False) else ''
        str_multi_year = '_MultiYear' if kwargs.get('do_multi_year_avg', True) else ''
        cloud_sensitivity_dataset_filename = (
            f'cloud_sensitivity_files/cloud_param_historic_sensitivity_ver2'
            f'{str_rel_coord}{str_2nd_ord}{str_cc_trans}{str_prod_fit}{str_multi_year}'
        )
        cld_fb_dict = {
            'cloud_sensitivity_dataset_filename': cloud_sensitivity_dataset_filename,
            'cld_fb_Tmin': kwargs.get('cld_fb_Tmin', 200.0),
        }
    else:
        cld_fb_dict = {}

    # --- grid ---
    if 'lat' in kwargs:
        lat = kwargs['lat']
    else:
        _, lat = era5_grid(config)
        lat = lat[::8]
    if 'lev' in kwargs:
        lev = kwargs['lev']
    else:
        num_lev = 50
        p0, ps = 0.0, 1000.0
        fac = 0.5
        dp0 = dps = fac * (ps - p0) / num_lev
        lev = lev_grid_construct(num_lev, dp0, dps, p0=p0)

    # --- GHGs ---
    GHGs = {
        'CO2': 420.0e-6, 'CH4': 1935.0e-9, 'N2O': 337.0e-9,
        'O3': Ozone(lev, lat, months, config),
    }
    GHGs_input = kwargs.get('GHGs', {})
    for k, v in GHGs_input.items():
        GHGs[k] = v

    Tmin = kwargs.get('Tmin', 180.0)
    qmin = kwargs.get('qmin', 5e-6)
    pmin = kwargs.get('pmin', 1.0)

    # --- surface drag coefficients ---
    lat_win_lhf = kwargs.get('lat_win_lhf', (-70.0, 90.0))
    lat_win_shf = kwargs.get('lat_win_shf', (-63.0, 52.0))

    if 'years' in kwargs:
        years_dict = {'years': kwargs['years']}
    else:
        years_dict = {}
    Cd_param_dict = {
        'n': 0, 'do_smooth': True, 'dT_min': 0.0, 'dq_min': 0.0,
        'override_q': True, 'Cd_lhf_min': 1e-4, 'Cd_lhf_max': None,
        'Cd_shf_min': 1e-4, 'Cd_shf_max': 0.01,
        'lat_win_lhf': lat_win_lhf, 'lat_win_shf': lat_win_shf,
    }
    Cd_param_dict_input = kwargs.get('Cd_param_dict', {})
    for k, v in Cd_param_dict_input.items():
        Cd_param_dict[k] = v

    # Fit drag coefficients (which loads the ERA5 SURFACE_FLUXES file) only
    # when the caller hasn't supplied Cd_dict. The minimal SARF config passes
    # Cd_dict={} so LHF/SHF are inert; computing-then-discarding the fit would
    # otherwise force a needless SURFACE_FLUXES load. Mirrors the LSC-sink
    # guard below.
    if 'Cd_dict' in kwargs:
        Cd_dict = kwargs['Cd_dict']
    else:
        Cd_lhf, Cd_shf, _ = get_surface_flux_drag_coe(
            config, new_lat=lat, months=months, **years_dict, **Cd_param_dict,
        )
        Cd_dict = {
            'Cd_lhf': Cd_lhf, 'Cd_shf': Cd_shf,
            'turbulent_do_analytic': False,
        }

    p_turb_layer = kwargs.get('p_turb_layer', 50.0)
    p_min_q_transfer = kwargs.get('p_min_q_transfer', 0.0)

    rh_ref_lsc_dict = {'rh_ref_lsc': kwargs['rh_ref_lsc']} if 'rh_ref_lsc' in kwargs else {}

    # --- LSC sink function ---
    if 'lsc_sink_func' in kwargs:
        lsc_sink_func = kwargs['lsc_sink_func']
    else:
        sink_func_arr = load_xr_from_repo(
            './mse_files/sink_vs_rh_15_05', config, file_type='numpy',
        )
        sink_func_raw = CubicSpline(
            sink_func_arr['rh'],
            sink_func_arr['sink'] / const.Lhvap,
        )

        def lsc_sink_func(state):
            from climlab.utils import thermo
            rh = state['q'] / thermo.qsat(state['Tatm'], state['Tatm'].domain.lev.points)
            return sink_func_raw(rh) + 0 * state['Tatm']

    condensation_time = kwargs.get('condensation_time', short_timestep)
    mycloud = kwargs.get('mycloud', None)

    # --- convection ---
    # Load the ERA5 RH reference profile only when convection is active and the
    # caller hasn't supplied its own 'rh'. The minimal SARF config sets
    # do_conv=False and passes conv_dict={}, so the profile would otherwise be
    # loaded and immediately discarded. Mirrors the LSC-sink guard above.
    conv_dict = {
        'tau_bm': 3 * short_timestep,
        'do_envsat': False, 'do_simp': False,
        'do_shallower': False, 'do_changeqref': True,
    }
    conv_dict_input = kwargs.get('conv_dict', {})
    if kwargs.get('do_conv', False) and 'rh' not in conv_dict_input:
        conv_dict['rh'] = Relative_Humidity_Profile(lev, lat, months, config)
    for k, v in conv_dict_input.items():
        conv_dict[k] = v

    # --- build state ---
    full_state0 = climlab.column_state(lev=lev, lat=lat, water_depth=water_depth)
    if 'ref_state' in kwargs:
        for k in kwargs['ref_state'].keys():
            if k in full_state0.keys():
                full_state0[k] = kwargs['ref_state'][k] + 0.0 * full_state0[k]
            else:
                assert np.all(full_state0['Tatm'].shape == kwargs['ref_state'][k].shape), \
                    f'axis {k} with incompatible domain with Tatm'
                full_state0[k] = kwargs['ref_state'][k] + 0.0 * full_state0['Tatm']
    full_state = kwargs.get('full_state', full_state0)

    # --- transport ---
    # When `disable_transport=True`, skip the entire Moist Diffusion build:
    # no transport-param load, no meridional_Kq / K_func fallbacks, no
    # MoistMeridionalAdvectionDiffusion subprocess. Paper-critical for
    # pure-radiative SARF: with Ts/q/tropospheric Tatm pinned, any
    # residual 2D transport in the (free) stratosphere is dominated by
    # calibration noise in Kyy rather than by the radiative adjustment
    # the paper is measuring. Setting this True gives a strictly
    # column-by-column radiative SARF.
    disable_transport = kwargs.get('disable_transport', False)
    q_transport_2d = kwargs.get('q_transport_2d', True)
    mse_transport_2d = kwargs.get('mse_transport_2d', True)
    do_analytic_gp = kwargs.get('do_analytic_gp', False)

    if disable_transport:
        # Skip the Moist Diffusion build entirely. Nothing in diffus_*_dict,
        # and `do_moist=False` passed to model_builder below so its
        # `do_diff` check stays False (no MoistMeridionalAdvectionDiffusion
        # subprocess is constructed). No Transport file / meridional_Kq
        # loads are triggered either.
        diffus_1d_dict = {}
        diffus_2d_dict = {}
        do_moist_value = False
    else:
        do_moist_value = True
        if q_transport_2d or mse_transport_2d:
            from climate_runs_ext.utils.transport_params import get_transport_param
            kpp_min = kwargs.get('kpp_min', 0.05)
            kyy_min = kwargs.get('kyy_min', 1000.0)
            small = kwargs.get('small', 1e-3)
            do_kyy_modification = kwargs.get('do_kyy_modification', True)
            do_pole_cutoff = kwargs.get('do_pole_cutoff', True)
            lat_max = kwargs.get('lat_max', 75.0)
            lat_fall_scale = kwargs.get('lat_fall_scale', 5.0)
            do_u_correction = kwargs.get('do_u_correction', True)
            do_uraw_smooth = kwargs.get('do_uraw_smooth', False)
            do_edge_preserving_smoothing = kwargs.get('do_edge_preserving_smoothing', True)
            kyy_mod_param_dict_input = kwargs.get('kyy_mod_param_dict', {})

            transport_param_dict = get_transport_param(
                full_state.Tatm.domain, config, months=months,
                kpp_min=kpp_min, kyy_min=kyy_min, **years_dict,
                small=small, do_kyy_modification=do_kyy_modification,
                do_pole_cutoff=do_pole_cutoff, lat_max=lat_max,
                lat_fall_scale=lat_fall_scale, do_u_correction=do_u_correction,
                do_uraw_smooth=do_uraw_smooth,
                do_edge_preserving_smoothing=do_edge_preserving_smoothing,
                kyy_mod_param_dict_input=kyy_mod_param_dict_input,
            )

        diffus_2d_dict = {}
        diffus_1d_dict = {}
        if q_transport_2d:
            Uq_diffusion = kwargs.get('Uq_diffusion', transport_param_dict['U'])
            Kq_diffusion = kwargs.get('Kq_diffusion', transport_param_dict['Kq'])
            diffus_2d_dict = {**diffus_2d_dict, 'Uq_diffusion': Uq_diffusion, 'Kq_diffusion': Kq_diffusion}
        else:
            # Lazy eval: only fall back to meridional_Kq (which loads
            # the Transport database file) if the caller didn't supply
            # a Kq_diffusion kwarg. Python evaluates default args
            # eagerly for .get(), so we must use `in` check.
            if 'Kq_diffusion' in kwargs:
                Kq_diffusion = kwargs['Kq_diffusion']
            else:
                Kq_diffusion = meridional_Kq(
                    full_state.Tatm.domain.lat.bounds, config,
                )
            diffus_1d_dict = {**diffus_1d_dict, 'Kq_diffusion': Kq_diffusion}

        if mse_transport_2d:
            U_diffusion = kwargs.get('U_diffusion', transport_param_dict['U'])
            K_diffusion = kwargs.get('K_diffusion', transport_param_dict['Kh'])
            if do_analytic_gp:
                geopotential_dict = {'do_analytic_gp': True}
            else:
                geopotential = kwargs.get('geopotential', transport_param_dict['geopotential'])
                geopotential_dict = {'do_analytic_gp': False, 'geopotential': geopotential}
            diffus_2d_dict = {**diffus_2d_dict, 'U_diffusion': U_diffusion, 'K_diffusion': K_diffusion, **geopotential_dict}
        else:
            K_func_param_dict = {
                'K_func_lat': [-90, -60, 0.0, 60.0, 90.0],
                'K_func_K': [2e6, 3e6, 0.5e6, 3e6, 2e6],
            }
            K_diffusion = kwargs.get('K_diffusion', K_func(full_state.Tatm.domain.lat.bounds, K_func_param_dict))
            diffus_1d_dict = {**diffus_1d_dict, 'K_diffusion': K_diffusion}

    bound_dict = {'q': (qmin, np.inf), 'Tatm': (Tmin, np.inf), 'Ts': (Tmin, np.inf)}

    add_fixed_ohu = kwargs.get('add_fixed_ohu', True)
    do_lsc = kwargs.get('do_lsc', True)
    do_conv = kwargs.get('do_conv', False)

    # Optional external prescribed surface fluxes (forcing test mode).
    external_flux_kwargs = {}
    if 'LHF_external' in kwargs:
        external_flux_kwargs['LHF_external'] = kwargs['LHF_external']
    if 'SHF_external' in kwargs:
        external_flux_kwargs['SHF_external'] = kwargs['SHF_external']

    model_ref = get_rce_sbm_model_annual_avg(
        full_state, config,
        q_transport_2d=q_transport_2d, mse_transport_2d=mse_transport_2d,
        conv_dict=conv_dict, mycloud=mycloud, GHGs=GHGs,
        **Cd_dict, do_merid_advection=True,
        **external_flux_kwargs,
        p_turb_layer=p_turb_layer,
        short_timestep=short_timestep, long_timestep=long_timestep,
        condensation_time=condensation_time,
        lsc_sink_func=lsc_sink_func,
        bound_dict=bound_dict, pmin=pmin, do_moist=do_moist_value,
        p_min_q_transfer=p_min_q_transfer,
        do_seed_permutation=do_seed_permutation,
        do_col_by_col=do_col_by_col,
        n_rrtmg_repeat=n_rrtmg_repeat,
        rad_with_aero_param_dict=rad_with_aero_param_dict,
        **cld_fb_dict, **diffus_1d_dict, **diffus_2d_dict,
        timestep_dict=timestep_dict,
        n_timestep_sw=n_timestep_sw, n_timestep_lw=n_timestep_lw,
        add_fixed_ohu=add_fixed_ohu,
        do_conv=do_conv, do_lsc=do_lsc,
        **rh_ref_lsc_dict, season_str=season_str,
    )

    # --- optionally load state from file or existing model ---
    do_update = False
    load_timeave = kwargs.get('load_timeave', False)
    if isinstance(model_source, TimeDependentProcess):
        update_dict = {'model_source': model_source, 'load_timeave': load_timeave}
        do_update = True
    elif isinstance(model_file_path, str):
        update_dict = {'file_path': model_file_path, 'load_timeave': load_timeave}
        do_update = True
    if do_update:
        update_model_from_file(model_ref, **update_dict)

    if 'ref_state' in kwargs:
        iteratively_update_internal(model_ref, kwargs['ref_state'])

    return model_ref
