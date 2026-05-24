"""Rev6 reference model configuration.

Ported from ``climate_runs/reference_model/versions_ref/rev6/lev100/ref_model_rev6.py``.

* ``construct_lev`` -- geometric pressure grid construction
* ``get_ref``       -- reference model (default: 100 levels, CO2 = 420 ppm)

Usage
-----
::

    from climate_runs_ext import load_project_config
    from climate_runs_ext.reference_model.rev6 import get_ref

    cfg = load_project_config()
    model = get_ref(config=cfg)
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import numpy as np
from scipy.interpolate import CubicSpline

from climlab import constants as const
from climlab import column_state

from climate_runs_ext.utils.data_loading import load_xr_from_repo, era5_grid
from climate_runs_ext.utils.era5_data import (
    SeasonTypes, Ozone, Relative_Humidity_Profile,
    get_era5_mycloud, get_surface_flux_drag_coe,
)
from climate_runs_ext.model_factory.model_generator import model_generator


# ---------------------------------------------------------------------------
# Grid construction
# ---------------------------------------------------------------------------

def construct_lev(dp_min=2.0, sp=1000.0, nlev=50):
    """Build a geometric pressure-level grid.

    Finds ratio *r* such that ``dp_min * sum(r^i, i=0..nlev-1) = sp``,
    then returns the midpoint pressures.

    Parameters
    ----------
    dp_min : float  Minimum layer thickness (hPa).
    sp     : float  Surface pressure (hPa).
    nlev   : int    Number of levels.

    Returns
    -------
    ndarray, shape (nlev,)
        Pressure at level midpoints (hPa), monotonically increasing.
    """
    a = sp / dp_min
    r = 1.0
    iter_max = 500
    conv = 1e-6
    for _ in range(iter_max):
        dr = float(-(np.sum(r ** np.arange(nlev)) - a)
                    / np.sum(np.arange(1, nlev) * r ** np.arange(0, nlev - 1)))
        r = r + dr
        if np.abs(dr / r) < conv:
            break
    dlev = dp_min * r ** np.arange(nlev)
    lev_bound = np.concatenate((np.array([0.0]), np.cumsum(dlev)))
    lev = 0.5 * (lev_bound[:-1] + lev_bound[1:])
    return lev


# ---------------------------------------------------------------------------
# Reference model
# ---------------------------------------------------------------------------

def get_ref(config, **kwargs):
    """Build the Rev6 reference climate model.

    Key characteristics:
    - 100 pressure levels by default (geometric spacing, override with nlev=)
    - CO2 = 420 ppm by default (override with CO2=)
    - 2D transport (q and MSE)
    - Analytic geopotential
    - Post-construction: zero out Kzz, W, U from SE transport;
      zero out Kzz, W from q transport; remove Geopotential Flux

    Parameters
    ----------
    config : dict
        Project configuration from ``load_project_config()``.
    **kwargs
        Override any parameter (lat, lev, season_str, etc.).

    Returns
    -------
    climlab.TimeDependentProcess
        The fully coupled reference model.
    """
    if 'season_str' in kwargs:
        season_str = kwargs.pop('season_str')
    else:
        season_str = 'Annual'
    months = SeasonTypes.months_dict[season_str]

    _, lat0 = era5_grid(config)
    lat0 = lat0[::8]
    nlev = kwargs.pop('nlev', 100)
    lev0 = construct_lev(dp_min=2.0, sp=1000.0, nlev=nlev)
    lat = kwargs.pop('lat', lat0)
    lev = kwargs.pop('lev', lev0)
    CO2_vmr = kwargs.pop('CO2', 420.0e-6)

    # `minimal_for_sarf=True` builds a stripped reference for SARF runs
    # where Ts, q, and tropospheric Tatm are pinned externally via the
    # fix_Ts / fix_q / fix_Tatm_trop Limiters. In that regime LSC, SBM
    # convection, LHF/SHF surface fluxes, and OHU are dynamically inert
    # (their tendencies are clobbered by the Limiters), so we skip
    # building them entirely. Stratospheric q + MSE 2D transport stays
    # active by default (that is the physics under study for papers
    # analysing stratospheric circulation). Skipping these subprocesses
    # also means the database files they load (sink_vs_rh, RH profile,
    # ERA5 surface-flux fitting) are never touched, so the public-release
    # data manifest can omit them.
    minimal = kwargs.pop('minimal_for_sarf', False)

    # `disable_transport=True` also skips the Moist Diffusion subprocess
    # entirely: no MoistMeridionalAdvectionDiffusion, no transport-param
    # or meridional_Kq loads. Use for pure-radiative SARF where any 2D
    # transport acting on an otherwise-pinned column is considered
    # calibration noise rather than physics. Orthogonal to
    # `minimal_for_sarf`; typical publication setup combines both.
    disable_transport = kwargs.pop('disable_transport', False)

    # --- Timesteps and numerical knobs (all overridable via kwargs) ---
    short_timestep = kwargs.pop('short_timestep', const.seconds_per_hour)
    long_timestep = kwargs.pop('long_timestep', 24 * const.seconds_per_hour)
    timestep_dict = kwargs.pop('timestep_dict', {
        'Moist Diffusion': short_timestep,
        'Radiation': long_timestep,
        'Convection': short_timestep,
    })

    n_timestep_sw = kwargs.pop('n_timestep_sw', 1)
    n_timestep_lw = kwargs.pop('n_timestep_lw', 1)

    water_depth = kwargs.pop('water_depth', 10.0)
    condensation_time = kwargs.pop('condensation_time', 2 * short_timestep)

    lev_mat = np.repeat(lev[np.newaxis, :], len(lat), axis=0)

    # Cloud / RRTMG sampling params — each individually overridable.
    cld_param_dict = {
        'n_rrtmg_repeat':      kwargs.pop('n_rrtmg_repeat', 5),
        'do_seed_permutation': kwargs.pop('do_seed_permutation', True),
        'do_col_by_col':       kwargs.pop('do_col_by_col', True),
    }
    params = {
        'season_str': season_str, 'water_depth': water_depth,
        'long_timestep': long_timestep, 'short_timestep': short_timestep,
        'n_timestep_sw': n_timestep_sw, 'n_timestep_lw': n_timestep_lw,
        **cld_param_dict,
    }

    state = column_state(lev=lev, lat=lat, water_depth=params['water_depth'])

    # mycloud: default = ERA5 climatology. Pass mycloud=None explicitly
    # for zero clouds, or pass a dict for a custom cloud field.
    if 'mycloud' in kwargs:
        mycloud = kwargs.pop('mycloud')
    else:
        mycloud = get_era5_mycloud(state.Tatm.domain, months, config)

    # GHGs: accept full dict; per-species overrides merge on top of defaults.
    # CO2= kwarg kept for backward compat; GHGs['CO2'] wins if both set.
    default_GHGs = {
        'CO2': CO2_vmr, 'CH4': 1935.0e-9, 'N2O': 337.0e-9,
    }
    user_GHGs = kwargs.pop('GHGs', {})
    GHGs = {**default_GHGs, **user_GHGs}
    if 'O3' not in GHGs:
        GHGs['O3'] = Ozone(lev, lat, months, config)

    params['p_turb_layer'] = kwargs.pop('p_turb_layer', 50.0)
    params['timestep_dict'] = timestep_dict
    params['p_min_q_transfer'] = kwargs.pop('p_min_q_transfer', 175.0)
    params['GHGs'] = GHGs
    params['rh_ref_lsc'] = kwargs.pop('rh_ref_lsc', 1.0 * np.ones_like(lev_mat))
    params['condensation_time'] = condensation_time
    params['mycloud'] = mycloud
    params['do_analytic_gp'] = kwargs.pop('do_analytic_gp', True)

    if not minimal:
        # --- LSC sink function (skipped for minimal SARF: q is pinned) ---
        lsc_sink_file = kwargs.pop(
            'lsc_sink_file', './mse_files/sink_vs_rh_15_05',
        )
        sink_func_arr = load_xr_from_repo(
            lsc_sink_file, config, file_type='numpy',
        )
        sink_func_raw = CubicSpline(
            sink_func_arr['rh'],
            sink_func_arr['sink'] / const.Lhvap,
        )

        def lsc_sink_func(state):
            from climlab.utils import thermo
            rh = state['q'] / thermo.qsat(
                state['Tatm'], state['Tatm'].domain.lev.points,
            )
            return sink_func_raw(rh) + 0 * state['Tatm']

        # --- SBM convection RH profile (skipped for minimal: T,q pinned) --
        default_conv_dict = {
            'tau_bm': const.seconds_per_hour,
            'do_envsat': False, 'do_simp': False,
            'do_shallower': False, 'do_changeqref': True,
            'rh': Relative_Humidity_Profile(lev, lat, months, config),
        }
        user_conv_dict = kwargs.pop('conv_dict', {})
        conv_dict = {**default_conv_dict, **user_conv_dict}
        params['conv_dict'] = conv_dict
        params['do_conv'] = kwargs.pop('do_conv', True)
        params['q_transport_2d'] = kwargs.pop('q_transport_2d', True)
        params['mse_transport_2d'] = kwargs.pop('mse_transport_2d', True)
        params['add_fixed_ohu'] = kwargs.pop('add_fixed_ohu', True)
        params['lsc_sink_func'] = kwargs.pop('lsc_sink_func', lsc_sink_func)

        # 2D transport Kyy tuning parameters — each individually overridable.
        transport_params = {
            'kpp_min':            kwargs.pop('kpp_min', 0.05),
            'kyy_min':            kwargs.pop('kyy_min', 1000.0),
            'small':              kwargs.pop('small', 1e-3),
            'do_kyy_modification':
                                  kwargs.pop('do_kyy_modification', True),
            'do_pole_cutoff':     kwargs.pop('do_pole_cutoff', True),
            'lat_max':            kwargs.pop('lat_max', 75.0),
            'lat_fall_scale':     kwargs.pop('lat_fall_scale', 5.0),
            'do_u_correction':    kwargs.pop('do_u_correction', True),
            'do_uraw_smooth':     kwargs.pop('do_uraw_smooth', False),
            'do_edge_preserving_smoothing':
                                  kwargs.pop('do_edge_preserving_smoothing', True),
            'kyy_mod_param_dict': kwargs.pop('kyy_mod_param_dict', {}),
        }
        params = {**params, **transport_params}

        # --- surface drag coefficients (skipped for minimal: Ts,q pinned) -
        # If caller supplied a ready-made Cd_dict (e.g. scalar Cd_lhf for
        # OLD-config reproduction), use it verbatim and skip ERA5 fitting.
        if 'Cd_dict' in kwargs:
            params['Cd_dict'] = kwargs.pop('Cd_dict')
        else:
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

            Cd_lhf, Cd_shf, _ = get_surface_flux_drag_coe(
                config, new_lat=lat, months=months,
                **years_dict, **Cd_param_dict,
            )
            params['Cd_dict'] = {
                'Cd_lhf': Cd_lhf, 'Cd_shf': Cd_shf,
                'turbulent_do_analytic': True,
            }
    else:
        # Minimal: explicitly disable the subprocesses whose construction
        # would otherwise trigger database loads AND that are dynamically
        # degenerate when the troposphere is pinned to ERA5.
        #
        # KEEP enabled (still needed):
        #   - q_transport_2d / mse_transport_2d: stratospheric Kyy
        #     redistributes meridional heat in the (free) stratosphere,
        #     so it affects the stratosphere-adjusted result. The paper
        #     is ABOUT the stratospheric transport.
        #
        # DISABLE (degenerate when Ts / q / tropospheric Tatm pinned):
        #   - LSC (q pinned)            → no sink_vs_rh load
        #   - SBM convection            → no RH profile load
        #   - LHF / SHF                 → no Cd fitting files load
        #   - OHU                       → no balance file load
        params['do_lsc'] = False
        params['do_conv'] = False
        params['add_fixed_ohu'] = False
        params['q_transport_2d'] = True
        params['mse_transport_2d'] = True
        # Empty Cd_dict so model_builder doesn't add LHF/SHF subprocesses
        # AND model_generator skips its get_surface_flux_drag_coe load.
        params['Cd_dict'] = {}
        # Explicit None so model_generator skips its sink_vs_rh auto-load.
        params['lsc_sink_func'] = None
        # Empty conv_dict so model_generator skips its RH-profile auto-load.
        params['conv_dict'] = {}

    if disable_transport:
        params['disable_transport'] = True

    model = model_generator(config, lat=lat, lev=lev, **params, **kwargs)

    # --- post-construction: zero out 2D transport vertical components ---
    # Only meaningful when 2D MSE transport was actually built (SE Transport
    # 2D subprocess exists). For 1D transport (q_transport_2d=False or
    # mse_transport_2d=False) there's nothing to zero.
    md = model.subprocess.get('Moist Diffusion')
    if md is not None and 'SE Transport 2D' in getattr(
            md, 'diff_mse', md).subprocess:
        md.diff_mse.remove_subprocess('Geopotential Flux')
        se = md.diff_mse.subprocess['SE Transport 2D']
        se.Kzz[:] = 0.0
        se.W[:] = 0.0
        se.U[:] = 0.0
        if hasattr(md, 'diff_q'):
            md.diff_q.Kzz[:] = 0.0
            md.diff_q.W[:] = 0.0

    return model
