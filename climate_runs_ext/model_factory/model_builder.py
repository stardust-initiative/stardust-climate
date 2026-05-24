"""Core model assembly -- build a coupled RCE climate model.

Ported from ``climate_runs/utils/utils_methods.py``:
``get_rce_sbm_model_annual_avg`` (lines 1448-1638).

Migration notes
---------------
* ``RRTMG`` → ``RRTMG_extended`` from climlab-stardust-extension
* ``SimplifiedBettsMiller`` → ``SimplifiedBettsMiller_extended``
* ``LatentHeatFlux`` / ``SensibleHeatFlux`` → extended versions
* ``LargeScaleCondensation`` → from climlab-stardust-extension
* ``MoistMeridionalAdvectionDiffusion`` → from climlab-stardust-extension
* ``couple(procs, bound_dict=...)`` → ``couple(procs)`` + ``add_bounds()``
* ``do_fixed_cells`` for stratospheric q → Limiter subprocess

Usage
-----
::

    from climate_runs_ext.model_factory.model_builder import (
        get_rce_sbm_model_annual_avg,
    )
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import numpy as np

from climlab import constants as const
from climlab import couple
from climlab.domain import field
from climlab.dynamics import MeridionalAdvectionDiffusion
from climlab.process.limiter import Limiter

from climlab_stardust_extension.radiation.rrtm import RRTMG_extended as RRTMG
from climlab_stardust_extension.convection.simplified_betts_miller import (
    SimplifiedBettsMiller_extended as SimplifiedBettsMiller,
)
from climlab_stardust_extension.dynamics.large_scale_condensation import (
    LargeScaleCondensation_extended as LargeScaleCondensation,
)
from climlab_stardust_extension.surface.turbulent import (
    LatentHeatFlux_extended as LatentHeatFlux,
    SensibleHeatFlux_extended as SensibleHeatFlux,
)
from climlab_stardust_extension.dynamics.meridional_moist_diffusion import (
    MoistMeridionalAdvectionDiffusion,
)

from climate_runs_ext.utils.era5_data import (
    SeasonTypes, lat_avg, Seasonal_insolation, Surface_albedo,
    get_era5_mycloud, oceanic_heat_uptake,
)
from climate_runs_ext.utils.model_control import add_bounds
from climate_runs_ext.diagnostics.cloud_feedback import CloudFeedback
from climate_runs_ext.diagnostics.fixed_oceanic_heat_uptake import FixedOceanicHeatUptake
from climate_runs_ext.diagnostics.relative_humidity import RelativeHumidity


# ---------------------------------------------------------------------------
# Core model builder
# ---------------------------------------------------------------------------

def get_rce_sbm_model_annual_avg(
    full_state, config,
    Tinit_dict=None, season_str='Annual',
    albedo=None, mycloud=None,
    GHGs=None,
    short_timestep=const.seconds_per_hour,
    long_timestep=const.seconds_per_day,
    conv_dict=None, rh_ref_lsc=0.9,
    condensation_time=4 * const.seconds_per_hour,
    **kwargs
):
    """Assemble a full coupled radiative-convective-diffusive climate model.

    Parameters
    ----------
    full_state : AttrDict
        Column state containing Tatm, Ts and optionally q.
    config : dict
        Project configuration from ``load_project_config()``.
    Tinit_dict : dict, optional
        Override initial values for Ts, Tatm, q.
    season_str : str
        Season for insolation/cloud averaging (default 'Annual').
    albedo : array-like or None
        Surface albedo; None loads ERA5 default.
    mycloud : dict or None
        Cloud parameters dict; None loads ERA5 default.
    GHGs : dict or None
        Greenhouse gas VMR dict (CO2, CH4, N2O, O3).
    short_timestep : float
        Short timestep in seconds for fast processes.
    long_timestep : float
        Long timestep in seconds for slow processes.
    conv_dict : dict, optional
        Convection scheme parameters.
    rh_ref_lsc : float
        Relative humidity threshold for large-scale condensation.
    condensation_time : float
        Relaxation timescale for LSC (seconds).

    Returns
    -------
    climlab.TimeDependentProcess
        The fully coupled climate model.
    """
    if Tinit_dict is None:
        Tinit_dict = {}
    if GHGs is None:
        GHGs = {'CO2': 420.0e-6, 'CH4': 1935.0e-9, 'N2O': 337.0e-9}
    if conv_dict is None:
        conv_dict = {}

    do_seed_permutation = kwargs.get('do_seed_permutation', False)
    do_col_by_col = kwargs.get('do_col_by_col', False)
    n_rrtmg_repeat = kwargs.get('n_rrtmg_repeat', 1)
    q_transport_2d = kwargs.get('q_transport_2d', False)
    mse_transport_2d = kwargs.get('mse_transport_2d', False)
    timestep_dict = kwargs.get('timestep_dict', {})
    do_spectral_diagnostics = kwargs.get('do_spectral_diagnostics', False)

    assert long_timestep // short_timestep == long_timestep / short_timestep, \
        'long_timestep must be an integer multiple of short_timestep'
    n_timestep_rad = int(long_timestep // short_timestep)
    n_timestep_sw = kwargs.get('n_timestep_sw', n_timestep_rad)
    n_timestep_lw = kwargs.get('n_timestep_lw', n_timestep_rad)

    lat = full_state.Ts.domain.lat.points
    bound_dict = kwargs.get('bound_dict', {})

    # --- initialise state ---
    if 'Ts' in Tinit_dict:
        full_state.Ts[:] = Tinit_dict['Ts']
    if 'Tatm' in Tinit_dict:
        full_state.Tatm[:] = Tinit_dict['Tatm']

    temp_state = {'Tatm': full_state['Tatm'], 'Ts': full_state['Ts']}
    if 'q' in Tinit_dict:
        full_state['q'] = Tinit_dict['q']
    elif 'q' not in full_state.keys():
        full_state['q'] = 5e-6 + 0.0 * temp_state['Tatm']
    atm_state = {'Tatm': full_state['Tatm'], 'q': full_state['q']}

    months = SeasonTypes.months_dict[season_str]

    # --- cloud setup ---
    do_cld_fb = False
    if not isinstance(mycloud, dict):
        cloud_dict = get_era5_mycloud(full_state.Tatm.domain, months, config)
    else:
        cloud_sensitivity_dataset_filename = kwargs.get('cloud_sensitivity_dataset_filename', '')
        do_cc_transformation = kwargs.get('do_cc_transformation', False)
        do_cld_fb_rel_coord = kwargs.get('do_cld_fb_rel_coord', False)
        do_product_fit = kwargs.get('do_clouds_product_fit', False)
        if len(cloud_sensitivity_dataset_filename) > 0:
            dt_cld_fb = timestep_dict.get('Cloud Feedback', short_timestep)
            cld_fb = CloudFeedback(
                name='Cloud Feedback', state=atm_state,
                mycloud0=mycloud,
                sensitivity_dataset_filename=cloud_sensitivity_dataset_filename,
                config=config,
                n_avg=int(long_timestep / dt_cld_fb),
                timestep=dt_cld_fb,
                do_cc_transformation=do_cc_transformation,
                do_product_fit=do_product_fit,
                do_rel_coord=do_cld_fb_rel_coord,
            )
            do_cld_fb = True
            cloud_dict = cld_fb.cloud_dict
        else:
            cloud_dict = mycloud

    if not isinstance(albedo, (float, np.ndarray)):
        albedo = Surface_albedo(lat, months, config)

    rad_with_aero_param_dict = kwargs.get('rad_with_aero_param_dict', {})

    # --- insolation ---
    insolation = field.Field(Seasonal_insolation(lat, season_str), temp_state['Ts'].domain)
    if season_str == SeasonTypes.Annual:
        a = lat_avg(insolation, lat) * 4 / const.S0
        insolation /= a
    coszen = insolation / const.S0

    # --- radiation ---
    rad = RRTMG(
        name='Radiation', state=temp_state,
        specific_humidity=full_state.q,
        albedo=albedo, S0=const.S0, coszen=coszen,
        irradiance_factor=np.ones_like(coszen),
        **cloud_dict,
        timestep=timestep_dict.get('Radiation', short_timestep),
        n_timestep_sw=n_timestep_sw, n_timestep_lw=n_timestep_lw,
        **rad_with_aero_param_dict,
        do_seed_permutation=do_seed_permutation,
        n_rrtmg_repeat=n_rrtmg_repeat,
        do_col_by_col=do_col_by_col,
        return_spectral_olr=do_spectral_diagnostics,
        return_spectral_asr=do_spectral_diagnostics,
    )
    for ghg, vmr in GHGs.items():
        rad.absorber_vmr[ghg] = vmr

    # --- surface fluxes ---
    surface_list = []
    turbulent_do_analytic = kwargs.get('turbulent_do_analytic', False)
    # Optional external-forcing hooks for diagnostic / prescribed-flux runs
    lhf_external = kwargs.get('LHF_external', None)
    shf_external = kwargs.get('SHF_external', None)
    if 'Cd_lhf' in kwargs:
        dt_lhf = timestep_dict.get('LHF', short_timestep)
        lhf_kwargs = dict(
            name='LHF', state=full_state,
            Cd=kwargs['Cd_lhf'], do_analytic=turbulent_do_analytic,
            p_turb_layer=kwargs.get('p_turb_layer', 50.0),
            timestep=dt_lhf,
        )
        if lhf_external is not None:
            lhf_kwargs['do_external'] = True
            lhf_kwargs['LHF_external'] = lhf_external
        lhf = LatentHeatFlux(**lhf_kwargs)
        surface_list.append(lhf)

    if 'Cd_shf' in kwargs:
        dt_shf = timestep_dict.get('SHF', short_timestep)
        shf_kwargs = dict(
            name='SHF', state=temp_state,
            Cd=kwargs['Cd_shf'], do_analytic=turbulent_do_analytic,
            p_turb_layer=kwargs.get('p_turb_layer', 50.0),
            timestep=dt_shf,
        )
        if shf_external is not None:
            shf_kwargs['do_external'] = True
            shf_kwargs['SHF_external'] = shf_external
        shf = SensibleHeatFlux(**shf_kwargs)
        surface_list.append(shf)

    # --- convection ---
    do_rh_proc = conv_dict.get('do_rh_proc', False)
    if do_rh_proc:
        do_simplified_dict = {'do_simplified': kwargs['do_simplified']} if 'do_simplified' in kwargs else {}
        small_dict = {'small': kwargs['rh_small']} if 'rh_small' in kwargs else {}
        do_era5_dict = {'do_era5': kwargs['do_era5']} if 'do_era5' in kwargs else {}
        params = {**do_simplified_dict, **small_dict, **do_era5_dict}
        rh_proc = RelativeHumidity(
            name='Relative Humidity',
            state={'Tatm': full_state['Tatm'], 'q': full_state['q']},
            **params,
        )
        rh_ref = rh_proc.rh
    else:
        rh_ref = conv_dict.get('rh', 0.8)

    q_threshold = kwargs.get('q_threshold', 2e-4)
    pmin = kwargs.get('pmin', 10.0)
    do_conv = kwargs.get('do_conv', False)
    if do_conv:
        conv = [SimplifiedBettsMiller(
            name='Convection', state=full_state,
            tau_bm=conv_dict.get('tau_bm', 3600.0), rhbm=rh_ref,
            do_simp=conv_dict.get('do_simp', False),
            do_shallower=conv_dict.get('do_shallower', False),
            do_changeqref=conv_dict.get('do_changeqref', True),
            do_envsat=conv_dict.get('do_envsat', False),
            do_taucape=conv_dict.get('do_taucape', False),
            capetaubm=conv_dict.get('capetaubm', False),
            tau_min=conv_dict.get('tau_min', 2400.0),
            q_threshold=q_threshold,
            timestep=timestep_dict.get('Convection', short_timestep),
            pmin=pmin,
        )]
    else:
        conv = []

    # --- large-scale condensation ---
    do_lsc = kwargs.get('do_lsc', True)
    if do_lsc:
        lsc = [LargeScaleCondensation(
            name='Large Scale Condensation', state=full_state,
            sink_func=kwargs.get('lsc_sink_func', None),
            condensation_time=condensation_time, RH_ref=rh_ref_lsc,
            timestep=timestep_dict.get('Large Scale Condensation', short_timestep),
        )]
    else:
        lsc = []

    if len(surface_list) > 0:
        surface = [couple(surface_list, name='Slab')]
    else:
        surface = []

    # --- atmosphere coupling ---
    atm_proc_list = [rad] + conv
    if do_cld_fb:
        atm_proc_list.append(cld_fb)
    if do_rh_proc:
        atm_proc_list.append(rh_proc)
    atm = couple(atm_proc_list, name='Atmosphere')

    # --- diffusion / transport ---
    do_moist = kwargs.get('do_moist', False)
    do_diff = False
    K = None
    Kq = None

    if 'K_diffusion' in kwargs:
        K = kwargs['K_diffusion']
        do_diff = True
    elif 'D_diffusion' in kwargs:
        D = kwargs['D_diffusion']
        K = D / full_state['Ts'].domain.heat_capacity[0] * const.a ** 2
        do_diff = True
    if 'Kq_diffusion' in kwargs:
        Kq = kwargs['Kq_diffusion']
        do_diff = True
    elif 'Dq_diffusion' in kwargs:
        Dq_fit = kwargs['Dq_diffusion']
        Kq = Dq_fit / full_state['Ts'].domain.heat_capacity[0] * const.a ** 2
        do_diff = True

    # Used later by the unified q Limiter even when do_diff=False (no
    # transport). Default 0 means no stratospheric q pin is added by the
    # limiter — caller (fix_q) may still do that explicitly.
    p_min_q_transfer = kwargs.get('p_min_q_transfer', 0.0)

    md = []
    if do_diff:
        do_merid_advection = kwargs.get('do_merid_advection', False)
        if do_merid_advection:
            # NOTE: import here to avoid circular; this is used rarely
            from climate_runs_ext.utils.era5_data import meridional_Kq as _unused_mKq
            Uq = 0.0  # meridional velocity handled by 2D transport
        else:
            Uq = 0.0

        if q_transport_2d:
            Uq = kwargs['Uq_diffusion']
        else:
            # 1D q path: downstream MeridionalAdvectionDiffusion computes
            # Uq.T, so Uq must be an array of matching shape, not scalar 0.
            Uq = np.zeros_like(np.asarray(Kq))
        if mse_transport_2d:
            U = kwargs['U_diffusion']
        else:
            U = 0.0
            K = K.T

        if do_moist:
            md_proc = MoistMeridionalAdvectionDiffusion(
                name='Moist Diffusion', state=atm_state,
                K=K, U=U, Kq=Kq, Uq=Uq,
                geopotential=kwargs.get('geopotential', 0.0),
                do_analytic_gp=kwargs.get('do_analytic_gp', False),
                timestep=timestep_dict.get('Moist Diffusion', long_timestep),
                q_transport_2d=q_transport_2d,
                mse_transport_2d=mse_transport_2d,
            )
            # NB: stratospheric q pinning used to be a nested Limiter under
            # Moist Diffusion. It is now coalesced with the bound_dict q
            # floor into a single top-level 'limiter_q' after couple() —
            # see below. Two adjustment-type Limiters on the same state
            # variable sum their adjustments in climlab and produced
            # incorrect clipping when tendencies pushed q far negative.
            md = [md_proc]
        else:
            mdq = MeridionalAdvectionDiffusion(
                name='q Diffusion',
                state={'q': atm_state['q']},
                K=Kq.T, U=Uq,
                timestep=timestep_dict.get('q Diffusion', long_timestep),
            )
            # stratospheric q pinning coalesced into top-level 'limiter_q'
            # below (see the comment in the `do_moist` branch for details).
            mdt = MeridionalAdvectionDiffusion(
                name='T Diffusion',
                state={'Tatm': atm_state['Tatm']},
                K=K.T, U=0.0,
                timestep=timestep_dict.get('T Diffusion', long_timestep),
            )
            md = [mdq, mdt]

    # --- ocean heat uptake ---
    ohu = []
    add_fixed_ohu = kwargs.get('add_fixed_ohu', False)
    if add_fixed_ohu:
        ohu.append(FixedOceanicHeatUptake(
            oceanic_heat_uptake(lat, months, config),
            name='Fixed OHU',
            state={'Ts': full_state['Ts']},
            timestep=short_timestep,
        ))

    # --- q state carrier (for minimal SARF with transport disabled) ---
    # When `do_moist=False` and no transport subprocesses are built, q is
    # not included in any subprocess's state, so the coupled model's
    # top-level state lacks 'q'. That breaks state-injection from
    # ERA5 (`model.state['q'][:] = ...`) and the fix_q SARF pin. Attach
    # a no-op Limiter (bounds ±inf) whose sole purpose is to register
    # q as part of the top-level state. Only added when there is no
    # other process contributing q — otherwise climlab's adjustment sum
    # on q would include a redundant 0-adjustment.
    q_carrier = []
    if not do_moist and len(md) == 0 and 'q' in full_state:
        q_carrier.append(Limiter(
            name='q_state_carrier',
            state={'q': full_state['q']},
            bounds={'q': {'minimum': -np.inf, 'maximum': np.inf}},
            timestep=short_timestep,
        ))

    # --- couple everything ---
    # Standard climlab couple() does not support bound_dict.
    # We couple first, then add bounds via Limiter subprocesses.
    model = couple(
        [atm] + lsc + md + surface + ohu + q_carrier,
        name='Moist Radiative-Convective',
    )

    # --- unified q Limiter (strat pin + tropo floor in a single process) ---
    # Pin q at stratospheric levels (lev < p_min_q_transfer) to its current
    # (initial) value, AND floor q at a tropospheric minimum (from
    # bound_dict['q'][0] if present). This replaces two separate adjustment-
    # type Limiters (q_stratospheric_pin nested under Moist Diffusion +
    # limiter_q added via add_bounds) which climlab silently SUMS together,
    # producing incorrect clipping when q goes far out of range. One
    # Limiter with per-level bounds behaves deterministically.
    if 'q' in model.state:
        lev = model.state['q'].domain.lev.points
        q_arr = np.array(model.state['q'])
        qmin, qmax = None, np.inf
        if bound_dict and 'q' in bound_dict:
            qmin, qmax = bound_dict['q']
            if qmin is None:
                qmin = -np.inf
            if qmax is None:
                qmax = np.inf

        # Start with the tropo floor/ceiling everywhere
        lower = np.full_like(q_arr, qmin if qmin is not None else -np.inf)
        upper = np.full_like(q_arr, qmax)

        # Overwrite stratospheric levels with the per-cell pin (= initial value)
        strat_ind = np.where(lev < p_min_q_transfer)[0]
        if len(strat_ind) > 0:
            lower[..., strat_ind] = q_arr[..., strat_ind]
            upper[..., strat_ind] = q_arr[..., strat_ind]

        if not (np.all(np.isneginf(lower)) and np.all(np.isposinf(upper))):
            q_lim = Limiter(
                state={'q': model.state['q']},
                bounds={'q': {'minimum': lower, 'maximum': upper}},
                # IMPORTANT: match the model's short timestep. If omitted,
                # climlab defaults the Limiter to 1-day timestep, which
                # causes the Limiter's adjustment to be scaled by
                # model_dt / limiter_dt = 1/24 per step. That's the same
                # "clipping doesn't stick" pathology that motivated this
                # fix in the first place.
                timestep=short_timestep,
            )
            model.add_subprocess('limiter_q', q_lim)

    # --- apply remaining bounds (Tatm, Ts, ...) via add_bounds -----------
    if bound_dict:
        # q has already been handled above; exclude it from add_bounds so
        # we don't create a second adjustment-type Limiter on the same
        # state variable.
        active_bounds = {
            k: v for k, v in bound_dict.items()
            if k in model.state and k != 'q'
        }
        if active_bounds:
            add_bounds(model, active_bounds)

    # --- model attributes (provenance) ---
    model.model_attributes = {
        'optical_table_commit': config.get('optical_table_commit', ''),
        'climate_database_commit': config.get('climate_database_commit', ''),
    }

    return model
