"""Lean climate response workflow — annual-average experiments.

This script drives the full experiment pipeline:

1. Build (or load) a reference climate model via ``get_ref()``.
2. Spin-up the reference with alternating short integration cycles.
3. (Optional) Build perturbed models (aerosol layer, fxCO2) from the
   converged reference state.
4. Compute instantaneous radiative forcing, ERF (fixed Ts), and
   steady-state response.
5. (Optional) Compute the stabilisation problem (optimal aerosol burden
   to offset a CO2 increase).

Usage
-----
::

    python -m examples.climatic_response_annual_avg \\
        -do_ref_calc True -do_fxco2_inst_rf True -n_cycle 36

Requires a valid ``config.json`` in the repository root (or pass
``-config_path``).
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import sys
import os
import argparse
import numpy as np
import xarray as xr
from copy import deepcopy

from climlab.utils import constants as const

from climate_runs_ext import load_project_config
from climate_runs_ext.reference_model import get_ref
from climate_runs_ext.utils.era5_data import (
    SeasonTypes, lat_avg, Smooth, era5_annual_initial_state,
)
from climate_runs_ext.utils.state_io import (
    update_model_from_xr,
    iteratively_update_internal,
)
from climate_runs_ext.utils.model_control import (
    fix_Ts,
    unfix_Ts,
    fix_q,
    unfix_q,
    fix_Tatm_trop,
    unfix_Tatm_trop,
)
from climate_runs_ext.diagnostics.radiation_helpers import get_inst_diag


# ---------------------------------------------------------------------------
# Helper: integration loop
# ---------------------------------------------------------------------------

def _run_integration(model, lat, n_cycle, t_cycle_days, t_avg_days,
                     save_path=None, label=''):
    """Run *n_cycle* short integrations + 1 long averaging integration.

    At each step the energy imbalance is printed.  If *save_path* is given,
    the model is serialised to NetCDF after every step (atomically via a
    temp file).

    Parameters
    ----------
    model : climlab Process
    lat : array-like   Latitude vector for energy-balance printing.
    n_cycle : int       Number of short spin-up cycles.
    t_cycle_days : float  Duration of each short cycle (days).
    t_avg_days : float    Duration of the final averaging period (days).
    save_path : str or None   If set, write NetCDF here.
    label : str         Tag for print messages.
    """
    save_temp = save_path.replace('.nc', '_temp.nc') if save_path else None

    for k in range(n_cycle + 1):
        t = t_cycle_days if k < n_cycle else t_avg_days
        model.integrate_days(t + 1e-9)
        model.compute_diagnostics()
        asr = model.timeave['ASR']
        olr = model.timeave['OLR']
        ohu = model.timeave.get('ohu', 0.0)
        eb = lat_avg((asr - olr - ohu)[:, 0], lat)
        print(f'{label} k={k}, EB={eb:.4f}')

        if save_path:
            m_xr = model.to_xarray(
                diagnostics=True, timeave=True,
            )
            m_xr.to_netcdf(save_temp)
            m_xr.close()

    if save_path and save_temp and os.path.exists(save_temp):
        os.replace(save_temp, save_path)


def _inject_era5_initial_state(model, season_str, config):
    """Overwrite model.state Tatm/Ts/q with ERA5 annual-mean values.

    Used in sarf mode so the Limiter snapshot is captured on ERA5 values
    rather than on climlab's idealized column_state defaults.
    """
    months = SeasonTypes.months_dict[season_str]
    era5_init = era5_annual_initial_state(
        model.state['Tatm'].domain, months, config,
    )
    model.state['Tatm'][:] = era5_init['Tatm']
    model.state['Ts'][:]   = era5_init['Ts']
    model.state['q'][:]    = era5_init['q']
    # Propagate to subprocesses (surface fluxes, RH diagnostics, etc.)
    iteratively_update_internal(model, model.state)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Annual-average climatic response experiments.',
    )

    # --- run flags --------------------------------------------------------
    parser.add_argument('-do_ref_calc', type=str, default='False')
    parser.add_argument('-do_layer_inst_rf', type=str, default='False')
    parser.add_argument('-do_layer_erf', type=str, default='False')
    parser.add_argument('-do_layer_steady', type=str, default='False')
    parser.add_argument('-do_fxco2_inst_rf', type=str, default='False')
    parser.add_argument('-do_fxco2_erf', type=str, default='False')
    parser.add_argument('-do_fxco2_steady', type=str, default='False')
    parser.add_argument('-do_stabilization_problem', type=str, default='False')
    parser.add_argument('-do_just_ref', type=str, default='False')

    # --- aerosol layer params ---------------------------------------------
    parser.add_argument('-rho', type=float, default=2196.0,
                        help='Particle density [kg/m^3]')
    parser.add_argument('-r_m', type=float, default=250e-9,
                        help='Particle radius [m]')
    parser.add_argument('-material_name', type=str, default='silica')
    parser.add_argument('-M_tot_Tg', type=float, default=10.0,
                        help='Total layer mass [Tg]')
    parser.add_argument('-p_min_layer', type=float, default=20.0,
                        help='Layer top pressure [hPa]')
    parser.add_argument('-p_max_layer', type=float, default=80.0,
                        help='Layer bottom pressure [hPa]')

    # --- file-based layer (transport-model output) -----------------------
    parser.add_argument(
        '-layer_state_file', type=str, default='',
        help=(
            'Path to a netCDF transport-model state file with per-bin '
            'mmr(lat, lev) variables. If set, overrides the uniform '
            'layer built from -M_tot_Tg / -p_min_layer / -p_max_layer '
            '(those flags are ignored). Must be used together with '
            '-radius_mapping_file.'
        ),
    )
    parser.add_argument(
        '-radius_mapping_file', type=str, default='',
        help=(
            'Path to a radius_mapping.npz file mapping each bin variable '
            "name in -layer_state_file to its particle radius in metres."
        ),
    )

    # --- integration params -----------------------------------------------
    parser.add_argument('-n_cycle', type=int, default=36)
    parser.add_argument('-t_cycle_days', type=float, default=30.0)
    parser.add_argument('-t_avg_days', type=float, default=365.0)
    parser.add_argument('-fxco2', type=float, default=2.0,
                        help='CO2 multiplier for perturbation')
    parser.add_argument('-n_rrtmg_repeat', type=int, default=100)
    parser.add_argument('-season', type=str, default='Annual')

    # --- ERF mode ---------------------------------------------------------
    parser.add_argument(
        '-erf_mode', type=str, default='full', choices=['full', 'sarf'],
        help=(
            "ERF calculation mode. 'full' = standard fixed-SST ERF "
            "(all adjustments active except Ts). 'sarf' = stratosphere-"
            "adjusted RF: fix Ts, q, and tropospheric Tatm from the "
            "surface up to p_trop_hPa; clouds held static at ERA5 annual "
            "mean; only stratospheric Tatm is free to relax."
        ),
    )
    parser.add_argument(
        '-p_trop_hPa', type=float, default=180.0,
        help='SARF tropopause pressure (nearest level selected)',
    )

    # --- paths / config ---------------------------------------------------
    parser.add_argument('-config_path', type=str, default=None,
                        help='Path to config.json')
    parser.add_argument('-base_folder', type=str, default='',
                        help='Output directory')

    args = parser.parse_args(argv)

    # --- parse boolean flags ----------------------------------------------
    def _tobool(s):
        return s.lower() == 'true'

    do_ref_calc = _tobool(args.do_ref_calc)
    do_layer_inst_rf = _tobool(args.do_layer_inst_rf)
    do_layer_erf = _tobool(args.do_layer_erf)
    do_layer_steady = _tobool(args.do_layer_steady)
    do_fxco2_inst_rf = _tobool(args.do_fxco2_inst_rf)
    do_fxco2_erf = _tobool(args.do_fxco2_erf)
    do_fxco2_steady = _tobool(args.do_fxco2_steady)
    do_stabilization_problem = _tobool(args.do_stabilization_problem)
    do_just_ref = _tobool(args.do_just_ref)

    erf_mode = args.erf_mode
    p_trop_hPa = args.p_trop_hPa
    is_sarf = (erf_mode == 'sarf')

    season_str = args.season
    n_cycle = args.n_cycle
    t_cycle_days = args.t_cycle_days
    t_avg_days = args.t_avg_days
    fxco2 = args.fxco2
    n_rrtmg_repeat = args.n_rrtmg_repeat

    # --- load project config ----------------------------------------------
    config = load_project_config(args.config_path)

    # --- output directory -------------------------------------------------
    base_folder = args.base_folder
    if not base_folder:
        base_folder = os.path.join('.', 'output', 'climatic_response')
    os.makedirs(base_folder, exist_ok=True)

    # --- build reference model --------------------------------------------
    kwargs = {'season_str': season_str}
    model_ref = get_ref(config, **kwargs)

    filepath_ref = os.path.join(base_folder, 'model_ref.nc')
    if os.path.exists(filepath_ref):
        print('Loading reference state from file')
        model_ref_xr = xr.open_dataset(filepath_ref).load()
        update_model_from_xr(model_ref, model_ref_xr, do_compute=True)

    lat = model_ref.Ts.domain.lat.points

    # --- SARF setup on reference model ------------------------------------
    # In sarf mode, overwrite the idealized climlab initial profile with
    # ERA5 annual means (unless a prior sarf run's state was already loaded
    # from filepath_ref), then pin Ts, q, and tropospheric Tatm.  The spin-
    # up then lets only stratospheric Tatm relax radiatively.
    if is_sarf:
        if not os.path.exists(filepath_ref):
            print('[sarf] injecting ERA5 annual-mean initial state into ref')
            _inject_era5_initial_state(model_ref, season_str, config)
        print(
            f'[sarf] fixing Ts, q, Tatm below p_trop={p_trop_hPa:g} hPa '
            f'on reference model'
        )
        fix_Ts(model_ref)
        fix_q(model_ref)
        fix_Tatm_trop(model_ref, p_trop_hPa=p_trop_hPa)

    # --- spin-up reference ------------------------------------------------
    if do_ref_calc:
        _run_integration(
            model_ref, lat, n_cycle, t_cycle_days, t_avg_days,
            save_path=filepath_ref, label='[ref]',
        )

    # Reload converged reference
    model_ref_xr = xr.open_dataset(filepath_ref).load()
    update_model_from_xr(model_ref, model_ref_xr, do_compute=True)
    ref_state = {k: model_ref.timeave[k] for k in model_ref.state.keys()}

    if do_just_ref:
        print('do_just_ref=True — stopping after reference calculation.')
        return model_ref

    # --- build perturbed models -------------------------------------------

    # Aerosol layer model
    from climlab_stardust_extension.radiation.optical_depth_tables_aerosols import (
        aerosol_instance,
        construct_uni_layer_vmr_p_based,
        AerosolsOptDepTables,
        get_radiation_with_aerosols_params,
    )

    rho = args.rho
    r_m = args.r_m
    material_name = args.material_name
    M_tot_Tg = args.M_tot_Tg
    p_min_layer_approx = args.p_min_layer
    p_max_layer_approx = args.p_max_layer
    layer_state_file = args.layer_state_file
    radius_mapping_file = args.radius_mapping_file

    rad = model_ref.subprocess['Atmosphere'].subprocess['Radiation']

    if layer_state_file and radius_mapping_file:
        # --- file-based multi-bin layer from transport-model output -----
        from climate_runs_ext.utils.aerosol_layer import (
            load_multi_bin_aerosol_state,
        )
        print(
            f'[layer] loading file-based multi-bin state from '
            f'{layer_state_file}'
        )
        loaded = load_multi_bin_aerosol_state(
            state_path=layer_state_file,
            radius_mapping_path=radius_mapping_file,
            material_name=material_name,
            rho_particle=rho,
            domain=rad.Tatm.domain,
        )
        print(
            f'[layer]   bins={len(loaded.bin_names)} '
            f'total_mass={loaded.total_mass_Tg:.3f} Tg '
            f'avg_D={loaded.avg_diameter_m * 1e9:.1f} nm'
        )
        aerosols_table_obj = AerosolsOptDepTables(
            aerosol_instance_list=loaded.aerosol_instance_list,
            domain=rad.Tatm.domain,
            coszen=rad.coszen,
            **config['aerosols_input_dict'],
        )
    elif layer_state_file or radius_mapping_file:
        raise ValueError(
            '-layer_state_file and -radius_mapping_file must be provided '
            'together (or neither, to use the uniform-layer path).'
        )
    else:
        # --- uniform-layer fallback (CLI-defined M_tot_Tg + p_min/max) --
        density_profile_kg_m2 = (
            1e9 * M_tot_Tg / (4 * np.pi * const.a**2) * np.ones_like(lat)
        )
        lev_bounds = model_ref.Tatm.domain.lev.bounds
        imin = np.argmin(np.abs(lev_bounds - p_min_layer_approx))
        imax = np.argmin(np.abs(lev_bounds - p_max_layer_approx))
        p_min_layer = lev_bounds[imin]
        p_max_layer = lev_bounds[imax]

        vmr = construct_uni_layer_vmr_p_based(
            rho, density_profile_kg_m2, p_min_layer, p_max_layer, r_m,
            rad.state,
        )
        aerosols_table_obj = AerosolsOptDepTables(
            aerosol_instance_list=[aerosol_instance(material_name, r_m, vmr)],
            domain=rad.Tatm.domain,
            coszen=rad.coszen,
            **config['aerosols_input_dict'],
        )
    rad_with_aero_param_dict = get_radiation_with_aerosols_params(
        rad.state, aerosols_table_obj, rad.coszen,
    )

    # In sarf mode we disable cloud feedback so clouds stay static at the
    # ERA5 annual-mean values baked into the model by get_ref().
    do_cld_fb_perturbed = not is_sarf

    model_layer = get_ref(
        config,
        rad_with_aero_param_dict=rad_with_aero_param_dict,
        do_cld_fb=do_cld_fb_perturbed,
        **kwargs,
    )
    if not is_sarf:
        # NB: order matters when do_cld_fb=True. compute_diagnostics must
        # fire AFTER iteratively_update_internal so CloudFeedback's
        # internal reference (T0/q0/rh0) is aligned with the loaded
        # state before its polynomial is first evaluated. Calling
        # do_compute=True here would run _compute against a stale
        # climlab-default T0 and cause a spurious cloud jump that biases
        # the first-step radiation (see fix/cloud-feedback-load-order).
        update_model_from_xr(model_layer, model_ref_xr, do_compute=False)
        iteratively_update_internal(model_layer, ref_state)
        model_layer.compute_diagnostics()

    # fxCO2 model
    model_fxco2 = get_ref(config, do_cld_fb=do_cld_fb_perturbed, **kwargs)
    model_fxco2.absorber_vmr['CO2'] = fxco2 * model_ref.absorber_vmr['CO2']
    if not is_sarf:
        # Same ordering constraint as model_layer: update_internal must
        # run before compute_diagnostics when cloud feedback is active.
        update_model_from_xr(model_fxco2, model_ref_xr, do_compute=False)
        iteratively_update_internal(model_fxco2, ref_state)
        model_fxco2.compute_diagnostics()

    # --- SARF setup on perturbed models -----------------------------------
    # Inject ERA5 annual-mean state and then pin Ts/q/tropospheric Tatm so
    # the clamped tropospheric columns match the reference model exactly.
    if is_sarf:
        print('[sarf] injecting ERA5 annual-mean initial state into perturbed models')
        _inject_era5_initial_state(model_layer, season_str, config)
        _inject_era5_initial_state(model_fxco2, season_str, config)
        print(
            f'[sarf] fixing Ts, q, Tatm below p_trop={p_trop_hPa:g} hPa '
            f'on perturbed models'
        )
        for m in (model_layer, model_fxco2):
            fix_Ts(m)
            fix_q(m)
            fix_Tatm_trop(m, p_trop_hPa=p_trop_hPa)

    rad_ref = model_ref.subprocess['Atmosphere'].subprocess['Radiation']

    # --- instantaneous radiative forcing ----------------------------------
    if do_layer_inst_rf:
        print('Calculating inst RF of aerosol layer')
        rad_layer_inst = deepcopy(
            model_layer.subprocess['Atmosphere'].subprocess['Radiation'],
        )
        diag = get_inst_diag(
            rad_layer_inst, rad_ref,
            rad_param_change_dict={'n_rrtmg_repeat': n_rrtmg_repeat},
            do_copy_ref=True,
        )
        rf_inst_layer = diag['ASR'] - diag['OLR']
        np.savez(
            os.path.join(base_folder, 'rf_inst_layer.npz'),
            lat=lat, rf_inst=rf_inst_layer,
        )
    else:
        path = os.path.join(base_folder, 'rf_inst_layer.npz')
        if os.path.exists(path):
            temp = np.load(path)
            rf_inst_layer = temp['rf_inst']
        else:
            rf_inst_layer = None

    if do_fxco2_inst_rf:
        print('Calculating inst RF of fxCO2')
        rad_fxco2_inst = deepcopy(
            model_fxco2.subprocess['Atmosphere'].subprocess['Radiation'],
        )
        diag = get_inst_diag(
            rad_fxco2_inst, rad_ref,
            rad_param_change_dict={'n_rrtmg_repeat': n_rrtmg_repeat},
            do_copy_ref=True,
        )
        rf_fxco2_inst = diag['ASR'] - diag['OLR']
        np.savez(
            os.path.join(base_folder, 'rf_inst_fxco2.npz'),
            lat=lat, rf_inst=rf_fxco2_inst,
        )
    else:
        path = os.path.join(base_folder, 'rf_inst_fxco2.npz')
        if os.path.exists(path):
            temp = np.load(path)
            rf_fxco2_inst = temp['rf_inst']
        else:
            rf_fxco2_inst = None

    # --- ERF (fixed Ts) ---------------------------------------------------
    if do_layer_erf:
        print('Calculating ERF of aerosol layer (fixed Ts)')
        fix_Ts(model_layer)
        _run_integration(
            model_layer, lat, n_cycle, t_cycle_days, t_avg_days,
            save_path=os.path.join(base_folder, 'model_layer_erf.nc'),
            label='[layer ERF]',
        )

    if do_fxco2_erf:
        print('Calculating ERF of fxCO2 (fixed Ts)')
        fix_Ts(model_fxco2)
        _run_integration(
            model_fxco2, lat, n_cycle, t_cycle_days, t_avg_days,
            save_path=os.path.join(base_folder, 'model_fxco2_erf.nc'),
            label='[fxCO2 ERF]',
        )

    # --- steady-state response --------------------------------------------
    if is_sarf and (do_layer_steady or do_fxco2_steady or do_stabilization_problem):
        print(
            '[sarf] skipping steady-state / stabilization branches: these '
            'are not meaningful with fixed tropospheric temperature + q.'
        )
        do_layer_steady = False
        do_fxco2_steady = False
        do_stabilization_problem = False

    if do_layer_steady:
        print('Calculating steady-state response of aerosol layer')
        unfix_Ts(model_layer)
        _run_integration(
            model_layer, lat, n_cycle, t_cycle_days, t_avg_days,
            save_path=os.path.join(base_folder, 'model_layer.nc'),
            label='[layer steady]',
        )

    if do_fxco2_steady:
        print('Calculating steady-state response of fxCO2')
        unfix_Ts(model_fxco2)
        _run_integration(
            model_fxco2, lat, n_cycle, t_cycle_days, t_avg_days,
            save_path=os.path.join(base_folder, 'model_fxco2.nc'),
            label='[fxCO2 steady]',
        )

    # --- stabilisation problem --------------------------------------------
    if do_stabilization_problem:
        print('Computing stabilisation problem')
        model_layer_erf_xr = xr.open_dataset(
            os.path.join(base_folder, 'model_layer_erf.nc'),
        ).load()
        model_fxco2_erf_xr = xr.open_dataset(
            os.path.join(base_folder, 'model_fxco2_erf.nc'),
        ).load()
        erf_layer = (
            (model_layer_erf_xr['ASR'] - model_layer_erf_xr['OLR'])
            - (model_ref_xr['ASR'] - model_ref_xr['OLR'])
        )
        erf_fxco2 = (
            (model_fxco2_erf_xr['ASR'] - model_fxco2_erf_xr['OLR'])
            - (model_ref_xr['ASR'] - model_ref_xr['OLR'])
        )
        erf_layer = xr.where(erf_layer < 0.0, erf_layer, 0.0)
        erf_layer_min = 0.05

        density_profile_kg_m2_stab = (
            -(erf_fxco2[:, 0] * erf_layer[:, 0])
            / (erf_layer[:, 0]**2 + erf_layer_min**2)
            * density_profile_kg_m2
        )
        density_profile_kg_m2_stab = np.where(
            density_profile_kg_m2_stab > 0.0, density_profile_kg_m2_stab, 0.0,
        )

        # Build stabilisation model
        vmr_stab = construct_uni_layer_vmr_p_based(
            rho, density_profile_kg_m2_stab,
            p_min_layer, p_max_layer, r_m, rad.state,
        )
        aerosols_table_obj_stab = AerosolsOptDepTables(
            aerosol_instance_list=[
                aerosol_instance(material_name, r_m, vmr_stab),
            ],
            domain=rad.Tatm.domain,
            coszen=rad.coszen,
            **config['aerosols_input_dict'],
        )
        rad_with_aero_param_dict_stab = get_radiation_with_aerosols_params(
            rad.state, aerosols_table_obj_stab, rad.coszen,
        )

        model_fxco2_layer = get_ref(
            config,
            rad_with_aero_param_dict=rad_with_aero_param_dict_stab,
            do_cld_fb=True,
            **kwargs,
        )
        model_fxco2_layer.absorber_vmr['CO2'] = (
            fxco2 * model_ref.absorber_vmr['CO2']
        )
        update_model_from_xr(model_fxco2_layer, model_ref_xr, do_compute=True)
        iteratively_update_internal(model_fxco2_layer, ref_state)

        _run_integration(
            model_fxco2_layer, lat, n_cycle, t_cycle_days, t_avg_days,
            save_path=os.path.join(base_folder, 'model_fxco2_layer.nc'),
            label='[stab]',
        )

        np.savez(
            os.path.join(base_folder, 'burden_optimal.npz'),
            lat=lat,
            density_profile_kg_m2=density_profile_kg_m2_stab,
        )

    print(f'Done. Results in {base_folder}')
    return model_ref


if __name__ == '__main__':
    main()
