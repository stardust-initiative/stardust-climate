"""Tests for Phase 6d: workflow script, radiation helpers, integration.

All tests use mocks to avoid network access, real RRTMG computation,
and heavy data dependencies.
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import numpy as np
import pytest
from unittest.mock import patch, MagicMock, PropertyMock
from copy import deepcopy

import climlab
from climlab import constants as const


# ---------------------------------------------------------------------------
# radiation_helpers: switch_rad_cld_fb_param
# ---------------------------------------------------------------------------

class TestSwitchRadCldFbParam:
    def test_sets_attribute_on_rad_and_children(self):
        from climate_runs_ext.diagnostics.radiation_helpers import (
            switch_rad_cld_fb_param,
        )
        # Build a simple mock radiation process with SW/LW children
        rad = MagicMock()
        sw = MagicMock()
        lw = MagicMock()
        rad.subprocess = {'SW': sw, 'LW': lw}

        # Give them an initial attribute via __dict__
        rad.__dict__['n_rrtmg_repeat'] = 1
        sw.__dict__['n_rrtmg_repeat'] = 1
        lw.__dict__['n_rrtmg_repeat'] = 1

        switch_rad_cld_fb_param(rad, n_rrtmg_repeat=100)

        assert rad.__dict__['n_rrtmg_repeat'] == 100
        assert sw.__dict__['n_rrtmg_repeat'] == 100
        assert lw.__dict__['n_rrtmg_repeat'] == 100

    def test_skips_missing_attribute(self):
        from climate_runs_ext.diagnostics.radiation_helpers import (
            switch_rad_cld_fb_param,
        )
        rad = MagicMock()
        sw = MagicMock()
        lw = MagicMock()
        rad.subprocess = {'SW': sw, 'LW': lw}

        # Only rad has the attribute, children don't
        rad.__dict__['custom_param'] = 5
        sw.__dict__.clear()
        lw.__dict__.clear()

        switch_rad_cld_fb_param(rad, custom_param=10)

        assert rad.__dict__['custom_param'] == 10
        assert 'custom_param' not in sw.__dict__
        assert 'custom_param' not in lw.__dict__


# ---------------------------------------------------------------------------
# radiation_helpers: get_inst_diag
# ---------------------------------------------------------------------------

class TestGetInstDiag:
    def test_returns_difference_dataset(self):
        from climate_runs_ext.diagnostics.radiation_helpers import get_inst_diag
        import xarray as xr

        nlat = 5
        lat = np.linspace(-80, 80, nlat)

        # Build a mock radiation process that supports deepcopy and
        # compute_diagnostics / to_xarray
        def _make_rad(asr_val, olr_val):
            """Create a simple mock radiation process."""
            rad = MagicMock()
            rad.subprocess = {'SW': MagicMock(), 'LW': MagicMock()}

            # Cloud fields
            for attr in ['cldfrac', 'clwp', 'ciwp', 'r_ice', 'r_liq']:
                rad.__dict__[attr] = np.ones(nlat) * 0.3

            # State fields
            for attr in ['specific_humidity', 'Ts', 'Tatm']:
                rad.__dict__[attr] = np.ones(nlat) * 300.0

            def _to_xarray(**kwargs):
                return xr.Dataset({
                    'ASR': xr.DataArray(asr_val * np.ones(nlat), dims=['lat']),
                    'OLR': xr.DataArray(olr_val * np.ones(nlat), dims=['lat']),
                })
            rad.to_xarray = _to_xarray
            rad.compute_diagnostics = MagicMock()
            return rad

        rad_perturbed = _make_rad(asr_val=250.0, olr_val=235.0)
        rad_ref = _make_rad(asr_val=240.0, olr_val=240.0)

        diag = get_inst_diag(rad_perturbed, rad_ref, do_copy_ref=True)

        # ASR difference: 250 - 240 = 10
        np.testing.assert_allclose(diag['ASR'].values, 10.0)
        # OLR difference: 235 - 240 = -5
        np.testing.assert_allclose(diag['OLR'].values, -5.0)
        # RF = ASR - OLR = 10 - (-5) = 15
        rf = diag['ASR'] - diag['OLR']
        np.testing.assert_allclose(rf.values, 15.0)

    def test_with_param_change(self):
        from climate_runs_ext.diagnostics.radiation_helpers import get_inst_diag
        import xarray as xr

        nlat = 3

        def _make_rad():
            rad = MagicMock()
            sw = MagicMock()
            lw = MagicMock()
            rad.subprocess = {'SW': sw, 'LW': lw}
            rad.__dict__['n_rrtmg_repeat'] = 1
            sw.__dict__['n_rrtmg_repeat'] = 1
            lw.__dict__['n_rrtmg_repeat'] = 1

            for attr in ['cldfrac', 'clwp', 'ciwp', 'r_ice', 'r_liq',
                          'specific_humidity', 'Ts', 'Tatm']:
                rad.__dict__[attr] = np.ones(nlat) * 0.5

            def _to_xarray(**kwargs):
                return xr.Dataset({
                    'ASR': xr.DataArray(np.ones(nlat) * 100.0, dims=['lat']),
                    'OLR': xr.DataArray(np.ones(nlat) * 100.0, dims=['lat']),
                })
            rad.to_xarray = _to_xarray
            rad.compute_diagnostics = MagicMock()
            return rad

        rad0 = _make_rad()
        rad_ref0 = _make_rad()

        diag = get_inst_diag(
            rad0, rad_ref0,
            rad_param_change_dict={'n_rrtmg_repeat': 50},
        )
        # Result is difference of identical processes → zero
        np.testing.assert_allclose(diag['ASR'].values, 0.0)


# ---------------------------------------------------------------------------
# Workflow: _run_integration helper
# ---------------------------------------------------------------------------

class TestRunIntegration:
    def test_run_integration_prints_and_saves(self, tmp_path):
        """Verify _run_integration calls integrate_days the right number
        of times and saves to the specified path."""
        from examples.climatic_response_annual_avg import _run_integration
        import xarray as xr

        nlat = 3
        lat = np.linspace(-60, 60, nlat)

        # Build a minimal mock model
        model = MagicMock()
        model.timeave = {
            'ASR': np.ones((nlat, 1)) * 240.0,
            'OLR': np.ones((nlat, 1)) * 240.0,
        }

        def _to_xarray(**kwargs):
            return xr.Dataset({
                'ASR': xr.DataArray(np.ones((nlat, 1)) * 240.0),
                'OLR': xr.DataArray(np.ones((nlat, 1)) * 240.0),
            })
        model.to_xarray = _to_xarray

        save_path = str(tmp_path / 'test_output.nc')

        _run_integration(
            model, lat,
            n_cycle=2, t_cycle_days=10.0, t_avg_days=30.0,
            save_path=save_path, label='[test]',
        )

        # Should have called integrate_days 3 times (2 cycles + 1 avg)
        assert model.integrate_days.call_count == 3
        assert model.compute_diagnostics.call_count == 3


# ---------------------------------------------------------------------------
# Workflow: main function smoke test (everything mocked)
# ---------------------------------------------------------------------------

class TestWorkflowMain:
    @patch('examples.climatic_response_annual_avg.get_ref')
    @patch('examples.climatic_response_annual_avg.load_project_config')
    def test_main_do_just_ref_no_ref_calc(self, mock_config, mock_get_ref,
                                          tmp_path):
        """When do_ref_calc=False and do_just_ref=True, main() should load
        a pre-existing ref file and return quickly."""
        import xarray as xr
        from examples.climatic_response_annual_avg import main

        nlat, nlev = 3, 5
        lat = np.linspace(-60, 60, nlat)
        lev = np.linspace(200, 1000, nlev)

        # Create a mock config
        mock_config.return_value = {
            'climate_database_files_http': 'http://fake',
            'climate_database_token': 'fake',
            'proj_name': 'test',
            'optical_table_commit': '', 'climate_database_commit': '',
            'aerosols_input_dict': {},
        }

        # Build a mock model
        full_state = climlab.column_state(lev=lev, lat=lat)
        mock_model = MagicMock()
        mock_model.state = {'Ts': full_state['Ts'], 'Tatm': full_state['Tatm']}
        ts_mock = MagicMock()
        ts_mock.domain.lat.points = lat
        mock_model.Ts = ts_mock
        mock_model.timeave = {
            'Ts': np.array(full_state['Ts']),
            'Tatm': np.array(full_state['Tatm']),
        }
        mock_get_ref.return_value = mock_model

        # Create a dummy ref file so the workflow can "load" it.
        # Ts shape must be (nlat, 1) to match climlab field shape.
        base = str(tmp_path)
        ds = xr.Dataset({
            'Ts': xr.DataArray(np.ones((nlat, 1)) * 290.0, dims=['lat', 'x']),
            'Tatm': xr.DataArray(np.ones((nlat, nlev)) * 250.0, dims=['lat', 'lev']),
            'ASR': xr.DataArray(np.ones((nlat, 1)) * 240.0, dims=['lat', 'x2']),
            'OLR': xr.DataArray(np.ones((nlat, 1)) * 240.0, dims=['lat', 'x3']),
        })
        ds.to_netcdf(str(tmp_path / 'model_ref.nc'))

        result = main([
            '-do_just_ref', 'True',
            '-base_folder', base,
        ])

        assert result is mock_model
        mock_get_ref.assert_called_once()


# ---------------------------------------------------------------------------
# Workflow: main with ref calc (mocked integration)
# ---------------------------------------------------------------------------

class TestWorkflowRefCalc:
    @patch('examples.climatic_response_annual_avg.get_ref')
    @patch('examples.climatic_response_annual_avg.load_project_config')
    def test_main_runs_ref_calc(self, mock_config, mock_get_ref, tmp_path):
        """When do_ref_calc=True and do_just_ref=True, the integration loop
        should run and save to disk."""
        import xarray as xr
        from examples.climatic_response_annual_avg import main

        nlat, nlev = 3, 5
        lat = np.linspace(-60, 60, nlat)
        lev = np.linspace(200, 1000, nlev)

        mock_config.return_value = {
            'climate_database_files_http': 'http://fake',
            'climate_database_token': 'fake',
            'proj_name': 'test',
            'optical_table_commit': '', 'climate_database_commit': '',
            'aerosols_input_dict': {},
        }

        # Build a mock model that supports integration
        full_state = climlab.column_state(lev=lev, lat=lat)
        mock_model = MagicMock()
        mock_model.state = {'Ts': full_state['Ts'], 'Tatm': full_state['Tatm']}
        ts_mock = MagicMock()
        ts_mock.domain.lat.points = lat
        mock_model.Ts = ts_mock
        mock_model.timeave = {
            'Ts': np.array(full_state['Ts']),
            'Tatm': np.array(full_state['Tatm']),
            'ASR': np.ones((nlat, 1)) * 240.0,
            'OLR': np.ones((nlat, 1)) * 240.0,
        }
        mock_model.absorber_vmr = {'CO2': 420e-6}

        def _to_xarray(**kwargs):
            # Ts must be (nlat, 1) to match climlab field shape
            return xr.Dataset({
                'Ts': xr.DataArray(np.ones((nlat, 1)) * 290.0,
                                   dims=['lat', 'x0']),
                'Tatm': xr.DataArray(np.ones((nlat, nlev)) * 250.0,
                                     dims=['lat', 'lev']),
                'ASR': xr.DataArray(np.ones((nlat, 1)) * 240.0,
                                    dims=['lat', 'x']),
                'OLR': xr.DataArray(np.ones((nlat, 1)) * 240.0,
                                    dims=['lat', 'x2']),
            })
        mock_model.to_xarray = _to_xarray

        mock_get_ref.return_value = mock_model

        base = str(tmp_path)

        result = main([
            '-do_ref_calc', 'True',
            '-do_just_ref', 'True',
            '-n_cycle', '2',
            '-t_cycle_days', '5',
            '-t_avg_days', '10',
            '-base_folder', base,
        ])

        assert result is mock_model
        # Integration should have been called (2 cycles + 1 avg = 3)
        assert mock_model.integrate_days.call_count == 3


# ---------------------------------------------------------------------------
# Smooth function test (used in workflow plotting)
# ---------------------------------------------------------------------------

class TestSmoothFunction:
    def test_smooth_preserves_shape(self):
        from climate_runs_ext.utils.era5_data import Smooth
        x = np.sin(np.linspace(0, 4 * np.pi, 100))
        result = Smooth(x, window_length=7)
        assert result.shape == x.shape

    def test_smooth_reduces_noise(self):
        from climate_runs_ext.utils.era5_data import Smooth
        np.random.seed(42)
        signal = np.sin(np.linspace(0, 2 * np.pi, 200))
        noisy = signal + 0.5 * np.random.randn(200)
        smoothed = Smooth(noisy, window_length=11)
        # Smoothed should be closer to true signal than noisy
        err_noisy = np.mean((noisy - signal)**2)
        err_smooth = np.mean((smoothed - signal)**2)
        assert err_smooth < err_noisy


# ---------------------------------------------------------------------------
# lat_avg (used in workflow for energy balance printing)
# ---------------------------------------------------------------------------

class TestLatAvgWorkflow:
    def test_energy_balance_zero(self):
        """If ASR == OLR, the lat_avg energy balance should be zero."""
        from climate_runs_ext.utils.era5_data import lat_avg
        nlat = 10
        lat = np.linspace(-80, 80, nlat)
        asr = 240.0 * np.ones(nlat)
        olr = 240.0 * np.ones(nlat)
        eb = lat_avg(asr - olr, lat)
        np.testing.assert_allclose(eb, 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# SeasonTypes (used in workflow for season selection)
# ---------------------------------------------------------------------------

class TestSeasonTypesWorkflow:
    def test_all_seasons_present(self):
        from climate_runs_ext.utils.era5_data import SeasonTypes
        expected = ['Annual', 'DJF', 'MAM', 'JJA', 'SON']
        for s in expected:
            assert s in SeasonTypes.months_dict

    def test_single_months(self):
        from climate_runs_ext.utils.era5_data import SeasonTypes
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        for i, m in enumerate(months):
            assert SeasonTypes.months_dict[m] == [i]
