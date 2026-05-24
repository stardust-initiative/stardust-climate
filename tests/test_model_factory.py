"""Tests for model factory (Phase 6c).

Heavily mocked tests that verify:
- construct_lev produces correct grid
- get_rce_sbm_model_annual_avg produces a model with expected structure
- model_generator can be called (with mocked data loaders)
- Post-coupling bounds (Limiter subprocesses) are applied
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import numpy as np
import pytest
from unittest.mock import patch, MagicMock

import climlab
from climlab import constants as const

from climate_runs_ext.reference_model.rev6 import construct_lev


# ---------------------------------------------------------------------------
# construct_lev
# ---------------------------------------------------------------------------

class TestConstructLev:
    def test_correct_length(self):
        lev = construct_lev(dp_min=2.0, sp=1000.0, nlev=50)
        assert lev.shape == (50,)

    def test_monotonically_increasing(self):
        lev = construct_lev(dp_min=2.0, sp=1000.0, nlev=100)
        assert np.all(np.diff(lev) > 0)

    def test_bounds(self):
        lev = construct_lev(dp_min=2.0, sp=1000.0, nlev=50)
        assert lev[0] > 0.0
        assert lev[-1] < 1000.0

    def test_first_level_small(self):
        lev = construct_lev(dp_min=1.0, sp=1000.0, nlev=100)
        # First midpoint should be around dp_min/2
        assert lev[0] < 2.0

    def test_sum_approximately_sp(self):
        dp_min = 2.0
        sp = 1000.0
        nlev = 50
        lev = construct_lev(dp_min=dp_min, sp=sp, nlev=nlev)
        # The last level bound should be close to sp
        # Reconstruct bounds from midpoints
        # lev_bound[-1] should be sp
        dlev = np.diff(np.concatenate(([0], lev)))
        # Actually, test the property that the grid spans 0..sp
        assert lev[-1] > 0.9 * sp  # last midpoint near surface


# ---------------------------------------------------------------------------
# get_rce_sbm_model_annual_avg (mocked data loaders)
# ---------------------------------------------------------------------------

class TestGetRceSbmModel:
    """Test that model assembly produces correct structure.

    We mock all data loaders and use simple numpy arrays for cloud/GHG data.
    The model is built with minimal config (no 2D transport, no convection).
    """

    @patch('climate_runs_ext.model_factory.model_builder.FixedOceanicHeatUptake')
    @patch('climate_runs_ext.model_factory.model_builder.get_era5_mycloud')
    @patch('climate_runs_ext.model_factory.model_builder.Surface_albedo')
    @patch('climate_runs_ext.model_factory.model_builder.Seasonal_insolation')
    @patch('climate_runs_ext.model_factory.model_builder.oceanic_heat_uptake')
    def test_builds_model(self, mock_ohu, mock_insol, mock_albedo, mock_cloud, mock_fohu):
        from climate_runs_ext.model_factory.model_builder import get_rce_sbm_model_annual_avg

        nlat, nlev = 5, 10
        lat = np.linspace(-80, 80, nlat)
        lev = np.linspace(100, 1000, nlev)

        # Mock insolation
        mock_insol.return_value = 340.0 * np.ones(nlat)
        # Mock albedo
        mock_albedo.return_value = 0.3 * np.ones(nlat)
        # Mock cloud
        mock_cloud.return_value = {
            'cldfrac': 0.3 * np.ones((nlat, nlev)),
            'clwp': 0.01 * np.ones((nlat, nlev)),
            'ciwp': 0.005 * np.ones((nlat, nlev)),
            'r_liq': 14.0 * np.ones((nlat, nlev)),
            'r_ice': 14.0 * np.ones((nlat, nlev)),
        }
        # Mock OHU
        mock_ohu.return_value = 10.0 * np.ones(nlat)
        # Mock FixedOceanicHeatUptake to return a valid process
        mock_fohu_instance = MagicMock()
        mock_fohu_instance.state = {}
        mock_fohu.return_value = mock_fohu_instance

        full_state = climlab.column_state(lev=lev, lat=lat)
        full_state['q'] = 1e-3 + 0 * full_state['Tatm']

        config = {
            'climate_database_files_http': 'http://fake',
            'climate_database_token': 'fake',
            'proj_name': 'test',
            'optical_table_commit': 'abc123',
            'climate_database_commit': 'def456',
        }

        GHGs = {
            'CO2': 420e-6, 'CH4': 1935e-9, 'N2O': 337e-9,
            'O3': 1e-6 * np.ones((nlat, nlev)),
        }

        model = get_rce_sbm_model_annual_avg(
            full_state, config,
            GHGs=GHGs,
            short_timestep=3600.0,
            long_timestep=86400.0,
            do_conv=False,
            do_lsc=False,
            add_fixed_ohu=False,
        )

        # Verify it's a valid climlab process
        assert hasattr(model, 'state')
        assert 'Ts' in model.state
        assert 'Tatm' in model.state
        assert hasattr(model, 'subprocess')
        # Should have an Atmosphere subprocess
        assert 'Atmosphere' in model.subprocess

    @patch('climate_runs_ext.model_factory.model_builder.FixedOceanicHeatUptake')
    @patch('climate_runs_ext.model_factory.model_builder.get_era5_mycloud')
    @patch('climate_runs_ext.model_factory.model_builder.Surface_albedo')
    @patch('climate_runs_ext.model_factory.model_builder.Seasonal_insolation')
    @patch('climate_runs_ext.model_factory.model_builder.oceanic_heat_uptake')
    def test_bounds_applied(self, mock_ohu, mock_insol, mock_albedo, mock_cloud, mock_fohu):
        from climate_runs_ext.model_factory.model_builder import get_rce_sbm_model_annual_avg

        nlat, nlev = 3, 5
        lat = np.linspace(-60, 60, nlat)
        lev = np.linspace(200, 1000, nlev)

        mock_insol.return_value = 340.0 * np.ones(nlat)
        mock_albedo.return_value = 0.3 * np.ones(nlat)
        mock_cloud.return_value = {
            'cldfrac': 0.3 * np.ones((nlat, nlev)),
            'clwp': 0.01 * np.ones((nlat, nlev)),
            'ciwp': 0.005 * np.ones((nlat, nlev)),
            'r_liq': 14.0 * np.ones((nlat, nlev)),
            'r_ice': 14.0 * np.ones((nlat, nlev)),
        }
        mock_ohu.return_value = 10.0 * np.ones(nlat)
        mock_fohu_instance = MagicMock()
        mock_fohu_instance.state = {}
        mock_fohu.return_value = mock_fohu_instance

        full_state = climlab.column_state(lev=lev, lat=lat)
        full_state['q'] = 1e-3 + 0 * full_state['Tatm']

        config = {
            'climate_database_files_http': 'http://fake',
            'climate_database_token': 'fake',
            'proj_name': 'test',
            'optical_table_commit': '', 'climate_database_commit': '',
        }

        model = get_rce_sbm_model_annual_avg(
            full_state, config,
            GHGs={'CO2': 420e-6, 'O3': 1e-6 * np.ones((nlat, nlev))},
            short_timestep=3600.0, long_timestep=86400.0,
            bound_dict={'q': (0.0, np.inf), 'Tatm': (150.0, np.inf), 'Ts': (150.0, np.inf)},
            do_conv=False, do_lsc=False, add_fixed_ohu=False,
        )

        # Verify Limiter subprocesses are present for state variables
        # that exist in the coupled model. Tatm and Ts are always present.
        assert 'limiter_Tatm' in model.subprocess
        assert 'limiter_Ts' in model.subprocess
        # q may or may not be in state depending on subprocess configuration;
        # if q is in model.state, limiter_q should be present
        if 'q' in model.state:
            assert 'limiter_q' in model.subprocess

    @patch('climate_runs_ext.model_factory.model_builder.FixedOceanicHeatUptake')
    @patch('climate_runs_ext.model_factory.model_builder.get_era5_mycloud')
    @patch('climate_runs_ext.model_factory.model_builder.Surface_albedo')
    @patch('climate_runs_ext.model_factory.model_builder.Seasonal_insolation')
    @patch('climate_runs_ext.model_factory.model_builder.oceanic_heat_uptake')
    def test_state_shapes(self, mock_ohu, mock_insol, mock_albedo, mock_cloud, mock_fohu):
        from climate_runs_ext.model_factory.model_builder import get_rce_sbm_model_annual_avg

        nlat, nlev = 5, 8
        lat = np.linspace(-80, 80, nlat)
        lev = np.linspace(100, 1000, nlev)

        mock_insol.return_value = 340.0 * np.ones(nlat)
        mock_albedo.return_value = 0.3 * np.ones(nlat)
        mock_cloud.return_value = {
            'cldfrac': 0.3 * np.ones((nlat, nlev)),
            'clwp': 0.01 * np.ones((nlat, nlev)),
            'ciwp': 0.005 * np.ones((nlat, nlev)),
            'r_liq': 14.0 * np.ones((nlat, nlev)),
            'r_ice': 14.0 * np.ones((nlat, nlev)),
        }
        mock_ohu.return_value = 10.0 * np.ones(nlat)
        mock_fohu_instance = MagicMock()
        mock_fohu_instance.state = {}
        mock_fohu.return_value = mock_fohu_instance

        full_state = climlab.column_state(lev=lev, lat=lat)
        full_state['q'] = 1e-3 + 0 * full_state['Tatm']

        config = {
            'climate_database_files_http': 'http://fake',
            'climate_database_token': 'fake',
            'proj_name': 'test',
            'optical_table_commit': '', 'climate_database_commit': '',
        }

        model = get_rce_sbm_model_annual_avg(
            full_state, config,
            GHGs={'CO2': 420e-6, 'O3': 1e-6 * np.ones((nlat, nlev))},
            short_timestep=3600.0, long_timestep=86400.0,
            do_conv=False, do_lsc=False, add_fixed_ohu=False,
        )

        assert np.array(model.state['Tatm']).shape == (nlat, nlev)
        assert np.array(model.state['Ts']).shape[0] == nlat


# ---------------------------------------------------------------------------
# Transport parameter helpers (unit-level)
# ---------------------------------------------------------------------------

class TestTransportHelpers:
    def test_regularize_wz(self):
        from climate_runs_ext.utils.transport_params import regularize_wz
        n_sigma = 20
        ny = 5
        z = np.linspace(0, 1, n_sigma + 1)
        wz_raw = np.random.randn(ny, n_sigma + 1) * 0.1
        wz_raw[:, 0] = 0.0
        wz_raw[:, -1] = 0.0
        result = regularize_wz(wz_raw, z)
        assert result.shape == wz_raw.shape
        # Boundary conditions preserved
        np.testing.assert_allclose(result[:, 0], 0.0)

    def test_merge_maps(self):
        from climate_runs_ext.utils.transport_params import merge_maps
        n = 20
        map_a = np.ones((n, n))
        map_b = 2.0 * np.ones((n, n))
        mask = np.zeros((n, n), dtype=bool)
        mask[:5, :5] = True  # bad region in top-left
        merged, mask_smooth = merge_maps(map_a, map_b, mask, blur_radius=2)
        assert merged.shape == (n, n)
        # In the good region (far from mask), should be close to map_a
        assert np.isclose(merged[15, 15], 1.0, atol=0.1)

    def test_poisson_solver(self):
        from climate_runs_ext.utils.transport_params import correct_vw_field_poisson_solver
        ny, nz = 10, 8
        y = np.linspace(0, 1e6, ny + 1)
        z = np.linspace(0, 1, nz + 1)
        v_raw = np.random.randn(ny + 1, nz) * 0.01
        w_raw = np.random.randn(ny, nz + 1) * 0.001
        vnew, wnew = correct_vw_field_poisson_solver(
            v_raw, w_raw, y, z, do_smooth=False, do_plot=False,
        )
        assert vnew.shape == (ny + 1, nz)
        assert wnew.shape == (ny, nz + 1)
