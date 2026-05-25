"""Tests for ERA5 data loading functions (Phase 6c).

All tests use mocks to avoid network access and real data dependencies.
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import numpy as np
import pytest
from unittest.mock import patch, MagicMock
import xarray as xr

from climate_runs_ext.utils.era5_data import (
    SeasonTypes,
    lat_avg,
    Smooth,
    Seasonal_insolation,
    Surface_albedo,
    Cloud_Cover,
    get_cloud_water_path,
    get_era5_mycloud,
    Ozone,
    Relative_Humidity_Profile,
    era5_annual_initial_state,
    oceanic_heat_uptake,
    get_surface_fluxes,
    get_surface_flux_drag_coe,
    surface_params_era5,
    surface_pressure_era5,
    meridional_Kq,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MOCK_CONFIG = {
    'climate_database_files_http': 'http://fake',
    'climate_database_token': 'fake',
    'proj_name': 'test',
}


def _make_era5_lat():
    """ERA5-like latitude array (south to north)."""
    return np.linspace(-90, 90, 37)


def _make_era5_lev():
    """ERA5-like pressure levels (hPa)."""
    return np.array([1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150,
                     175, 200, 225, 250, 300, 350, 400, 450, 500, 550, 600,
                     650, 700, 750, 775, 800, 825, 850, 875, 900, 925, 950,
                     975, 1000], dtype=float)


# ---------------------------------------------------------------------------
# SeasonTypes
# ---------------------------------------------------------------------------

class TestSeasonTypes:
    def test_months_dict_annual(self):
        assert SeasonTypes.months_dict['Annual'] == list(range(12))

    def test_months_dict_djf(self):
        assert SeasonTypes.months_dict['DJF'] == [11, 0, 1]

    def test_months_dict_single_month(self):
        assert SeasonTypes.months_dict['Jul'] == [6]

    def test_days_in_month(self):
        assert SeasonTypes.days_in_month_dict['Feb'] == 28

    def test_month_str_dict(self):
        assert SeasonTypes.month_str_dict[0] == 'Jan'
        assert SeasonTypes.month_str_dict[11] == 'Dec'


# ---------------------------------------------------------------------------
# lat_avg
# ---------------------------------------------------------------------------

class TestLatAvg:
    def test_uniform_field(self):
        lat = np.linspace(-90, 90, 10)
        field = np.ones(10)
        result = lat_avg(field, lat)
        assert np.isclose(result, 1.0, atol=1e-10)

    def test_weighted(self):
        lat = np.array([0.0, 60.0])
        field = np.array([1.0, 1.0])
        result = lat_avg(field, lat)
        # cos(0)=1, cos(60)=0.5; weighted avg = (1+0.5)/(1+0.5) = 1.0
        assert np.isclose(result, 1.0, atol=1e-10)

    def test_2d_field(self):
        lat = np.linspace(-90, 90, 5)
        field = np.ones((5, 3))
        result = lat_avg(field, lat)
        assert result.shape == (3,)
        np.testing.assert_allclose(result, 1.0, atol=1e-10)


# ---------------------------------------------------------------------------
# Smooth
# ---------------------------------------------------------------------------

class TestSmooth:
    def test_basic(self):
        x = np.sin(np.linspace(0, 2 * np.pi, 50))
        result = Smooth(x, window_length=5)
        assert result.shape == x.shape


# ---------------------------------------------------------------------------
# Seasonal_insolation
# ---------------------------------------------------------------------------

class TestSeasonalInsolation:
    def test_annual_shape(self):
        lat = np.linspace(-90, 90, 19)
        result = Seasonal_insolation(lat, 'Annual')
        assert result.shape == (19,)

    def test_positive(self):
        lat = np.linspace(-80, 80, 17)
        result = Seasonal_insolation(lat, 'Annual')
        assert np.all(result >= 0.0)

    def test_seasonal_shape(self):
        lat = np.linspace(-90, 90, 10)
        result = Seasonal_insolation(lat, 'JJA')
        assert result.shape == (10,)


# ---------------------------------------------------------------------------
# Surface_albedo (mocked)
# ---------------------------------------------------------------------------

class TestSurfaceAlbedo:
    @patch('climate_runs_ext.utils.era5_data.load_xr_from_repo')
    def test_shape(self, mock_load):
        lat_era5 = _make_era5_lat()
        months_dim = np.arange(12)
        nlat = len(lat_era5)
        # Create fake dataset
        sw_down = xr.DataArray(
            300.0 * np.ones((12, nlat)),
            dims=['month', 'latitude'],
            coords={'month': months_dim, 'latitude': lat_era5},
        )
        sw_net = xr.DataArray(
            200.0 * np.ones((12, nlat)),
            dims=['month', 'latitude'],
            coords={'month': months_dim, 'latitude': lat_era5},
        )
        ds = MagicMock()
        ds.msdwswrf = sw_down
        ds.msnswrf = sw_net
        mock_load.return_value = ds

        lat = np.linspace(-80, 80, 10)
        months = list(range(12))
        result = Surface_albedo(lat, months, MOCK_CONFIG)
        assert result.shape == (10,)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)


# ---------------------------------------------------------------------------
# oceanic_heat_uptake (mocked)
# ---------------------------------------------------------------------------

class TestOceanicHeatUptake:
    @patch('climate_runs_ext.utils.era5_data.load_xr_from_repo')
    def test_shape(self, mock_load):
        lat_era5 = _make_era5_lat()
        oha = xr.DataArray(
            3600.0 * 10.0 * np.ones((12, len(lat_era5))),
            dims=['month', 'latitude'],
            coords={'month': np.arange(12), 'latitude': lat_era5},
        )
        ds = MagicMock()
        ds.oha = oha
        mock_load.return_value = ds

        lat = np.linspace(-80, 80, 10)
        result = oceanic_heat_uptake(lat, list(range(12)), MOCK_CONFIG)
        assert result.shape == (10,)
        # 3600*10/3600 = 10 W/m2
        np.testing.assert_allclose(result, 10.0, atol=0.5)


# ---------------------------------------------------------------------------
# Cloud_Cover (mocked)
# ---------------------------------------------------------------------------

class TestCloudCover:
    @patch('climate_runs_ext.utils.era5_data.load_xr_from_repo')
    def test_shape_and_range(self, mock_load):
        lat_era5 = _make_era5_lat()
        lev_era5 = _make_era5_lev()
        cc_data = 0.5 * np.ones((12, len(lev_era5), len(lat_era5)))
        cc = xr.DataArray(
            cc_data,
            dims=['time', 'level', 'latitude'],
            coords={
                'time': np.arange(12),
                'level': lev_era5,
                'latitude': lat_era5[::-1],
            },
        )
        ds = MagicMock()
        ds.cloud_cover = cc
        mock_load.return_value = ds

        new_lev = np.array([100, 500, 900], dtype=float)
        new_lat = np.array([-60, 0, 60], dtype=float)
        result = Cloud_Cover(new_lev, new_lat, list(range(12)), MOCK_CONFIG)
        assert result.shape == (3, 3)
        assert np.all(result >= 0.0)


class TestGetCloudWaterPath:
    """Regression test: the Cloud_content ERA5 file only covers 100-1000 hPa.
    Any cubic extrapolation above 100 hPa produces spurious monotonically-
    increasing clouds up to TOA (the bug fixed in this PR).  The contract
    is: ciwp / clwp must be exactly 0.0 strictly above the ERA5 top level.
    """

    @patch('climate_runs_ext.utils.era5_data.load_xr_from_repo')
    def test_zero_above_era5_top(self, mock_load):
        # ERA5 Cloud_content levels mirror the real file: 100-1000 hPa only.
        lat_era5 = _make_era5_lat()
        lev_era5 = np.array([100, 125, 150, 175, 200, 225, 250, 300, 350,
                             400, 450, 500, 550, 600, 650, 700, 750, 775,
                             800, 825, 850, 875, 900, 925, 950, 975, 1000],
                            dtype=float)
        shape = (12, len(lev_era5), len(lat_era5))
        # Non-trivial cloud content that would extrapolate upward under cubic
        ciwc = (1e-5 * np.maximum(0.0, 1.0 - (lev_era5 - 250.0) ** 2 / 5e4))
        ciwc = np.broadcast_to(
            ciwc[np.newaxis, :, np.newaxis], shape,
        ).copy()
        clwc = 5e-6 * np.ones(shape)
        ds = xr.Dataset({
            'ciwc': xr.DataArray(
                ciwc, dims=['month', 'level', 'latitude'],
                coords={'month': np.arange(12), 'level': lev_era5,
                        'latitude': lat_era5},
            ),
            'clwc': xr.DataArray(
                clwc, dims=['month', 'level', 'latitude'],
                coords={'month': np.arange(12), 'level': lev_era5,
                        'latitude': lat_era5},
            ),
        })
        mock_load.return_value = ds

        # Build a minimal domain covering 5-1000 hPa (model top at 5 hPa
        # is 20x above ERA5 top; cubic extrapolation was producing ~8 g/m^2
        # here before the fix).
        import climlab
        num_lev = 50
        state = climlab.column_state(
            num_lev=num_lev, num_lat=len(lat_era5), water_depth=1.0,
        )
        model_lev = np.linspace(5.0, 995.0, num_lev)
        state['Tatm'].domain.axes['lev'].points[:] = model_lev
        # Rebuild lev.delta consistently with new points
        state['Tatm'].domain.axes['lev'].bounds[:] = np.concatenate(
            [[0.0], 0.5 * (model_lev[:-1] + model_lev[1:]), [1000.0]],
        )
        state['Tatm'].domain.axes['lev'].delta[:] = np.diff(
            state['Tatm'].domain.axes['lev'].bounds,
        )

        clwp, ciwp = get_cloud_water_path(
            state['Tatm'].domain, list(range(12)), MOCK_CONFIG,
        )
        # Regression contract: strictly above ERA5 top (100 hPa),
        # ciwp and clwp must be exactly zero.  Cubic extrapolation
        # with fill_value=None violated this.
        above_era5 = model_lev < 100.0
        assert np.all(ciwp[:, above_era5] == 0.0), (
            f'ciwp leaked above ERA5 top: max={ciwp[:, above_era5].max():.3e}'
        )
        assert np.all(clwp[:, above_era5] == 0.0), (
            f'clwp leaked above ERA5 top: max={clwp[:, above_era5].max():.3e}'
        )
        # And sanity: below ERA5 top we still get positive water path
        # (i.e. the interpolation itself is still working).
        below_era5 = (model_lev >= 100.0) & (model_lev <= 1000.0)
        assert ciwp[:, below_era5].max() > 0.0
        assert clwp[:, below_era5].max() > 0.0


# ---------------------------------------------------------------------------
# Ozone (mocked)
# ---------------------------------------------------------------------------

class TestOzone:
    @patch('climate_runs_ext.utils.era5_data.load_xr_from_repo')
    def test_shape_and_non_negative(self, mock_load):
        lat_era5 = _make_era5_lat()
        lev_era5 = _make_era5_lev()
        o3_data = 1e-5 * np.ones((12, len(lev_era5), len(lat_era5)))
        o3 = xr.DataArray(
            o3_data,
            dims=['time', 'level', 'latitude'],
            coords={
                'time': np.arange(12),
                'level': lev_era5,
                'latitude': lat_era5[::-1],
            },
        )
        ds = MagicMock()
        ds.o3 = o3
        mock_load.return_value = ds

        new_lev = np.array([50, 200, 500, 850], dtype=float)
        new_lat = np.array([-60, -30, 0, 30, 60], dtype=float)
        result = Ozone(new_lev, new_lat, list(range(12)), MOCK_CONFIG)
        assert result.shape == (5, 4)
        assert np.all(result >= 0.0)


# ---------------------------------------------------------------------------
# Relative_Humidity_Profile (mocked)
# ---------------------------------------------------------------------------

class TestRelativeHumidityProfile:
    @patch('climate_runs_ext.utils.era5_data.load_xr_from_repo')
    def test_shape(self, mock_load):
        lat_era5 = _make_era5_lat()
        lev_era5 = _make_era5_lev()
        rh_data = 50.0 * np.ones((3, 12, len(lev_era5), len(lat_era5)))
        ds = xr.Dataset(
            {'RH': xr.DataArray(
                rh_data,
                dims=['year', 'month', 'level', 'latitude'],
                coords={
                    'year': [2008, 2009, 2010],
                    'month': np.arange(12),
                    'level': lev_era5,
                    'latitude': lat_era5,
                },
            )},
        )
        mock_load.return_value = ds

        new_lev = np.array([200, 500, 850], dtype=float)
        new_lat = np.array([-60, 0, 60], dtype=float)
        result = Relative_Humidity_Profile(new_lev, new_lat, list(range(12)), MOCK_CONFIG)
        assert result.shape == (3, 3)
        # 50% -> 0.5
        np.testing.assert_allclose(result, 0.5, atol=0.05)


# ---------------------------------------------------------------------------
# get_era5_mycloud (mocked, composite)
# ---------------------------------------------------------------------------

class TestGetEra5Mycloud:
    @patch('climate_runs_ext.utils.era5_data.get_cloud_water_path')
    @patch('climate_runs_ext.utils.era5_data.Cloud_Cover')
    def test_keys_and_shapes(self, mock_cc, mock_cwp):
        nlat, nlev = 5, 10
        mock_cc.return_value = 0.3 * np.ones((nlat, nlev))
        mock_cwp.return_value = (
            0.01 * np.ones((nlat, nlev)),
            0.005 * np.ones((nlat, nlev)),
        )

        domain = MagicMock()
        domain.lev.points = np.linspace(100, 1000, nlev)
        domain.lat.points = np.linspace(-80, 80, nlat)

        result = get_era5_mycloud(domain, list(range(12)), MOCK_CONFIG)
        assert set(result.keys()) == {'cldfrac', 'ciwp', 'clwp', 'r_ice', 'r_liq'}
        assert result['cldfrac'].shape == (nlat, nlev)
        assert result['r_liq'].shape == (nlat, nlev)
        assert result['r_ice'].shape == (nlat, nlev)


# ---------------------------------------------------------------------------
# get_surface_fluxes (mocked)
# ---------------------------------------------------------------------------

class TestGetSurfaceFluxes:
    @patch('climate_runs_ext.utils.era5_data.load_xr_from_repo')
    def test_shape(self, mock_load):
        lat_era5 = _make_era5_lat()
        slhf = xr.DataArray(
            -50.0 * np.ones(len(lat_era5)),
            dims=['latitude'],
            coords={'latitude': lat_era5},
        )
        sshf = xr.DataArray(
            -20.0 * np.ones(len(lat_era5)),
            dims=['latitude'],
            coords={'latitude': lat_era5},
        )
        ds = MagicMock()
        ds.slhf = slhf
        ds.sshf = sshf
        # mean returns the same mock for simplicity
        ds.mean.return_value = ds
        ds.sel.return_value = ds
        mock_load.return_value = ds

        lhf, shf = get_surface_fluxes(MOCK_CONFIG)
        assert lhf.shape == (len(lat_era5),)


# ---------------------------------------------------------------------------
# get_surface_flux_drag_coe (mocked)
# ---------------------------------------------------------------------------

class TestGetSurfaceFluxDragCoe:
    @patch('climate_runs_ext.utils.era5_data.surface_params_era5')
    @patch('climate_runs_ext.utils.era5_data.get_surface_fluxes')
    def test_shapes(self, mock_fluxes, mock_sfc_params):
        nlat = 10
        lat = np.linspace(-80, 80, nlat)

        lhf_vals = -50.0 * np.ones(nlat)
        shf_vals = -20.0 * np.ones(nlat)
        lhf_da = xr.DataArray(lhf_vals, dims=['latitude'], coords={'latitude': lat})
        shf_da = xr.DataArray(shf_vals, dims=['latitude'], coords={'latitude': lat})
        mock_fluxes.return_value = (lhf_da, shf_da)

        ts = xr.DataArray(300.0 * np.ones(nlat), dims=['latitude'], coords={'latitude': lat})
        ts.values = 300.0 * np.ones(nlat)
        t2m = xr.DataArray(295.0 * np.ones(nlat), dims=['latitude'], coords={'latitude': lat})
        qs = xr.DataArray(0.02 * np.ones(nlat), dims=['latitude'], coords={'latitude': lat})
        q2m = xr.DataArray(0.01 * np.ones(nlat), dims=['latitude'], coords={'latitude': lat})
        mock_sfc_params.return_value = {'ts': ts, 't2m': t2m, 'qs': qs, 'q2m': q2m}

        Cd_lhf, Cd_shf, lat_out = get_surface_flux_drag_coe(
            MOCK_CONFIG, new_lat=lat, months=list(range(12)),
            do_smooth=False,
        )
        assert Cd_lhf.shape == (nlat, 1)
        assert Cd_shf.shape == (nlat, 1)
        assert len(lat_out) == nlat


# ---------------------------------------------------------------------------
# meridional_Kq (mocked)
# ---------------------------------------------------------------------------

class TestMeridionalKq:
    @patch('climate_runs_ext.utils.era5_data.load_xr_from_repo')
    def test_shape(self, mock_load):
        lat_era5 = np.linspace(-80, 80, 30)
        Kq_vals = 5e4 * np.ones(30)
        ds = MagicMock()
        ds.Kq_avg.latitude.values = lat_era5
        ds.Kq_avg.values = Kq_vals
        mock_load.return_value = ds

        new_lat = np.linspace(-70, 70, 15)
        result = meridional_Kq(new_lat, MOCK_CONFIG, Kmin=1e3)
        assert result.shape == (15,)


# ---------------------------------------------------------------------------
# era5_annual_initial_state (mocked)
# ---------------------------------------------------------------------------

class TestEra5AnnualInitialState:
    """Verify ERA5 -> model grid initial-state loader."""

    def _make_domain(self, num_lev=20, num_lat=5):
        import climlab
        state = climlab.column_state(num_lev=num_lev, num_lat=num_lat)
        return state['Tatm'].domain

    def _mock_variables_ds(self):
        """Build a fake Monthly_Zonal_Variables_2008_2017 dataset."""
        lat_era5 = _make_era5_lat()[::-1]  # ERA5 native: 90 -> -90
        lev_era5 = _make_era5_lev()
        nlat, nlev = len(lat_era5), len(lev_era5)
        # Simple profiles: T linear in level, q exponential in level
        T_data = np.broadcast_to(
            np.linspace(220.0, 290.0, nlev)[None, :, None],
            (12, nlev, nlat),
        ).copy()
        q_data = np.broadcast_to(
            np.linspace(1e-6, 1e-2, nlev)[None, :, None],
            (12, nlev, nlat),
        ).copy()
        T_da = xr.DataArray(
            T_data, dims=['month', 'level', 'latitude'],
            coords={'month': np.arange(12), 'level': lev_era5,
                    'latitude': lat_era5},
        )
        q_da = xr.DataArray(
            q_data, dims=['month', 'level', 'latitude'],
            coords={'month': np.arange(12), 'level': lev_era5,
                    'latitude': lat_era5},
        )
        ds = MagicMock()
        ds.T = T_da
        ds.q = q_da
        return ds

    def _mock_surface_params(self):
        """Fake surface_params_era5 return — just needs a ts DataArray."""
        lat_era5 = _make_era5_lat()[::-1]
        ts_vals = 288.0 + 10.0 * np.cos(np.deg2rad(lat_era5))  # lat-dep
        ts_da = xr.DataArray(
            ts_vals, dims=['latitude'],
            coords={'latitude': lat_era5},
        )
        srf = MagicMock()
        srf.__getitem__.side_effect = lambda k: ts_da if k == 'ts' else None
        return srf

    @patch('climate_runs_ext.utils.era5_data.surface_params_era5')
    @patch('climate_runs_ext.utils.era5_data.load_xr_from_repo')
    def test_shapes(self, mock_load, mock_srf):
        mock_load.return_value = self._mock_variables_ds()
        mock_srf.return_value = self._mock_surface_params()

        domain = self._make_domain(num_lev=20, num_lat=5)
        result = era5_annual_initial_state(domain, list(range(12)), MOCK_CONFIG)

        assert set(result.keys()) == {'Tatm', 'q', 'Ts'}
        assert result['Tatm'].shape == (5, 20)
        assert result['q'].shape == (5, 20)
        assert result['Ts'].shape == (5, 1)

    @patch('climate_runs_ext.utils.era5_data.surface_params_era5')
    @patch('climate_runs_ext.utils.era5_data.load_xr_from_repo')
    def test_q_nonnegative(self, mock_load, mock_srf):
        mock_load.return_value = self._mock_variables_ds()
        mock_srf.return_value = self._mock_surface_params()

        domain = self._make_domain(num_lev=30, num_lat=7)
        result = era5_annual_initial_state(domain, list(range(12)), MOCK_CONFIG)
        assert np.all(result['q'] > 0.0)

    @patch('climate_runs_ext.utils.era5_data.surface_params_era5')
    @patch('climate_runs_ext.utils.era5_data.load_xr_from_repo')
    def test_T_within_physical_range(self, mock_load, mock_srf):
        mock_load.return_value = self._mock_variables_ds()
        mock_srf.return_value = self._mock_surface_params()

        domain = self._make_domain(num_lev=20, num_lat=5)
        result = era5_annual_initial_state(domain, list(range(12)), MOCK_CONFIG)
        # Input T was in [220, 290]; interpolation + extrapolation should
        # stay in a reasonable neighborhood.
        assert result['Tatm'].min() > 150.0
        assert result['Tatm'].max() < 350.0

    @patch('climate_runs_ext.utils.era5_data.surface_params_era5')
    @patch('climate_runs_ext.utils.era5_data.load_xr_from_repo')
    def test_Ts_latitude_dependence(self, mock_load, mock_srf):
        mock_load.return_value = self._mock_variables_ds()
        mock_srf.return_value = self._mock_surface_params()

        domain = self._make_domain(num_lev=20, num_lat=9)
        result = era5_annual_initial_state(domain, list(range(12)), MOCK_CONFIG)
        # Equator warmer than poles (mock used cos(lat))
        Ts = result['Ts'].flatten()
        lat = domain.lat.points
        i_eq = int(np.argmin(np.abs(lat)))
        i_pole = int(np.argmax(np.abs(lat)))
        assert Ts[i_eq] > Ts[i_pole]
