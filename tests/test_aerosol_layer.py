"""Tests for climate_runs_ext.utils.aerosol_layer.

Covers the multi-bin, file-based layer loader that mirrors the AGU2025
transport-to-radiation pipeline.  Tests use real on-disk inputs (small
hand-built netCDF + npz in a tmp_path) rather than mocks, so the actual
interpolation and unit conversion paths are exercised.
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import os
import numpy as np
import pytest
import xarray as xr
import climlab

from climate_runs_ext.utils.aerosol_layer import load_multi_bin_aerosol_state


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_domain(num_lev=20, num_lat=9):
    """Return a climlab Tatm domain suitable as a model target."""
    state = climlab.column_state(num_lev=num_lev, num_lat=num_lat)
    return state['Tatm'].domain


def _write_state_and_mapping(
    tmp_path, bin_name_to_radius, mmr_profile_fn,
    nlat_src=45, nlev_src=30,
):
    """Build a minimal state_xr.nc + radius_mapping.npz in tmp_path.

    ``mmr_profile_fn(name, lat_arr, lev_arr) -> (nlat, nlev)`` returns
    the mmr field to write for a given bin.
    """
    lat = np.linspace(-89, 89, nlat_src)
    lev = np.linspace(1.0, 1000.0, nlev_src)

    data_vars = {}
    for name in bin_name_to_radius:
        vals = mmr_profile_fn(name, lat, lev)
        data_vars[name] = (('lat', 'lev'), vals)
    ds = xr.Dataset(
        data_vars=data_vars,
        coords={'lat': lat, 'lev': lev},
    )
    state_path = str(tmp_path / 'state_xr.nc')
    ds.to_netcdf(state_path)

    mapping_path = str(tmp_path / 'radius_mapping.npz')
    np.savez(
        mapping_path,
        **{k: np.array(v) for k, v in bin_name_to_radius.items()},
    )
    return state_path, mapping_path


# ---------------------------------------------------------------------------
# Single-bin basic checks
# ---------------------------------------------------------------------------

class TestSingleBin:

    def _setup(self, tmp_path):
        # 250 nm single bin, uniform mmr=1e-7 in lev=[30, 80] hPa
        mapping = {'Si_1': 2.5e-7}

        def profile(name, lat_arr, lev_arr):
            out = np.zeros((len(lat_arr), len(lev_arr)))
            mask = (lev_arr >= 30.0) & (lev_arr <= 80.0)
            out[:, mask] = 1e-7
            return out

        return _write_state_and_mapping(tmp_path, mapping, profile)

    def test_returns_one_aerosol_instance(self, tmp_path):
        state_path, mapping_path = self._setup(tmp_path)
        result = load_multi_bin_aerosol_state(
            state_path=state_path,
            radius_mapping_path=mapping_path,
            material_name='silica',
            rho_particle=2196.0,
            domain=_make_domain(),
        )
        assert len(result.aerosol_instance_list) == 1
        assert result.bin_names == ['Si_1']
        assert result.bin_radii_m == [2.5e-7]

    def test_avg_diameter_equals_single_bin_diameter(self, tmp_path):
        state_path, mapping_path = self._setup(tmp_path)
        result = load_multi_bin_aerosol_state(
            state_path=state_path,
            radius_mapping_path=mapping_path,
            material_name='silica',
            rho_particle=2196.0,
            domain=_make_domain(),
        )
        assert result.avg_diameter_m == pytest.approx(5.0e-7)

    def test_total_mass_positive_and_reasonable(self, tmp_path):
        state_path, mapping_path = self._setup(tmp_path)
        result = load_multi_bin_aerosol_state(
            state_path=state_path,
            radius_mapping_path=mapping_path,
            material_name='silica',
            rho_particle=2196.0,
            domain=_make_domain(),
        )
        # mmr=1e-7 over ~50 hPa column, globally. Expect O(few Tg).
        assert result.total_mass_Tg > 0.0
        assert result.total_mass_Tg < 100.0

    def test_vmr_on_model_grid(self, tmp_path):
        """The aerosol_instance's vmr should have the model grid shape
        and be non-negative."""
        state_path, mapping_path = self._setup(tmp_path)
        domain = _make_domain(num_lev=25, num_lat=7)
        result = load_multi_bin_aerosol_state(
            state_path=state_path,
            radius_mapping_path=mapping_path,
            material_name='silica',
            rho_particle=2196.0,
            domain=domain,
        )
        vmr = result.aerosol_instance_list[0].vmr
        assert vmr.shape == (7, 25)
        assert np.all(vmr >= 0.0)


# ---------------------------------------------------------------------------
# Multi-bin aggregation
# ---------------------------------------------------------------------------

class TestMultiBin:

    def _setup(self, tmp_path):
        mapping = {'Si_1': 2.5e-7, 'Si_2': 5.0e-7}

        def profile(name, lat_arr, lev_arr):
            out = np.zeros((len(lat_arr), len(lev_arr)))
            mask = (lev_arr >= 30.0) & (lev_arr <= 80.0)
            # Bin 1 carries twice the mmr of bin 2 so the mass-weighted
            # mean diameter is closer to bin 1's diameter.
            out[:, mask] = 2e-7 if name == 'Si_1' else 1e-7
            return out

        return _write_state_and_mapping(tmp_path, mapping, profile)

    def test_two_aerosol_instances(self, tmp_path):
        state_path, mapping_path = self._setup(tmp_path)
        result = load_multi_bin_aerosol_state(
            state_path=state_path,
            radius_mapping_path=mapping_path,
            material_name='silica',
            rho_particle=2196.0,
            domain=_make_domain(),
        )
        assert len(result.aerosol_instance_list) == 2

    def test_mass_weighted_avg_diameter(self, tmp_path):
        """With bin 1 mass > bin 2 mass, avg D should sit between the two
        monodisperse diameters, closer to bin 1."""
        state_path, mapping_path = self._setup(tmp_path)
        result = load_multi_bin_aerosol_state(
            state_path=state_path,
            radius_mapping_path=mapping_path,
            material_name='silica',
            rho_particle=2196.0,
            domain=_make_domain(),
        )
        D1, D2 = 5.0e-7, 1.0e-6  # bin 1 and bin 2 diameters
        assert D1 < result.avg_diameter_m < D2


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrors:

    def test_missing_bin_raises(self, tmp_path):
        """Bin in the mapping but not in the state file -> KeyError."""
        mapping = {'Si_1': 2.5e-7, 'Ghost_bin': 1e-7}

        def profile(name, lat_arr, lev_arr):
            # Only Si_1 is written to the file
            if name != 'Si_1':
                return np.zeros((len(lat_arr), len(lev_arr)))
            out = np.zeros((len(lat_arr), len(lev_arr)))
            out[:, (lev_arr >= 30) & (lev_arr <= 80)] = 1e-7
            return out

        # Manually build the state to exclude Ghost_bin
        lat = np.linspace(-89, 89, 45)
        lev = np.linspace(1.0, 1000.0, 30)
        ds = xr.Dataset(
            {'Si_1': (('lat', 'lev'), profile('Si_1', lat, lev))},
            coords={'lat': lat, 'lev': lev},
        )
        state_path = str(tmp_path / 'state_xr.nc')
        ds.to_netcdf(state_path)
        mapping_path = str(tmp_path / 'radius_mapping.npz')
        np.savez(mapping_path, **{k: np.array(v) for k, v in mapping.items()})

        with pytest.raises(KeyError, match='Ghost_bin'):
            load_multi_bin_aerosol_state(
                state_path=state_path,
                radius_mapping_path=mapping_path,
                material_name='silica',
                rho_particle=2196.0,
                domain=_make_domain(),
            )

    def test_negative_mmr_clipped(self, tmp_path):
        """Tiny negative mmr values (numerical noise) should be clipped
        to zero, not propagated into vmr."""
        mapping = {'Si_1': 2.5e-7}

        def profile(name, lat_arr, lev_arr):
            out = np.zeros((len(lat_arr), len(lev_arr)))
            out[:, (lev_arr >= 30) & (lev_arr <= 80)] = 1e-7
            # Inject a tiny negative noise value
            out[0, 0] = -1e-16
            return out

        state_path, mapping_path = _write_state_and_mapping(
            tmp_path, mapping, profile,
        )
        result = load_multi_bin_aerosol_state(
            state_path=state_path,
            radius_mapping_path=mapping_path,
            material_name='silica',
            rho_particle=2196.0,
            domain=_make_domain(),
        )
        vmr = result.aerosol_instance_list[0].vmr
        assert np.all(vmr >= 0.0)
