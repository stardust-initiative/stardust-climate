"""Tests for diagnostic processes (Phase 6b).

Covers:
- RelativeHumidity: RH computation from state
- FixedOceanicHeatUptake: prescribed OHU tendency
- CloudFeedback: polynomial cloud-feedback parameterisation (with mock data)
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import numpy as np
import pytest
import xarray as xr
import climlab
from climlab.domain.field import Field
from unittest.mock import patch

from climate_runs_ext.diagnostics.relative_humidity import RelativeHumidity
from climate_runs_ext.diagnostics.fixed_oceanic_heat_uptake import (
    FixedOceanicHeatUptake,
)
from climate_runs_ext.diagnostics.cloud_feedback import CloudFeedback


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state_with_q(nlev=30, nlat=10):
    """Build a multi-latitude state with humidity for diagnostics.

    The 'q' variable is created as a proper climlab Field with the
    atmospheric domain so that processes can use it in their state dict.
    """
    state = climlab.column_state(num_lev=nlev, num_lat=nlat)
    # Add a moisture field — exponentially decreasing with height
    lev = state['Tatm'].domain.lev.points
    q_profile = 1e-2 * np.exp(-lev / 500.0)
    q_arr = np.tile(q_profile, (nlat, 1))
    # Wrap as a Field with the atmospheric domain
    atm_domain = state['Tatm'].domain
    state['q'] = Field(q_arr, domain=atm_domain)
    return state


def _make_mock_sensitivity_ds(nlat=10, nlev=30):
    """Create a minimal mock cloud-sensitivity xarray Dataset.

    The sensitivity dataset has:
    - 'cloud fraction-sensitivity' with key dimension ['T', 'rh']
    - 'liquid water content-sensitivity' with same keys
    - 'ice water content-sensitivity' with same keys
    """
    lat = np.linspace(-85, 85, nlat)
    lev = np.linspace(50, 1000, nlev)

    # Sensitivity coefficients: simple uniform values for testing
    # key dimension: 'T' and 'rh'
    keys = ['T', 'rh']
    nkeys = len(keys)

    data_vars = {}
    for var_name in [
        'cloud fraction-sensitivity',
        'liquid water content-sensitivity',
        'ice water content-sensitivity',
    ]:
        vals = 0.01 * np.ones((nkeys, nlev, nlat))
        data_vars[var_name] = (
            ['key', 'level', 'latitude'], vals,
        )

    ds = xr.Dataset(
        data_vars,
        coords={
            'key': keys,
            'level': lev,
            'latitude': lat,
        },
    )
    return ds


def _make_mock_cloud_dict(nlat=10, nlev=30):
    """Create a reference cloud dict for CloudFeedback initialisation."""
    return {
        'cldfrac': 0.5 * np.ones((nlat, nlev)),
        'clwp':    50.0 * np.ones((nlat, nlev)),
        'ciwp':    10.0 * np.ones((nlat, nlev)),
        'r_liq':   10e-6 * np.ones((nlat, nlev)),
        'r_ice':   30e-6 * np.ones((nlat, nlev)),
    }


def _mock_config():
    """Return a minimal config dict (not used for actual data loading)."""
    return {
        'climate_database_files_http': 'https://mock.example.com',
        'climate_database_token': 'mock_token',
        'proj_name': 'test_project',
    }


# ---------------------------------------------------------------------------
# RelativeHumidity
# ---------------------------------------------------------------------------

class TestRelativeHumidity:
    """Verify relative humidity diagnostic."""

    def test_rh_shape(self):
        """RH should have the same shape as Tatm."""
        state = _make_state_with_q()
        rh_proc = RelativeHumidity(state=state)
        assert rh_proc.rh.shape == state['Tatm'].shape

    def test_rh_positive(self):
        """RH should be non-negative."""
        state = _make_state_with_q()
        rh_proc = RelativeHumidity(state=state)
        assert np.all(rh_proc.rh >= 0)

    def test_rh_bounded_near_surface(self):
        """RH should be below 1 near the surface for reasonable q values."""
        state = _make_state_with_q()
        rh_proc = RelativeHumidity(state=state)
        # Near the surface (last few levels) q/qsat should be < 1
        # since we use a modest exponential profile
        rh_surface = rh_proc.rh[:, -1]
        assert np.all(rh_surface < 1.0)

    def test_rh_static_func(self):
        """The static rh_func should agree with the property."""
        state = _make_state_with_q()
        rh_proc = RelativeHumidity(state=state)
        rh_from_func = RelativeHumidity.rh_func(state)
        np.testing.assert_allclose(rh_proc.rh, rh_from_func)

    def test_compute_returns_empty(self):
        """_compute should return empty tendencies dict."""
        state = _make_state_with_q()
        rh_proc = RelativeHumidity(state=state)
        result = rh_proc._compute()
        assert isinstance(result, dict)
        assert len(result) == 0

    def test_rh_updates_on_state_change(self):
        """RH should change when q changes."""
        state = _make_state_with_q()
        rh_proc = RelativeHumidity(state=state)
        rh_before = rh_proc.rh.copy()
        state['q'][:] *= 2.0
        rh_after = rh_proc.rh
        assert not np.allclose(rh_before, rh_after)


# ---------------------------------------------------------------------------
# FixedOceanicHeatUptake
# ---------------------------------------------------------------------------

class TestFixedOceanicHeatUptake:
    """Verify prescribed ocean heat uptake process."""

    def _make_ohu(self, nlat=10, nlev=20):
        """Build a FixedOceanicHeatUptake with a sinusoidal OHU profile."""
        state = climlab.column_state(num_lev=nlev, num_lat=nlat)
        lat = state['Ts'].domain.lat.points
        # OHU: positive in tropics, negative at poles
        oha = 20.0 * np.cos(np.radians(lat))
        return FixedOceanicHeatUptake(oha_lat=oha, state=state)

    def test_init_shape(self):
        """OHU field should match Ts shape."""
        ohu = self._make_ohu()
        assert ohu.ohu.shape == ohu.state['Ts'].shape

    def test_tendency_sign(self):
        """Positive oha_lat should produce negative Ts tendency."""
        ohu = self._make_ohu()
        tendencies = ohu._compute()
        assert 'Ts' in tendencies
        # Where oha_lat > 0, Ts tendency should be < 0
        oha = np.array(ohu.oha_lat).squeeze()
        dTs = np.array(tendencies['Ts']).squeeze()
        # Only check where oha_lat is significantly positive
        mask = oha > 5.0
        assert np.all(dTs[mask] < 0)

    def test_oha_lat_settable(self):
        """Setting oha_lat should update the internal field."""
        ohu = self._make_ohu()
        new_val = 42.0 * np.ones_like(ohu.oha_lat)
        ohu.oha_lat = new_val
        np.testing.assert_allclose(ohu.oha_lat, new_val)

    def test_diagnostic_registered(self):
        """'ohu' should be in the diagnostics dict."""
        ohu = self._make_ohu()
        assert 'ohu' in ohu.diagnostics


# ---------------------------------------------------------------------------
# CloudFeedback
# ---------------------------------------------------------------------------

class TestCloudFeedback:
    """Verify cloud feedback parameterisation with mock sensitivity data."""

    def _make_cloud_feedback(self, nlat=10, nlev=30):
        """Build a CloudFeedback with a mock sensitivity dataset."""
        state = _make_state_with_q(nlev=nlev, nlat=nlat)
        mycloud0 = _make_mock_cloud_dict(nlat=nlat, nlev=nlev)
        mock_ds = _make_mock_sensitivity_ds(nlat=nlat, nlev=nlev)
        config = _mock_config()

        with patch(
            'climate_runs_ext.diagnostics.cloud_feedback.load_xr_from_repo',
            return_value=mock_ds,
        ):
            cf = CloudFeedback(
                mycloud0=mycloud0,
                sensitivity_dataset_filename='mock_sensitivity',
                state=state,
                config=config,
            )
        return cf

    def test_cloud_dict_keys(self):
        """cloud_dict should have the expected cloud-parameter keys."""
        cf = self._make_cloud_feedback()
        cd = cf.cloud_dict
        for key in ('cldfrac', 'clwp', 'ciwp', 'r_liq', 'r_ice'):
            assert key in cd

    def test_cloud_dict_shapes(self):
        """Cloud fields should have (nlat, nlev) shape."""
        nlat, nlev = 10, 30
        cf = self._make_cloud_feedback(nlat=nlat, nlev=nlev)
        cd = cf.cloud_dict
        assert cd['cldfrac'].shape == (nlat, nlev)
        assert cd['clwp'].shape == (nlat, nlev)

    def test_cldfrac_bounded(self):
        """Cloud fraction should be in [0, 1]."""
        cf = self._make_cloud_feedback()
        cd = cf.cloud_dict
        assert np.all(cd['cldfrac'] >= 0)
        assert np.all(cd['cldfrac'] <= 1.0)

    def test_reference_unchanged_at_init(self):
        """With zero anomalies, cloud_dict should match mycloud0."""
        cf = self._make_cloud_feedback()
        cd = cf.cloud_dict
        # At init, dT=0 and drh=0, so cloud_dict ~ mycloud0
        # (small numeric differences due to clipping and nan handling)
        np.testing.assert_allclose(
            cd['cldfrac'], cf.mycloud0['cldfrac'], atol=1e-6,
        )

    def test_update_internal_fields(self):
        """update_internal_fields should update T0 and rh0."""
        cf = self._make_cloud_feedback()
        old_T0 = cf.T0.copy()
        # Modify state and call update
        cf.state['Tatm'][:] += 5.0
        cf.update_internal_fields(ref_state=cf.state)
        assert not np.allclose(cf.T0, old_T0)

    def test_compute_returns_empty(self):
        """_compute should return empty tendencies."""
        cf = self._make_cloud_feedback()
        result = cf._compute()
        assert isinstance(result, dict)
        assert len(result) == 0

    def test_n_avg_setter_resets_avg(self):
        """Setting n_avg should reset the running-average state."""
        cf = self._make_cloud_feedback()
        cf._steps_current = 10  # simulate some steps
        cf.n_avg = 5
        assert cf._steps_current == 0

    def test_cloud_dict_at_ref_matches_mycloud0(self):
        """After a proper reference-state load sequence, cloud_dict
        should equal mycloud0 exactly (within clip tolerance).

        Regression test for the fix/cloud-feedback-load-order bug: if
        update_internal_fields is called with the new reference state
        BEFORE compute_diagnostics fires, CloudFeedback evaluates its
        polynomial against matched T0/q0/rh0 and state, so dT=0, drh=0
        and the polynomial perturbation is zero — cloud_dict should
        reduce to mycloud0. Previously, the driver code path invoked
        compute_diagnostics() via update_model_from_xr(do_compute=True)
        BEFORE calling iteratively_update_internal, which left T0 at
        the climlab column_state defaults and produced a large spurious
        cloud anomaly at step 0.
        """
        cf = self._make_cloud_feedback()
        # Simulate a state load: overwrite Tatm and q to a different
        # profile (as if loading a saved reference netCDF).
        new_Tatm = cf.state['Tatm'].copy() + 10.0  # shifted profile
        new_q = cf.state['q'].copy() * 1.5
        cf.state['Tatm'][:] = new_Tatm
        cf.state['q'][:] = new_q
        # Correct order: update_internal_fields first, then any compute.
        cf.update_internal_fields(ref_state=cf.state)
        cd = cf.cloud_dict
        np.testing.assert_allclose(
            cd['cldfrac'], cf.mycloud0['cldfrac'], atol=1e-10,
        )
        np.testing.assert_allclose(
            cd['clwp'], cf.mycloud0['clwp'], atol=1e-10,
        )
        np.testing.assert_allclose(
            cd['ciwp'], cf.mycloud0['ciwp'], atol=1e-10,
        )

    def test_cloud_dict_pollution_if_state_loaded_before_update_internal(self):
        """Demonstrate the bug path: if state is loaded AND cloud_dict
        is accessed BEFORE update_internal_fields, the result does NOT
        match mycloud0 — it reflects the polynomial evaluated against
        the stale T0.  This is the failure mode the load-order fix is
        designed to prevent.
        """
        cf = self._make_cloud_feedback()
        T0_init = cf.T0.copy()
        # Load a shifted state without refreshing T0 first
        cf.state['Tatm'][:] = cf.state['Tatm'] + 10.0
        cf.state['q'][:] = cf.state['q'] * 1.5
        # Cloud_dict access now computes dT = state - T0 != 0
        cd = cf.cloud_dict
        # At least one of the cloud fields must differ from mycloud0
        polluted = (
            not np.allclose(cd['cldfrac'], cf.mycloud0['cldfrac'], atol=1e-6)
            or not np.allclose(cd['clwp'], cf.mycloud0['clwp'], atol=1e-6)
            or not np.allclose(cd['ciwp'], cf.mycloud0['ciwp'], atol=1e-6)
        )
        assert polluted, (
            "CloudFeedback should show non-trivial cloud anomaly when "
            "T0 is stale relative to current state; the fact that it "
            "doesn't here suggests the sensitivity dataset mock has zero "
            "coefficients — adjust the test fixture."
        )
        # T0 should still be the initial value (not updated)
        np.testing.assert_allclose(cf.T0, T0_init)
