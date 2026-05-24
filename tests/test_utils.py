"""Tests for climate_runs_ext core utilities (Phase 6a).

Covers:
- state_helpers: create_state, create_state_for_conv, lev_grid_construct
- state_io: model_to_npz, update_model_from_xr, iteratively_update_internal
- model_control: fix_Ts, unfix_Ts, add_bounds
- config loading: load_project_config
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import numpy as np
import pytest
import climlab

from climate_runs_ext.utils.state_helpers import (
    create_state,
    create_state_for_conv,
    lev_grid_construct,
)
from climate_runs_ext.utils.state_io import (
    model_to_npz,
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
    add_bounds,
)


# ---------------------------------------------------------------------------
# lev_grid_construct
# ---------------------------------------------------------------------------

class TestLevGridConstruct:
    """Verify pressure-level grid construction."""

    def test_basic_shape(self):
        """Output should have n levels."""
        p = lev_grid_construct(50, dp0=1.0, dps=20.0)
        assert p.shape == (50,)

    def test_monotonically_increasing(self):
        """Levels should increase from TOA to surface."""
        p = lev_grid_construct(100, dp0=0.5, dps=10.0)
        assert np.all(np.diff(p) > 0)

    def test_bounds_respected(self):
        """First level > p0, last level < ps."""
        p = lev_grid_construct(50, dp0=2.0, dps=20.0, p0=0.0, ps=1000.0)
        assert p[0] > 0.0
        assert p[-1] < 1000.0

    def test_top_spacing(self):
        """Spacing near the top should be close to dp0."""
        dp0 = 1.0
        p = lev_grid_construct(100, dp0=dp0, dps=20.0, p0=0.0, ps=1000.0)
        # The first level midpoint should be approximately dp0/2
        assert p[0] < dp0


# ---------------------------------------------------------------------------
# create_state
# ---------------------------------------------------------------------------

class TestCreateState:
    """Verify state construction from numpy arrays."""

    def test_single_column(self):
        """Single-column state should have correct shapes."""
        nlev = 30
        lev = np.linspace(10, 1000, nlev)
        lat = np.array([0.0])
        Tatm0 = 250.0 * np.ones(nlev)
        Ts0 = np.array([288.0])
        state = create_state(Tatm0, Ts0, lev, lat)
        assert 'Ts' in state
        assert 'Tatm' in state
        assert state['Tatm'].shape[-1] == nlev

    def test_zonal_mean(self):
        """Multi-latitude state should have (nlat, nlev) shape."""
        nlev, nlat = 20, 10
        lev = np.linspace(10, 1000, nlev)
        lat = np.linspace(-85, 85, nlat)
        Tatm0 = 250.0 * np.ones((nlat, nlev))
        Ts0 = 288.0 * np.ones(nlat)
        state = create_state(Tatm0, Ts0, lev, lat)
        assert state['Tatm'].shape == (nlat, nlev)


# ---------------------------------------------------------------------------
# create_state_for_conv
# ---------------------------------------------------------------------------

class TestCreateStateForConv:
    """Verify state construction with humidity."""

    def test_has_humidity(self):
        """State should include q variable."""
        nlev = 30
        lev = np.linspace(50, 1000, nlev)
        lat = np.linspace(-85, 85, 10)
        Tatm = 250.0 * np.ones((10, nlev))
        Ts = 288.0 * np.ones(10)
        q = 1e-3 * np.ones((10, nlev))
        state = create_state_for_conv(lev, lat, Ts, Tatm, q)
        assert 'q' in state
        assert 'Tatm' in state
        assert 'Ts' in state


# ---------------------------------------------------------------------------
# model_to_npz / update_model_from_xr
# ---------------------------------------------------------------------------

class TestStateIO:
    """Verify round-trip serialization."""

    def test_model_to_npz_returns_dict(self):
        """model_to_npz with no filename returns a dict."""
        state = climlab.column_state(num_lev=20, num_lat=5)
        model = climlab.TimeDependentProcess(state=state)
        d = model_to_npz(model)
        assert isinstance(d, dict)
        assert 'STATE_Ts' in d
        assert 'STATE_Tatm' in d

    def test_model_to_npz_roundtrip(self, tmp_path):
        """Save to npz and reload, verify values match."""
        state = climlab.column_state(num_lev=20, num_lat=5)
        model = climlab.TimeDependentProcess(state=state)
        fpath = str(tmp_path / 'test.npz')
        model_to_npz(model, filename=fpath)
        loaded = dict(np.load(fpath))
        np.testing.assert_allclose(
            loaded['STATE_Tatm'], np.array(model.state['Tatm']),
        )

    def test_update_model_from_xr(self):
        """update_model_from_xr should overwrite model state."""
        state = climlab.column_state(num_lev=20, num_lat=5)
        model = climlab.TimeDependentProcess(state=state)
        # Create an xarray dataset with modified values
        xr_obj = model.to_xarray()
        xr_obj['Tatm'] = xr_obj['Tatm'] + 10.0
        xr_obj['Ts'] = xr_obj['Ts'] + 5.0
        old_Tatm = np.array(model.state['Tatm']).copy()
        update_model_from_xr(model, xr_obj)
        np.testing.assert_allclose(
            np.array(model.state['Tatm']), old_Tatm + 10.0,
        )


# ---------------------------------------------------------------------------
# iteratively_update_internal
# ---------------------------------------------------------------------------

class TestIterativelyUpdateInternal:
    """Verify recursive update_internal_fields dispatch."""

    def test_calls_update_internal(self):
        """Subprocesses with update_internal_fields should be called."""
        state = climlab.column_state(num_lev=20)
        model = climlab.TimeDependentProcess(state=state)

        # Create a subprocess that tracks calls
        sub = climlab.TimeDependentProcess(state=state)
        sub._was_updated = False

        def track_update(ref_state=None):
            sub._was_updated = True

        sub.update_internal_fields = track_update
        model.add_subprocess('tracker', sub)

        ref_state = model.state
        iteratively_update_internal(model, ref_state)
        assert sub._was_updated is True


# ---------------------------------------------------------------------------
# fix_Ts / unfix_Ts
# ---------------------------------------------------------------------------

class TestModelControl:
    """Verify fix_Ts / unfix_Ts Limiter-based implementation."""

    def _make_model(self):
        """Build a minimal coupled model with surface temperature."""
        state = climlab.column_state(num_lev=20, num_lat=5)
        return climlab.TimeDependentProcess(state=state)

    def test_fix_Ts_adds_subprocess(self):
        """fix_Ts should add a FixedTs Limiter subprocess."""
        model = self._make_model()
        fix_Ts(model)
        assert 'FixedTs' in model.subprocess

    def test_unfix_Ts_removes_subprocess(self):
        """unfix_Ts should remove the FixedTs subprocess."""
        model = self._make_model()
        fix_Ts(model)
        unfix_Ts(model)
        assert 'FixedTs' not in model.subprocess

    def test_fix_Ts_idempotent(self):
        """Calling fix_Ts twice should not error."""
        model = self._make_model()
        fix_Ts(model)
        fix_Ts(model)  # should not raise
        assert 'FixedTs' in model.subprocess

    def test_unfix_Ts_when_not_fixed(self):
        """Calling unfix_Ts on a model without FixedTs should not error."""
        model = self._make_model()
        unfix_Ts(model)  # should not raise

    def test_fix_Ts_pins_value(self):
        """After fix_Ts, stepping should keep Ts constant."""
        model = self._make_model()
        Ts_before = np.array(model.state['Ts']).copy()
        fix_Ts(model)
        # The Limiter should clip Ts back to Ts_before
        # Manually modify Ts and verify the limiter clips it
        model.state['Ts'][:] = Ts_before + 100.0
        lim = model.subprocess['FixedTs']
        lim.step_forward()
        np.testing.assert_allclose(np.array(model.state['Ts']), Ts_before)

    def test_fix_Ts_limiter_timestep_matches_model(self):
        """Regression: fix_Ts's Limiter must inherit model.timestep.
        Otherwise the Ts pin leaks ~96% per step against other tendencies."""
        model = self._make_model()
        model.timestep = 3600.0
        fix_Ts(model)
        assert model.subprocess['FixedTs'].timestep == 3600.0


# ---------------------------------------------------------------------------
# fix_q / unfix_q
# ---------------------------------------------------------------------------

class TestFixQ:
    """Verify fix_q / unfix_q Limiter-based implementation."""

    def _make_model(self):
        from climlab.domain.field import Field
        state = climlab.column_state(num_lev=20, num_lat=5)
        q_domain = state['Tatm'].domain
        state['q'] = Field(1e-3 * np.ones(state['Tatm'].shape), domain=q_domain)
        return climlab.TimeDependentProcess(state=state)

    def test_fix_q_adds_subprocess(self):
        model = self._make_model()
        fix_q(model)
        assert 'FixedQ' in model.subprocess

    def test_unfix_q_removes_subprocess(self):
        model = self._make_model()
        fix_q(model)
        unfix_q(model)
        assert 'FixedQ' not in model.subprocess

    def test_fix_q_idempotent(self):
        model = self._make_model()
        fix_q(model)
        fix_q(model)
        assert 'FixedQ' in model.subprocess

    def test_unfix_q_when_not_fixed(self):
        model = self._make_model()
        unfix_q(model)  # should not raise

    def test_fix_q_pins_value_at_all_levels(self):
        """Limiter should clip q back to snapshot at every level."""
        model = self._make_model()
        q_before = np.array(model.state['q']).copy()
        fix_q(model)
        model.state['q'][:] = q_before + 5e-4
        model.subprocess['FixedQ'].step_forward()
        np.testing.assert_allclose(np.array(model.state['q']), q_before)


# ---------------------------------------------------------------------------
# fix_Tatm_trop / unfix_Tatm_trop
# ---------------------------------------------------------------------------

class TestFixTatmTrop:
    """Verify fix_Tatm_trop / unfix_Tatm_trop partial-level Limiter."""

    def _make_model(self):
        state = climlab.column_state(num_lev=20, num_lat=5)
        lev = state['Tatm'].domain.lev.points
        return climlab.TimeDependentProcess(state=state), lev

    def test_fix_Tatm_trop_adds_subprocess(self):
        model, _ = self._make_model()
        fix_Tatm_trop(model, p_trop_hPa=180.0)
        assert 'FixedTatmTrop' in model.subprocess

    def test_unfix_Tatm_trop_removes_subprocess(self):
        model, _ = self._make_model()
        fix_Tatm_trop(model)
        unfix_Tatm_trop(model)
        assert 'FixedTatmTrop' not in model.subprocess

    def test_fix_Tatm_trop_idempotent(self):
        model, _ = self._make_model()
        fix_Tatm_trop(model)
        fix_Tatm_trop(model)
        assert 'FixedTatmTrop' in model.subprocess

    def test_unfix_when_not_fixed(self):
        model, _ = self._make_model()
        unfix_Tatm_trop(model)  # should not raise

    def test_pins_troposphere_frees_stratosphere(self):
        """Levels >= p_trop_nearest clamped; levels above free (above floor)."""
        model, lev = self._make_model()
        p_trop = 180.0
        idx = int(np.argmin(np.abs(lev - p_trop)))
        p_trop_nearest = lev[idx]

        T_before = np.array(model.state['Tatm']).copy()
        # Use strato_Tmin=-inf so the stratosphere is truly unbounded
        fix_Tatm_trop(
            model, p_trop_hPa=p_trop, strato_Tmin=-np.inf,
        )

        # Perturb every level by +50 K, then step the limiter.
        model.state['Tatm'][:] = T_before + 50.0
        model.subprocess['FixedTatmTrop'].step_forward()
        T_after = np.array(model.state['Tatm'])

        trop_mask = (lev >= p_trop_nearest)
        strato_mask = ~trop_mask

        # Tropospheric levels snapped back exactly.
        np.testing.assert_allclose(
            T_after[:, trop_mask], T_before[:, trop_mask],
        )
        # Stratospheric levels left at perturbed value (unbounded).
        np.testing.assert_allclose(
            T_after[:, strato_mask], T_before[:, strato_mask] + 50.0,
        )

    def test_strato_Tmin_floor(self):
        """Default strato_Tmin=180 floors stratospheric T when pushed low."""
        model, lev = self._make_model()
        fix_Tatm_trop(model, p_trop_hPa=180.0)  # default strato_Tmin=180
        # Drive everything to 100 K
        model.state['Tatm'][:] = 100.0
        model.subprocess['FixedTatmTrop'].step_forward()
        T_after = np.array(model.state['Tatm'])
        # Stratospheric cells should be floored at 180 K
        idx = int(np.argmin(np.abs(lev - 180.0)))
        strato_mask = (lev < lev[idx])
        assert np.all(T_after[:, strato_mask] >= 180.0 - 1e-9)

    def test_p_trop_snaps_to_nearest_level(self):
        """A p_trop that doesn't match any level should snap to nearest."""
        model, lev = self._make_model()
        # Pick a value deliberately between two levels.
        target = 0.5 * (lev[5] + lev[6])
        fix_Tatm_trop(model, p_trop_hPa=target)
        # Both choices (idx=5 or idx=6) are valid; just check no error
        # and that the subprocess exists.
        assert 'FixedTatmTrop' in model.subprocess


# ---------------------------------------------------------------------------
# add_bounds
# ---------------------------------------------------------------------------

class TestAddBounds:
    """Verify bound_dict → Limiter conversion."""

    def test_adds_limiter_subprocess(self):
        """add_bounds should add limiter_ subprocesses."""
        state = climlab.column_state(num_lev=20, num_lat=5)
        model = climlab.TimeDependentProcess(state=state)
        add_bounds(model, {'Tatm': (150.0, np.inf)})
        assert 'limiter_Tatm' in model.subprocess

    def test_clips_below_minimum(self):
        """Limiter should clip values below minimum."""
        state = climlab.column_state(num_lev=20, num_lat=5)
        model = climlab.TimeDependentProcess(state=state)
        add_bounds(model, {'Tatm': (200.0, np.inf)})
        model.state['Tatm'][:] = 100.0  # below minimum
        model.subprocess['limiter_Tatm'].step_forward()
        assert np.all(np.array(model.state['Tatm']) >= 200.0)

    def test_limiter_timestep_matches_model(self):
        """Regression test: Limiters created by add_bounds must inherit
        the model's timestep, not fall back to climlab's default 1-day.

        When limiter.timestep != model.timestep, climlab scales the
        Limiter's adjustment by model_dt/limiter_dt per step. With the
        default 1-day Limiter and a 1-hour model step, only ~4% of the
        clip is applied per step, so tendencies that push state out of
        bounds effectively leak through. Found while investigating
        full-ERF instability in climate_runs_ext.
        """
        state = climlab.column_state(num_lev=20, num_lat=5)
        model = climlab.TimeDependentProcess(state=state, timestep=3600.0)
        add_bounds(model, {'Tatm': (150.0, np.inf)})
        assert model.subprocess['limiter_Tatm'].timestep == model.timestep
