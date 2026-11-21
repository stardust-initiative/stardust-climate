"""Regression checks for fixed-season illumination/coefficient consistency."""
import ast
from pathlib import Path

import numpy as np
import pytest
from climlab.utils import constants as const
from climlab.solar.insolation import daily_insolation

from climate_runs_ext.utils.era5_data import (
    Seasonal_insolation, seasonal_insolation_days,
)


@pytest.mark.parametrize('season', ['Annual', 'DJF', 'MAM', 'JJA', 'SON'])
def test_dates_reproduce_illumination(season):
    lat = np.array([-89., -61., 1., 61., 89.])
    days = seasonal_insolation_days(season)
    np.testing.assert_array_equal(
        Seasonal_insolation(lat, season),
        daily_insolation(lat, days).mean(axis=1),
    )
    assert np.isfinite(days).all()
    assert len(days) == (365 if season == 'Annual' else 93)


def test_djf_wrap_and_annual_grid():
    days = seasonal_insolation_days('DJF')
    assert days[0] == 11 * const.days_per_month
    assert days[31] == 0
    assert days[62] == const.days_per_month
    np.testing.assert_array_equal(
        seasonal_insolation_days('Annual'),
        np.linspace(0, const.days_per_year, 365),
    )
    # Polar winter illumination stays dark; no annual-illumination fallback.
    assert Seasonal_insolation(np.array([89.]), 'DJF')[0] == 0


@pytest.mark.parametrize('season', ['Annual', 'DJF', 'MAM', 'JJA', 'SON'])
def test_public_example_passes_shared_dates(season):
    path = Path(__file__).resolve().parents[1] / 'examples/climatic_response_annual_avg.py'
    tree = ast.parse(path.read_text())
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id in
             {'AerosolsOptDepTables', 'get_radiation_with_aerosols_params'}]
    assert len(calls) == 5  # transported, uniform and stabilisation paths
    for call in calls:
        value = next(k.value for k in call.keywords if k.arg == 'days')
        actual = eval(compile(ast.Expression(value), str(path), 'eval'),
                      {'coefficient_days': seasonal_insolation_days(season)})
        np.testing.assert_array_equal(actual, seasonal_insolation_days(season))
