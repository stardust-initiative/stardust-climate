"""Portable defaults and equivalence of calendar preparation on the actual grid."""

import climlab
import numpy as np
import pytest
import xarray as xr

from climate_runs_ext.utils.isentropic_runtime import (
    DEFAULT_DIFFUSION_SCHEME, prepare_climatology, source_path,
)
from climate_runs_ext.utils.transport_data import get_atm_data



def test_source_resolution_and_missing_file_do_not_fall_back_to_nominal(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv('STARDUST_ISENTROPIC_FILE', raising=False)
    with pytest.raises(FileNotFoundError, match='--isentropic-file'):
        source_path()
    f = tmp_path/'input.nc'
    f.touch()
    monkeypatch.setenv('STARDUST_ISENTROPIC_FILE', str(f))
    assert source_path() == f
    with pytest.raises(FileNotFoundError):
        source_path('explicit-missing.nc')


def test_raw_preparation_retains_tilt_cross_and_excludes_mixed_terms():
    state = {'q': climlab.column_state(lev=[10.,30.], lat=[-30.,0.,30.])['Tatm']*0}
    coords = dict(year=np.arange(2008,2018), month=np.arange(1,13),
                  model_pressure=[1.,1000.,3000.,100000.], latitude=[-90.,-30.,0.,30.,90.])
    shape = tuple(len(v) for v in coords.values())
    values = dict(D_yy=4., D_pp_tilt=4., D_pp_theta=-1., D_yp_tilt=.1,
                  D_yp_theta_cross=np.nan, D_pp_mixed=np.nan, D_yp=np.nan, D_pp=np.nan)
    source = xr.Dataset({f'iso_mean_first_on_model_{k}':
        xr.DataArray(np.full(shape,v), dims=tuple(coords), coords=coords)
        for k,v in values.items()})
    fields = prepare_climatology(source, state, 'test-hash')
    for key,value in [('kyy',4.),('kzz',3.),('kyz',.1)]:
        supported = fields[key].level >= .01
        np.testing.assert_allclose(fields[key].isel(level=supported), value)
        np.testing.assert_array_equal(fields[key].isel(level=~supported), 0.)
        audit=fields[key].attrs['diffusion_source_audit']
        assert audit['mapping']=='mean_first' and audit['selection']=='reduced'
        assert audit['source_sha256']=='test-hash'
        assert len(audit['fields'])==4
        assert not any('mixed' in f or 'theta_cross' in f for f in audit['fields'])


def test_raw_scheme_rejects_mismatched_time_settings_before_loading():
    with pytest.raises(ValueError, match='time_type=1'):
        get_atm_data({}, time_type=0, diffusion_scheme=DEFAULT_DIFFUSION_SCHEME)
    with pytest.raises(ValueError, match='state/grid'):
        get_atm_data({}, diffusion_scheme=DEFAULT_DIFFUSION_SCHEME)
