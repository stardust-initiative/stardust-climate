"""Test the reduced recipe using signed and undefined saved contributions."""
from contextlib import contextmanager
import numpy as np
import xarray as xr
import pytest
from climate_runs_ext.utils import isentropic_diffusion as inputs


def test_assemble_before_clipping_and_zero_entire_undefined_tensor(monkeypatch):
    @contextmanager
    def load(key):
        coords=dict(model_pressure=[100.,1000.],latitude=[-10.,10.])
        values={'D_yy':[[2.,-1.],[2.,2.]],'D_yp_tilt':[[6.,4.],[4.,4.]],
                'D_pp_tilt':[[-4.,1.],[1.,1.]],'D_pp_theta':[[5.,1.],[1.,np.nan]]}
        yield xr.Dataset({inputs.PREFIX+k:(('model_pressure','latitude'),v) for k,v in values.items()},coords=coords)
    monkeypatch.setattr(inputs,'load',load)
    pp,yy,yp=inputs.load_reduced_tensor()
    # Clipping each retained Dpp contribution separately would incorrectly give5.
    assert float(pp.isel(month=0,level=0,latitude=0))==1
    assert float(yp.isel(month=0,level=0,latitude=0))==pytest.approx(np.sqrt(2))
    for a in [pp,yy,yp]:
        assert float(a.isel(month=0,level=1,latitude=1))==0
    assert float(yy.isel(month=0,level=0,latitude=1))==0
    assert float(yp.isel(month=0,level=0,latitude=1))==0
    np.testing.assert_array_equal(pp.level,[1.,10.])


def test_full_uses_stored_full_fields_instead_of_reduced_contributions(monkeypatch):
    @contextmanager
    def load(key):
        coords=dict(model_pressure=[100.,1000.],latitude=[-10.,10.])
        values={'D_yy':np.full((2,2),4.),'D_yp':np.full((2,2),2.),'D_pp':np.full((2,2),8.)}
        yield xr.Dataset({inputs.PREFIX+k:(('model_pressure','latitude'),v) for k,v in values.items()},coords=coords)
    monkeypatch.setattr(inputs,'load',load)
    pp,yy,yp=inputs.load_tensor('full')
    np.testing.assert_array_equal(pp,8);np.testing.assert_array_equal(yp,2)
    audit=yy.attrs['diffusion_source_audit']
    assert audit['scheme']==inputs.FULL_SCHEME and audit['excluded']==[]
    assert audit['exact_fields']['Dyp']==inputs.PREFIX+'D_yp'
    assert audit['exact_fields']['Dpp']==inputs.PREFIX+'D_pp'
