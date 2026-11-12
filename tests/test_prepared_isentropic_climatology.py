import json
from datetime import datetime
import climlab
import numpy as np
import pytest
import xarray as xr
from climlab_stardust_extension.dynamics.atmospheric_data import AtmosphericData
from climate_runs_ext.utils.prepared_isentropic_climatology import AXES,calendar_mean,decode,validate_grid


def prepared():
    state={'q':climlab.column_state(lev=[10.,30.],lat=[-30.,0.,30.])['Tatm']*0}
    domain=state['q'].domain
    lb=domain.axes['lat'].bounds;pb=domain.axes['lev'].bounds
    coords=dict(month=np.arange(1,13),lat_bound=lb,lat_center=(lb[:-1]+lb[1:])/2,
                lev_bound=pb,lev_center=(pb[:-1]+pb[1:])/2)
    ds=xr.Dataset(coords=coords,attrs={'preparation_json':json.dumps(dict(mapping='transform_first',selection='reduced',years=list(range(2008,2018))))})
    for key,value in [('kyy',4.),('kzz',1.),('kyz',.1)]:
        a,b=AXES[key];ds[key]=(('month',a,b),np.full((12,len(coords[a]),len(coords[b])),value))
    return state,ds


def test_calendar_mean_includes_zero_filled_missing_months():
    data=xr.DataArray(np.r_[np.ones(12)*4,np.zeros(12)],dims='month')
    np.testing.assert_array_equal(calendar_mean(data,np.tile(np.arange(1,13),2)),2.)


def test_roundtrip_on_identical_grid_does_not_blur_or_rescale():
    state,ds=prepared();f=decode(ds);validate_grid(f,state)
    specs={k:dict(data=v,grid_type={'kyy':'bounds*centers','kzz':'centers*bounds','kyz':'bounds*bounds'}[k]) for k,v in f.items()}
    atm=AtmosphericData(param_configs=specs,state=state,time_type=1,t_0=datetime(2025,1,1),symmetric_diffusion=True)
    for key,value in [('kyy',4.),('kzz',1.)]:np.testing.assert_array_equal(atm._current_data[key],value)


def test_prepared_grid_mismatch_is_rejected():
    _,ds=prepared();state={'q':climlab.column_state(lev=[11.,31.],lat=[-30.,0.,30.])['Tatm']*0}
    with pytest.raises(AssertionError,match='pressure grid mismatch'):validate_grid(decode(ds),state)


def test_mean_first_mapping_is_explicit_and_preserves_cross():
    from climate_runs_ext.utils.prepared_isentropic_climatology import MEAN_FIRST_SCHEME
    state,ds=prepared()
    with pytest.raises(AssertionError):decode(ds,MEAN_FIRST_SCHEME)
    audit=json.loads(ds.attrs['preparation_json']);audit['mapping']='mean_first'
    ds.attrs['preparation_json']=json.dumps(audit)
    fields=decode(ds,MEAN_FIRST_SCHEME);validate_grid(fields,state)
    np.testing.assert_array_equal(fields['kyz'],.1)
    assert fields['kyz'].attrs['diffusion_source_audit']['mapping']=='mean_first'
    with pytest.raises(AssertionError):decode(ds)
