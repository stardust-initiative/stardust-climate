import numpy as np
import xarray as xr
from climate_runs_ext.utils.isentropic_climatology import monthly_tensor


def source():
    coords=dict(year=[2010],month=[1,2],model_pressure=[100.,200.],latitude=[-1.,1.])
    values={'D_yy':4.,'D_pp':3.,'D_yp':1.,'D_pp_tilt':4.,'D_pp_theta':-1.,'D_yp_tilt':1.}
    return xr.Dataset({f'iso_{m}_on_model_{k}':xr.DataArray(np.full((1,2,2,2),v),dims=tuple(coords),coords=coords)
        for m in ('mean_first','transform_first') for k,v in values.items()})


def test_missing_month_zeroes_whole_tensor_before_averaging():
    d=source();d.iso_transform_first_on_model_D_yp.values[0,0,0,0]=np.nan
    f,a=monthly_tensor(d,selection='full')
    assert all(float(v.isel(month=0,level=0,latitude=0))==0 for v in f.values())
    assert float(f['kyy'].mean('month').isel(level=0,latitude=0))==2.
    assert a['monthly_fields']==2 and a['undefined_monthly_cells']==1


def test_reduced_assembles_before_diagonal_clipping():
    d=source();f,_=monthly_tensor(d)
    np.testing.assert_array_equal(f['kzz'],3.)
    np.testing.assert_array_equal(f['kyy'],4.)


def test_mapping_selection_and_psd_bound():
    d=source();d.iso_mean_first_on_model_D_yp.values[:]=100.
    f,_=monthly_tensor(d,mapping='mean_first',selection='full')
    np.testing.assert_allclose(f['kyz'],np.sqrt(12.))
    g,_=monthly_tensor(d,mapping='transform_first',selection='full')
    np.testing.assert_array_equal(g['kyz'],1.)
