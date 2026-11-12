"""Polar continuation preserves supplied values and never borrows across rows."""
import numpy as np
import pytest
import xarray as xr
from climate_runs_ext.utils.polar_extension import FIELDS, extend_polar_tables


def test_independent_polar_donors_and_unchanged_interior():
    lat=[-90.,-86.,-84.,0.,84.,86.,90.]
    values=np.array([[[[np.nan,np.nan,2.,np.nan,3.,np.nan,np.nan],
                       [np.nan,np.nan,7.,8.,9.,np.nan,np.nan],
                       [np.nan]*7]]])
    ds=xr.Dataset({f'iso_mean_first_on_model_{k}':(('year','month','model_pressure','latitude'),values*(i+1)) for i,k in enumerate(FIELDS)},coords=dict(year=[2010],month=[1],model_pressure=[1000,2000,3000],latitude=lat))
    original=ds.copy(deep=True)
    result,counts=extend_polar_tables(ds)
    xr.testing.assert_identical(ds,original)
    for i,k in enumerate(FIELDS):
        a=result[f'iso_mean_first_on_model_{k}'].values[0,0]
        np.testing.assert_allclose(a[0],np.array([2,2,2,np.nan,3,3,3])*(i+1),equal_nan=True)
        np.testing.assert_allclose(a[1],np.array([7,7,7,8,9,9,9])*(i+1))
        assert np.isnan(a[2]).all()
        assert counts[k]==8
    with pytest.raises(ValueError,match='increasing'):
        extend_polar_tables(ds.isel(latitude=slice(None,None,-1)))
