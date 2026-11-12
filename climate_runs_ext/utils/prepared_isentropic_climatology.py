"""Grid-pinned monthly climatology of individually supported saved tensors.

Preparation reuses the maintained AtmosphericData spatial interpolation for
each year-month, then averages matching calendar months. Runtime interpolation
is an identity in space on the declared grid; face bounds and time evolution
remain in the maintained AtmosphericData/transport components.
"""
import json
import numpy as np
import xarray as xr
from stardust_2d_inputs.core.loader import load

SCHEME='isentropic_transform_first_reduced_2008_2017'
KEY='isentropic_transform_first_reduced_2008_2017_monthly_90x50'
MEAN_FIRST_SCHEME='isentropic_mean_first_reduced_2008_2017'
MEAN_FIRST_KEY='isentropic_mean_first_reduced_2008_2017_monthly_90x50'
SCHEMES={SCHEME:('transform_first',KEY),MEAN_FIRST_SCHEME:('mean_first',MEAN_FIRST_KEY)}
AXES={'kyy':('lat_bound','lev_center'),'kzz':('lat_center','lev_bound'),'kyz':('lat_bound','lev_bound')}


def calendar_mean(data,months):
    """Mean zero-filled, already spatially interpolated samples by month."""
    assert len(months)==data.sizes['month'] and np.isfinite(data.values).all()
    a=data.assign_coords(calendar_month=('month',months)).groupby('calendar_month').mean('month',skipna=False)
    return a.rename(calendar_month='month')


def decode(ds,scheme=SCHEME):
    mapping,dataset_key=SCHEMES[scheme]
    np.testing.assert_array_equal(ds.month.values,np.arange(1,13))
    audit=json.loads(ds.attrs['preparation_json'])
    assert audit['mapping']==mapping and audit['selection']=='reduced'
    assert audit['years']==list(range(2008,2018))
    result={}
    for key,(lat,p) in AXES.items():
        a=ds[key].rename({lat:'latitude',p:'level'}).transpose('month','level','latitude').load()
        assert np.isfinite(a.values).all()
        if key!='kyz':assert (a.values>=0).all()
        a.attrs.update(diffusion_tensor_component=key,diffusion_pregridded=True,
            diffusion_source_audit=dict(audit,scheme=scheme,dataset=dataset_key,
                spatial_runtime='Identity on pinned target grid; reject a different simulation grid'))
        result[key]=a.assign_coords(tensor_valid=(a.dims,np.ones(a.shape)))
    return result


def load_climatology(scheme=SCHEME):
    with load(SCHEMES[scheme][1]) as ds:fields=decode(ds,scheme)
    return fields['kzz'],fields['kyy'],fields['kyz']


def validate_grid(fields,state):
    domain=next(iter(state.values())).domain
    lat_bound=domain.axes['lat'].bounds;lev_bound=domain.axes['lev'].bounds
    coordinates=dict(lat_bound=lat_bound,lat_center=(lat_bound[1:]+lat_bound[:-1])/2,
                     lev_bound=lev_bound,lev_center=(lev_bound[1:]+lev_bound[:-1])/2)
    for key,(lat,p) in AXES.items():
        np.testing.assert_array_equal(fields[key].latitude.values,coordinates[lat],err_msg='Prepared diffusion latitude grid mismatch')
        np.testing.assert_array_equal(fields[key].level.values,coordinates[p],err_msg='Prepared diffusion pressure grid mismatch')
