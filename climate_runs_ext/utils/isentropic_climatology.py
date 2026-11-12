"""Assemble signed saved products into supported monthly symmetric tensors.

Monthly missing tensors become zero before a static transport climatology is
formed. This differs deliberately from a raw diagnostic mean requiring every
month to be finite. Physics/interpolation/face bounds remain in the extension.
"""
import numpy as np
import xarray as xr


def monthly_tensor(ds, mapping='transform_first', selection='reduced', years=None):
    if mapping not in ('mean_first','transform_first') or selection not in ('full','reduced'):
        raise ValueError((mapping,selection))
    source = ds if years is None else ds.sel(year=list(years))
    from .polar_extension import extend_polar_tables
    source, polar_counts = extend_polar_tables(source, mapping)
    prefix = f'iso_{mapping}_on_model_'
    names = ['D_yy','D_pp','D_yp'] if selection == 'full' else ['D_yy','D_pp_tilt','D_pp_theta','D_yp_tilt']
    raw = {k:source[prefix+k].transpose('year','month','model_pressure','latitude').values.astype(float) for k in names}
    valid = np.logical_and.reduce([np.isfinite(v) for v in raw.values()])
    yy = raw['D_yy']
    pp = raw['D_pp'] if selection == 'full' else raw['D_pp_tilt'] + raw['D_pp_theta']
    yp = raw['D_yp'] if selection == 'full' else raw['D_yp_tilt']
    ypositive = np.maximum(np.where(valid,yy,0),0)
    ppositive = np.maximum(np.where(valid,pp,0),0)
    bound = np.sqrt(ypositive*ppositive)
    cross = np.clip(np.where(valid,yp,0),-bound,bound)
    n = source.sizes['year']*source.sizes['month']
    coords = dict(month=np.arange(1,n+1),level=source.model_pressure.values/100.,latitude=source.latitude.values)
    shape = (n,source.sizes['model_pressure'],source.sizes['latitude'])
    fields = {}
    for key,values,units in [('kyy',ypositive,'m2/s'),('kzz',ppositive,'Pa2/s'),('kyz',cross,'m Pa/s')]:
        field = xr.DataArray(values.reshape(shape),dims=('month','level','latitude'),coords=coords,
                             attrs=dict(units=units,diffusion_tensor_component=key))
        fields[key] = field.assign_coords(tensor_valid=(('month','level','latitude'),valid.astype(float).reshape(shape)))
    audit = dict(mapping=mapping,selection=selection,prefix=prefix,fields=[prefix+k for k in names],
        years=source.year.values.tolist(),calendar_months=source.month.values.tolist(),monthly_fields=n,
        undefined_monthly_cells=int((~valid).sum()),negative_Dyy_monthly_cells=int((valid&(yy<0)).sum()),
        negative_Dpp_monthly_cells=int((valid&(pp<0)).sum()),cross_clipped_monthly_cells=int((valid&(abs(yp)>bound)).sum()),
        support_policy='Any required monthly NaN zeros the whole monthly tensor; complete spatial stencil; no extrapolation.',
        averaging_policy='Each monthly tensor is assembled, diagonals clipped and cross bounded; interpolate monthly tensors to model grid, then equal-weight average all months; bound annual staggered cross again.',
        antisymmetric_used=False,positive_floor=False,noise_mask=False, polar_continuation='nearest finite latitude per component at abs(latitude)>=85; exterior missing tails only', polar_filled_cells=polar_counts)
    return fields,audit
