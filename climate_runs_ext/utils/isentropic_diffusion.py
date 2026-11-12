"""Assemble explicit reduced or full transform-first pressure-coordinate tensors."""
import numpy as np
import xarray as xr
from stardust_2d_inputs.core.loader import load

SCHEME='isentropic_transform_first_reduced'
FULL_SCHEME='isentropic_transform_first_full'
PREFIX='iso_transform_first_on_model_'
FIELDS={'Dyy':'D_yy','Dyp':'D_yp_tilt','Dpp_tilt':'D_pp_tilt','Dpp_theta':'D_pp_theta'}


def load_tensor(selection="reduced"):
    """Return Kpp, Kyy, physical Dyp and a source-policy audit.

    Source negatives are clipped on the ASSEMBLED diagonal. Undefined values
    in any selected term zero the entire tensor. No SNR filter or positive
    floor is imposed. Saved products are used without multiplying mean slopes.
    """
    if selection not in ('reduced','full'):
        raise ValueError(f'Unknown tensor selection: {selection}')
    fields=FIELDS if selection=='reduced' else {'Dyy':'D_yy','Dyp':'D_yp','Dpp':'D_pp'}
    arrays=[];audits=[]
    for month in range(1,13):
        with load(f'isentropic_diffusion_2010_{month:02d}') as ds:
            raw={key:ds[PREFIX+name].transpose('model_pressure','latitude').values.astype(float) for key,name in fields.items()}
            lat=ds.latitude.values.copy();pressure=ds.model_pressure.values.copy()/100
        if arrays:
            np.testing.assert_array_equal(lat,latitude)
            np.testing.assert_array_equal(pressure,levels)
        latitude=lat;levels=pressure
        yy=raw['Dyy'];pp=raw['Dpp_tilt']+raw['Dpp_theta'] if selection=='reduced' else raw['Dpp'];yp=raw['Dyp']
        valid=np.logical_and.reduce([np.isfinite(a) for a in raw.values()])
        ypositive=np.maximum(np.where(valid,yy,0),0)
        ppositive=np.maximum(np.where(valid,pp,0),0)
        bound=np.sqrt(ypositive*ppositive)
        cross=np.clip(np.where(valid,yp,0),-bound,bound)
        arrays.append((ypositive,ppositive,cross,valid.astype(float)))
        audits.append(dict(month=month,undefined_cells=int((~valid).sum()),
            negative_Dyy_cells=int((valid&(yy<0)).sum()),negative_assembled_Dpp_cells=int((valid&(pp<0)).sum()),
            clipped_cross_cells=int((valid&(abs(yp)>bound)).sum())))
    stacked=np.asarray(arrays);coords=dict(month=np.arange(1,13),level=levels,latitude=latitude)
    result={}
    for index,key,units in [(0,'kyy','m2/s'),(1,'kzz','Pa2/s'),(2,'kyz','m Pa/s')]:
        field=xr.DataArray(stacked[:,index],dims=['month','level','latitude'],coords=coords,name=key,
            attrs=dict(units=units,diffusion_tensor_component=key))
        result[key]=field.assign_coords(tensor_valid=(('month','level','latitude'),stacked[:,3]))
    audit=dict(scheme=SCHEME if selection=='reduced' else FULL_SCHEME,prefix=PREFIX,exact_fields={k:PREFIX+v for k,v in fields.items()},
        Dpp_recipe='D_pp_tilt + D_pp_theta, assembled before diagonal clipping' if selection=='reduced' else 'Stored full D_pp, including tilt + theta + mixed',
        excluded=[PREFIX+'D_yp_theta_cross',PREFIX+'D_pp_mixed'] if selection=='reduced' else [],source_months=audits,
        support_hPa=[float(levels.min()),float(levels.max())],
        undefined_policy='Zero whole tensor; require complete interpolation stencil; no extrapolation',
        negative_policy='Clip assembled diagonals to zero; bound signed cross by sqrt(Dyy*Dpp)',
        noise_mask=False,positive_floor=False,antisymmetric_part_used=False)
    for field in result.values():field.attrs['diffusion_source_audit']=audit
    return result['kzz'],result['kyy'],result['kyz']


def load_reduced_tensor():
    """Backward-compatible reduced-tensor entry point."""
    return load_tensor('reduced')
