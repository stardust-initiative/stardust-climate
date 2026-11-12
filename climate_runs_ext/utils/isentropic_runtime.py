"""Portable mean-first reduced diffusion from the unmodified ERA5 product.

Use the same per-month spatial preparation and calendar averaging as Figure
7a, on the actual simulation grid. No VM paths or prepared-data registration
are needed; the existing input engine still supplies the prescribed winds.
"""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import xarray as xr

from .isentropic_climatology import monthly_tensor
from .prepared_isentropic_climatology import (
    AXES, MEAN_FIRST_SCHEME, calendar_mean, decode, validate_grid,
)

DEFAULT_DIFFUSION_SCHEME = 'isentropic_mean_first_reduced'
SOURCE_FILENAME = 'era5_iso_diffusion_2008_2017_monthly_reduced.nc'
REFERENCE_SHA256 = '455d998741419fd6edb0d9593a2feb5be5a0643802cab1cdf89c0c5ef10fb16c'
DATASET_KEY = 'isentropic_mean_first_reduced_2008_2017_monthly'


def source_path(path=None):
    """Explicit filename, environment variable, then the working directory."""
    filename = path or os.environ.get('STARDUST_ISENTROPIC_FILE') or SOURCE_FILENAME
    result = Path(filename).expanduser().resolve()
    if not result.is_file():
        raise FileNotFoundError(
            f'Isentropic diffusion file not found: {result}. Supply '
            '--isentropic-file FILE or set STARDUST_ISENTROPIC_FILE. '
            'Use --diffusion-scheme nominal for the previous coefficients.'
        )
    return result


def prepare_climatology(source, state, source_sha256=None):
    """Assemble, spatially interpolate, then average ten calendar-month samples.

    Negative assembled diagonals are clipped; a missing required contributor
    zeros its entire tensor. The maintained component applies complete-stencil
    support and PSD bounds. Unsupported regions have zero diffusion. Kyp is
    retained, without theta-cross; Kpp excludes the mixed contribution.
    """
    from climlab_stardust_extension.dynamics.atmospheric_data import AtmosphericData

    np.testing.assert_array_equal(source.year.values, np.arange(2008, 2018))
    np.testing.assert_array_equal(source.month.values, np.arange(1, 13))
    fields, audit = monthly_tensor(source, 'mean_first', 'reduced')
    grids = {'kyy': 'bounds*centers', 'kzz': 'centers*bounds', 'kyz': 'bounds*bounds'}
    atmosphere = AtmosphericData(
        param_configs={k: dict(data=v, method='linear', grid_type=grids[k])
                       for k, v in fields.items()},
        time_type=0, t_0=datetime(2024, 1, 1), symmetric_diffusion=True,
        state=state, timestep=43200.,
    )
    domain = next(iter(state.values())).domain
    lat = domain.axes['lat'].bounds
    pressure = domain.axes['lev'].bounds
    prepared = xr.Dataset(coords=dict(
        month=np.arange(1, 13), lat_bound=lat,
        lat_center=(lat[:-1]+lat[1:])/2, lev_bound=pressure,
        lev_center=(pressure[:-1]+pressure[1:])/2,
    ))
    for key, (latitude_axis, pressure_axis) in AXES.items():
        data = calendar_mean(atmosphere._param_configs[key]['data'],
                             np.tile(np.arange(1, 13), 10))
        prepared[key] = xr.DataArray(
            data.values, dims=('month', latitude_axis, pressure_axis),
            attrs=dict(units=fields[key].attrs['units']),
        )
    audit.update(
        averaging_policy='Assemble/clip/zero each year-month; complete-stencil spatial interpolation; equal-weight mean of ten realizations of each calendar month.',
        source_sha256=source_sha256, source_filename=SOURCE_FILENAME,
        positive_floor=False, noise_mask=False, upper_level_damping=False,
        spatial_runtime='Validated identity on the actual simulation grid',
    )
    prepared.attrs['preparation_json'] = json.dumps(audit)
    result = decode(prepared, MEAN_FIRST_SCHEME)
    for field in result.values():
        provenance = field.attrs['diffusion_source_audit']
        provenance.update(scheme=DEFAULT_DIFFUSION_SCHEME,
                          dataset='unmodified local ERA5 NetCDF; prepared in memory')
    validate_grid(result, state)
    return result


def load_climatology(state, path=None):
    # Production uses the pinned input engine. Explicit overrides remain for
    # the portable handoff and controlled comparisons, and are labelled as such.
    resolution = {'mode': 'explicit-file'}
    if path or os.environ.get('STARDUST_ISENTROPIC_FILE'):
        filename = source_path(path)
    else:
        from stardust_2d_inputs.core.config import load_config
        from stardust_2d_inputs.core.loader import Loader
        loader = Loader.from_config(load_config())
        entry, via = loader.resolve(DATASET_KEY)
        if via != 'release':
            raise ValueError('Isentropic diffusion must be pinned by the input release')
        filename = loader.fetch_path(DATASET_KEY)
        resolution = dict(mode='input-engine', key=DATASET_KEY,
                          version=entry.version, content_hash=entry.content_hash,
                          status=entry.status, release=loader.default_release)
    digest = hashlib.sha256()
    with filename.open('rb') as handle:
        for block in iter(lambda: handle.read(8*1024**2), b''):
            digest.update(block)
    with xr.open_dataset(filename) as source:
        fields = prepare_climatology(source, state, digest.hexdigest())
    for field in fields.values():
        field.attrs['diffusion_source_audit']['input_resolution'] = resolution
    return fields['kzz'], fields['kyy'], fields['kyz']
