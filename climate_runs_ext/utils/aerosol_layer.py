"""File-based aerosol layer construction.

Ported from the AGU2025 workflow (``climate_runs/examples/AGU2025/
calculate_ERF_grid.py``): load a multi-bin aerosol state produced by a
transport model and turn it into a list of ``aerosol_instance`` objects on
the target model grid, ready to be passed to
``AerosolsOptDepTables``.

Expected on-disk format
-----------------------

The caller provides two files:

1. **State netCDF** (``state_xr.nc``) — one data variable per particle
   size bin, each with dimensions ``(lat, lev)`` and units of mass
   mixing ratio (kg aerosol per kg air).  The file must also expose
   coordinate arrays ``lat`` (degrees) and ``lev`` (hPa).
2. **Radius mapping npz** (``radius_mapping.npz``) — maps each bin's
   variable name to its representative particle radius in metres.
   Example::

       {'Si_1': 2.5e-7, 'Si_2': 5.0e-7, ...}

Bin variables present in the state file but absent from the mapping are
ignored; bin names present in the mapping but absent from the state file
raise an error.

Usage
-----
::

    from climate_runs_ext.utils.aerosol_layer import load_multi_bin_aerosol_state

    result = load_multi_bin_aerosol_state(
        state_path='path/to/state_xr.nc',
        radius_mapping_path='path/to/radius_mapping.npz',
        material_name='silica',
        rho_particle=2196.0,
        domain=model_ref.Tatm.domain,
    )
    # result.aerosol_instance_list -> feed into AerosolsOptDepTables
    # result.total_mass_Tg          -> diagnostic (scalar)
    # result.avg_diameter_m         -> mass-weighted diagnostic (scalar)
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

from dataclasses import dataclass
from typing import List

import numpy as np
import xarray as xr
from scipy import interpolate
from scipy import constants as sc_const

from climlab import constants as const
from climlab_stardust_extension.radiation.optical_depth_tables_aerosols import (
    aerosol_instance,
)

# Mass of one molecule of dry air (kg).  climlab stores molecular_weight in
# g/mol as ``const.molecular_weight['dry air'] = 28.97``; dividing by
# Avogadro's number and converting g → kg gives the per-molecule mass.
_M_AIR_PER_MOLECULE_KG = (
    1e-3 * const.molecular_weight['dry air'] / sc_const.Avogadro
)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class MultiBinAerosolLayer:
    """Loaded multi-bin aerosol state, interpolated to the model grid.

    Attributes
    ----------
    aerosol_instance_list : list
        One ``aerosol_instance`` per size bin, ready to be passed to
        ``AerosolsOptDepTables(aerosol_instance_list=...)``.
    total_mass_Tg : float
        Sum of the column-integrated, globally-averaged mass across all
        bins, in teragrams.  Computed from the *original* (uninterpolated)
        mmr fields so it matches whatever total the transport run
        produced.
    avg_diameter_m : float
        Mass-weighted mean particle diameter (2 × r) across all bins, in
        metres.  Equals the single bin's diameter when there is only one
        bin.
    bin_names : list of str
        Names of the bins loaded from the state file (in insertion order
        from the radius_mapping).
    bin_radii_m : list of float
        Corresponding particle radii in metres.
    bin_masses_Tg : list of float
        Per-bin column-integrated total mass in teragrams.
    """

    aerosol_instance_list: List[object]
    total_mass_Tg: float
    avg_diameter_m: float
    bin_names: List[str]
    bin_radii_m: List[float]
    bin_masses_Tg: List[float]


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def _load_radius_mapping(path):
    """Load a bin_name → radius_m mapping from an ``.npz`` file."""
    raw = np.load(path)
    return {name: float(raw[name]) for name in raw.files}


def _interp_mmr_to_grid(mmr_da, model_lat, model_lev):
    """Linearly interpolate an ``(lat, lev)`` mmr field to the model grid.

    Values outside the source grid are extrapolated (``fill_value=None``)
    and then clipped to be non-negative.
    """
    # Normalize axis order to ('lat', 'lev')
    if mmr_da.dims == ('lev', 'lat'):
        mmr_da = mmr_da.transpose('lat', 'lev')
    elif mmr_da.dims != ('lat', 'lev'):
        raise ValueError(
            f"mmr field has unexpected dims {mmr_da.dims}; "
            "expected ('lat', 'lev') or ('lev', 'lat')"
        )

    lat_src = np.asarray(mmr_da.lat.values)
    lev_src = np.asarray(mmr_da.lev.values)
    # Ensure ascending axes for RegularGridInterpolator
    if np.diff(lat_src).mean() < 0:
        lat_src = lat_src[::-1]
        vals = mmr_da.values[::-1, :]
    else:
        vals = mmr_da.values
    if np.diff(lev_src).mean() < 0:
        lev_src = lev_src[::-1]
        vals = vals[:, ::-1]

    interp = interpolate.RegularGridInterpolator(
        (lat_src, lev_src), vals,
        bounds_error=False, fill_value=None, method='linear',
    )

    lat_mat, lev_mat = np.meshgrid(model_lat, model_lev, indexing='ij')
    coords = np.column_stack([lat_mat.ravel(), lev_mat.ravel()])
    mmr_interp = interp(coords).reshape(lat_mat.shape)
    return np.where(mmr_interp > 0.0, mmr_interp, 0.0)


def _cosine_lat_avg(field_1d, lat_deg):
    """Cosine-weighted latitude average of a 1-D field."""
    w = np.cos(np.deg2rad(lat_deg))
    return float(np.average(field_1d, weights=w))


def load_multi_bin_aerosol_state(
    state_path,
    radius_mapping_path,
    material_name,
    rho_particle,
    domain,
):
    """Load a multi-bin aerosol state and prepare aerosol_instance objects.

    Parameters
    ----------
    state_path : str
        Path to the netCDF file written by the transport model.  Expected
        to contain per-bin ``mmr(lat, lev)`` variables plus ``lat`` and
        ``lev`` coordinates.
    radius_mapping_path : str
        Path to the ``.npz`` file containing the mapping of bin variable
        names to particle radii in metres.
    material_name : str
        Optical-property key (e.g. ``'silica'``) passed to
        ``aerosol_instance`` for each bin.  All bins share the same
        material.
    rho_particle : float
        Bulk particle density in kg/m^3 — used to derive the particle
        mass ``m_p = (4/3) π r_m^3 ρ``.
    domain : climlab Domain
        Target atmospheric domain (``state.Tatm.domain``) supplying
        ``lat.points`` and ``lev.points`` for grid interpolation.

    Returns
    -------
    MultiBinAerosolLayer
        Populated result object (see class docstring).
    """
    radius_mapping = _load_radius_mapping(radius_mapping_path)

    model_lat = np.asarray(domain.lat.points)
    model_lev = np.asarray(domain.lev.points)

    m_air = _M_AIR_PER_MOLECULE_KG

    aerosol_instance_list = []
    bin_names = []
    bin_radii = []
    bin_masses_Tg = []

    with xr.open_dataset(state_path) as state_xr:
        for name, r_m in radius_mapping.items():
            if name not in state_xr.data_vars:
                raise KeyError(
                    f"bin variable {name!r} declared in radius mapping is "
                    f"missing from {state_path}"
                )
            mmr_da = state_xr[name]

            # Interpolate to the model grid (lat, lev)
            mmr_grid = _interp_mmr_to_grid(mmr_da, model_lat, model_lev)

            # Convert mass mixing ratio -> volume mixing ratio
            m_p = (4.0 / 3.0) * np.pi * r_m ** 3 * rho_particle
            vmr = (m_air / m_p) * mmr_grid

            aerosol_instance_list.append(
                aerosol_instance(material_name, r_m, vmr)
            )

            # Diagnostic: total mass of this bin, computed on the ORIGINAL
            # (non-interpolated) grid so it matches the transport run.
            # column burden = (1/g) ∫ mmr dp  (with dp in Pa, g in m/s^2
            # → kg/m^2). Here 1e2 converts hPa → Pa.
            mmr_on_src = mmr_da.transpose('lat', 'lev') if mmr_da.dims == ('lev', 'lat') else mmr_da
            col_burden_kg_m2 = (1e2 / const.g) * mmr_on_src.integrate('lev').values
            lat_src = np.asarray(mmr_on_src.lat.values)
            global_mean_burden = _cosine_lat_avg(col_burden_kg_m2, lat_src)
            mass_Tg = global_mean_burden * 4 * np.pi * const.a**2 / 1e9

            bin_names.append(name)
            bin_radii.append(float(r_m))
            bin_masses_Tg.append(float(mass_Tg))

    total_mass_Tg = float(sum(bin_masses_Tg))
    if total_mass_Tg > 0.0:
        avg_diameter_m = float(
            sum(2.0 * r * m for r, m in zip(bin_radii, bin_masses_Tg))
            / total_mass_Tg
        )
    else:
        avg_diameter_m = 0.0

    return MultiBinAerosolLayer(
        aerosol_instance_list=aerosol_instance_list,
        total_mass_Tg=total_mass_Tg,
        avg_diameter_m=avg_diameter_m,
        bin_names=bin_names,
        bin_radii_m=bin_radii,
        bin_masses_Tg=bin_masses_Tg,
    )
