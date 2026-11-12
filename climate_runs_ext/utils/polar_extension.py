"""Nearest finite latitude continuation only in exterior polar missing tails."""
import numpy as np

FIELDS = ('D_yy', 'D_pp_tilt', 'D_pp_theta', 'D_yp_tilt')

def extend_polar_tables(source, mapping='mean_first', cap_latitude=85.0):
    """Preserve finite entries/interior holes; never fill across time or pressure.

    Raw source remains immutable. Existing downstream clipping and tensor
    safeguards are deliberately retained. Rows without support remain missing.
    """
    result = source.copy(deep=False)
    lat = source.latitude.values
    if not np.all(np.diff(lat) > 0):
        raise ValueError('Polar continuation requires increasing latitude')
    counts = {}
    for component in FIELDS:
        name = 'iso_' + mapping + '_on_model_' + component
        original = source[name]
        table = original.transpose('year', 'month', 'model_pressure', 'latitude')
        values = table.values.copy()
        rows = values.reshape(-1, len(lat))
        count = 0
        for row in rows:
            indices = np.flatnonzero(np.isfinite(row))
            if not len(indices):
                continue
            lo, hi = indices[0], indices[-1]
            south = (np.arange(len(lat)) < lo) & (lat <= -cap_latitude)
            north = (np.arange(len(lat)) > hi) & (lat >= cap_latitude)
            row[south] = row[lo]
            row[north] = row[hi]
            count += int(south.sum() + north.sum())
        result[name] = table.copy(data=values).transpose(*original.dims)
        counts[component] = count
    result.attrs = dict(source.attrs, polar_continuation='nearest finite latitude per component/year/month/pressure; exterior missing tails only, abs(latitude)>=85; existing tensor sanitation retained')
    return result, counts
