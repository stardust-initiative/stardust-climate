"""Cloud feedback diagnostic process.

Ported from ``climate_runs/utils/utils_methods.py``: ``CloudFeedback``.

Parameterises dynamic cloud response to temperature and relative-humidity
anomalies using polynomial sensitivity coefficients loaded from a NetCDF
dataset.  The cloud properties (``cldfrac``, ``clwp``, ``ciwp``) are
updated at every diagnostic step and made available via the ``cloud_dict``
property.

Usage
-----
::

    from climate_runs_ext.diagnostics.cloud_feedback import CloudFeedback

    cf = CloudFeedback(
        mycloud0=initial_cloud_dict,
        sensitivity_dataset_filename='cloud_sensitivity_20lat_37lev',
        state=model.state,
        timestep=model.timestep,
    )
    model.add_subprocess('CloudFeedback', cf)
    # After stepping: cf.cloud_dict contains updated cloud fields.
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import numpy as np
from copy import deepcopy
from scipy import interpolate

from climlab import constants as const
from climlab.process.diagnostic import DiagnosticProcess

from climate_runs_ext.diagnostics.relative_humidity import RelativeHumidity
from climate_runs_ext.utils.data_loading import load_xr_from_repo


# ---------------------------------------------------------------------------
# CloudFeedback
# ---------------------------------------------------------------------------

class CloudFeedback(DiagnosticProcess):
    """Diagnostic process implementing polynomial cloud feedback.

    Cloud properties are expressed as perturbations from a reference
    state (``mycloud0``), driven by anomalies in temperature (dT) and
    relative humidity (drh) relative to a reference profile.

    The sensitivity coefficients are loaded from a NetCDF file stored in
    the climate database GitHub repository.  Each cloud variable
    (cldfrac, clwp, ciwp) has its own set of multivariate polynomial
    coefficients keyed by ``(n_T, n_rh, ...)``.

    Parameters
    ----------
    mycloud0 : dict
        Reference cloud dict with keys ``'cldfrac'``, ``'clwp'``,
        ``'ciwp'``, ``'r_liq'``, ``'r_ice'``.
    sensitivity_dataset_filename : str
        File key (without extension) for the sensitivity NetCDF in the
        climate database repository.
    config : dict
        Project configuration (from ``load_project_config()``).
        Required for loading the sensitivity dataset.
    do_cc_transformation : bool, optional
        Apply log-transform to cloud fraction (default False).
    do_product_fit : bool, optional
        Interpret water-path fit as Δ(cc·wp) (default False).
    do_rel_coord : bool, optional
        Use relative coordinates for dT and drh (default True).
    cld_fb_Tmin : float, optional
        Temperature floor — cloud anomalies are zeroed where
        T < Tmin (default 200 K).
    kind : str, optional
        Interpolation method for the sensitivity grid (default 'linear').
    n_avg : int, optional
        Running-average window for dT and drh (default 1 = no averaging).
    do_era5, do_simplified, small : optional
        Forwarded to the internal ``RelativeHumidity`` process.
    **kwargs
        Passed to ``DiagnosticProcess.__init__``.  Must include ``state``
        with ``'Tatm'``, ``'q'``, ``'Ts'``.

    Properties
    ----------
    cloud_dict : dict
        Updated cloud parameter dict (recomputed on access).
    """

    def __init__(self, mycloud0, sensitivity_dataset_filename, **kwargs):
        # --- extract cloud-feedback-specific kwargs -----------------------
        config = kwargs.pop('config')
        do_cc_transformation = kwargs.pop('do_cc_transformation', False)
        do_product_fit = kwargs.pop('do_product_fit', False)
        do_rel_coord = kwargs.pop('do_rel_coord', True)
        Tmin = kwargs.pop('cld_fb_Tmin', 200.0)
        method = kwargs.pop('kind', 'linear')
        n_avg = kwargs.pop('n_avg', 1)

        # --- extract qsat kwargs before super().__init__ ------------------
        qsat_kw_names = ('do_simplified', 'small', 'do_era5')
        qsat_params = {
            k: kwargs.pop(k) for k in list(kwargs)
            if k in qsat_kw_names
        }

        super(CloudFeedback, self).__init__(**kwargs)

        # --- store config -------------------------------------------------
        self.do_cc_transformation = do_cc_transformation
        self.do_product_fit = do_product_fit
        self.do_rel_coord = do_rel_coord
        self.Tmin = Tmin
        self.qsat_params = qsat_params

        # --- physical limits on cloud parameters --------------------------
        self.limit_dict = {
            'cldfrac': (1e-9, 1.0),
            'clwp':    (0.0,  1.0e4),
            'ciwp':    (0.0,  1.0e4),
        }

        # --- reference cloud state (deep-copied and clipped) --------------
        self.mycloud0 = deepcopy(mycloud0)
        for param_name, lim in self.limit_dict.items():
            self.mycloud0[param_name] = np.clip(
                self.mycloud0[param_name], lim[0], lim[1],
            )

        # --- reference atmospheric state ----------------------------------
        self.T0 = deepcopy(self.Tatm)
        self.q0 = deepcopy(self.q)
        self.rh_proc = RelativeHumidity(
            state=self.state,
            timestep=getattr(self, 'timestep_in_seconds', self.timestep),
            **self.qsat_params,
        )
        self.rh0 = deepcopy(self.rh_proc.rh)

        # --- load sensitivity dataset and build interpolators -------------
        self._build_sensitivity_interpolators(
            sensitivity_dataset_filename, config, method,
        )

        # --- external parameter anomalies (for extended fits) -------------
        self.external_dparam_dict = {
            name: 0.0 * self.T0
            for name in self.param_external_names
        }

        # --- initialise running average -----------------------------------
        self.first_step = True
        self._cloud_dict = deepcopy(self.mycloud0)
        self.n_avg = n_avg   # triggers initialize_avg via setter

    # -----------------------------------------------------------------------
    # Sensitivity dataset loading
    # -----------------------------------------------------------------------

    def _build_sensitivity_interpolators(self, filename, config, method):
        """Load the sensitivity NetCDF and pre-compute interpolated grids.

        For each cloud parameter (cldfrac, clwp, ciwp) and each polynomial
        key (e.g. 'T', 'rh', 'T,T', 'T,rh', ...), we interpolate the
        sensitivity coefficient from the dataset grid onto the model grid.
        """
        # --- load the sensitivity dataset ---------------------------------
        fields = load_xr_from_repo(filename, config)

        # --- build (lat, lev) coordinate arrays for the model grid --------
        lat = self.state.Tatm.domain.lat.points
        lev = self.state.Tatm.domain.lev.points
        lat_mat, lev_mat = np.meshgrid(lat, lev, indexing='ij')
        coords = np.column_stack([lat_mat.ravel(), lev_mat.ravel()])

        # --- pressure-to-water-path conversion ----------------------------
        # delta_p / g  converts mixing ratios (kg/kg per layer) to paths
        dp_g = (
            1e5 * self.state.Tatm.domain.lev.delta / const.g
        )  # shape (nlev,)

        map_dict = {
            'cldfrac': ('cloud fraction',      1.0),
            'clwp':    ('liquid water content', dp_g[np.newaxis, :]),
            'ciwp':    ('ice water content',    dp_g[np.newaxis, :]),
        }

        # --- optional variable-range limits (dT, drh clipping) ------------
        self.var_limit_dict = {}
        for var_name in ['T', 'rh']:
            key_min = f'Delta {var_name}: min'
            key_max = f'Delta {var_name}: max'
            if key_min in fields:
                var_min_ds = fields[key_min]
                var_max_ds = fields[key_max]
                interp_min = interpolate.RegularGridInterpolator(
                    (var_min_ds.latitude, var_min_ds.level),
                    var_min_ds.values.T,
                    bounds_error=False, fill_value=None, method=method,
                )
                interp_max = interpolate.RegularGridInterpolator(
                    (var_max_ds.latitude, var_max_ds.level),
                    var_max_ds.values.T,
                    bounds_error=False, fill_value=None, method=method,
                )
                var_min_grid = interp_min(coords).reshape(lat_mat.shape)
                var_max_grid = interp_max(coords).reshape(lat_mat.shape)
                self.var_limit_dict[var_name] = (var_min_grid, var_max_grid)

        # --- polynomial sensitivity coefficients --------------------------
        self.f_dict = {}
        for count, (param_name, param_tup) in enumerate(map_dict.items()):
            db_param_name = f'{param_tup[0]}-sensitivity'
            param_factor = param_tup[1]
            param = fields[db_param_name]

            # Discover external parameter names (beyond T, rh)
            if count == 0:
                param_external_list = list(param.key.values)
                param_external_list.remove('T')
                param_external_list.remove('rh')
                self.param_external_names = tuple(param_external_list)

            f_param_dict = {}
            for key in ['T', 'rh'] + list(self.param_external_names):
                param_sel = param.sel(key=key, drop=True)
                interp = interpolate.RegularGridInterpolator(
                    (param_sel.latitude, param_sel.level),
                    param_sel.values.T,
                    bounds_error=False, fill_value=None, method=method,
                )
                f_key = param_factor * interp(coords).reshape(lat_mat.shape)

                # Parse polynomial exponents from the key string
                s = key.split(',')
                n_t  = s.count('T')
                n_rh = s.count('rh')
                exp_list = [n_t, n_rh]
                for ext_name in self.param_external_names:
                    exp_list.append(s.count(ext_name))
                f_param_dict[tuple(exp_list)] = f_key

            self.f_dict[param_name] = f_param_dict

    # -----------------------------------------------------------------------
    # Reference-state update protocol
    # -----------------------------------------------------------------------

    def update_internal_fields(self, **kwargs):
        """Update the reference state for cloud anomaly calculations.

        Called by ``iteratively_update_internal`` when a new reference
        model is loaded.

        Parameters
        ----------
        ref_state : AttrDict, optional
            New reference state.  If not provided, uses ``self.state``.
        """
        state = kwargs.get('ref_state', self.state)
        self.T0[:] = state['Tatm'][:]
        self.q0[:] = state['q'][:]
        self.rh0 = self.rh_proc.rh_func(state, **self.qsat_params)
        self.first_step = True
        self._compute_cloud_param()

    # -----------------------------------------------------------------------
    # Core cloud computation
    # -----------------------------------------------------------------------

    def _compute_cloud_param(self):
        """Recompute cloud properties from current dT and drh anomalies."""
        small_val = 1e-10

        # --- compute anomalies --------------------------------------------
        if self.do_rel_coord:
            dT  = (self.state.Tatm - self.T0) / self.T0
            drh = (self.rh_proc.rh - self.rh0) / (self.rh0 + small_val)
        else:
            dT  = self.state.Tatm - self.T0
            drh = self.rh_proc.rh - self.rh0

        # --- running average ----------------------------------------------
        n = min(self.n_avg, self._steps_current + 1)
        self.dT  = ((n - 1) * self._prev_dT  + dT)  / n
        self.drh = ((n - 1) * self._prev_drh + drh) / n

        # Advance step counter if the model has actually stepped
        if self._steps0 + self._steps_current < self.time['steps']:
            self._steps_current += 1
            self._prev_dT[:]  = self.dT[:]
            self._prev_drh[:] = self.drh[:]

        # --- optional clipping of anomalies -------------------------------
        if 'T' in self.var_limit_dict:
            self.dT = np.clip(
                self.dT, *self.var_limit_dict['T'],
            )
        if 'rh' in self.var_limit_dict:
            self.drh = np.clip(
                self.drh, *self.var_limit_dict['rh'],
            )

        # --- evaluate polynomial expansion for each cloud parameter -------
        params_list = ['cldfrac', 'clwp', 'ciwp']
        for param_name in params_list:
            dparam_sum = None
            for exp_tup, f in self.f_dict[param_name].items():
                n_t, n_rh = exp_tup[:2]
                dparam_temp = f * self.dT**n_t * self.drh**n_rh

                # Multiply by external parameter anomalies if present
                for i_ext, ext_name in enumerate(self.param_external_names):
                    nexp = exp_tup[2 + i_ext]
                    if nexp > 0:
                        dparam_temp *= (
                            self.external_dparam_dict[ext_name] ** nexp
                        )

                # Zero out above the temperature floor
                dparam = np.where(self.Tatm >= self.Tmin, dparam_temp, 0.0)

                if dparam_sum is None:
                    dparam_sum = 0.0 * dparam
                dparam_sum = dparam_sum + dparam

            # --- product-fit correction for water path --------------------
            if self.do_product_fit and param_name in ('clwp', 'ciwp'):
                cc0 = self.mycloud0['cldfrac']
                dcc = self._cloud_dict['cldfrac'] - cc0
                dparam_sum = (
                    (dparam_sum - self.mycloud0[param_name] * dcc)
                    / (cc0 + dcc)
                )

            # --- apply perturbation to reference --------------------------
            param0 = self.mycloud0[param_name].copy()
            if self.do_cc_transformation and param_name == 'cldfrac':
                param0 = np.log(1.0 - param0)

            param = param0 + dparam_sum

            if self.do_cc_transformation and param_name == 'cldfrac':
                param = 1.0 - np.exp(param)

            # Clip and sanitise
            param = np.clip(param, *self.limit_dict[param_name])
            param = np.nan_to_num(param, nan=0.0)
            self._cloud_dict[param_name][:] = param

        # --- r_liq and r_ice are always unchanged from reference ----------
        self._cloud_dict['r_liq'][:] = self.mycloud0['r_liq'][:]
        self._cloud_dict['r_ice'][:] = self.mycloud0['r_ice'][:]

    # -----------------------------------------------------------------------
    # climlab interface
    # -----------------------------------------------------------------------

    def _compute(self):
        """Called by climlab on each diagnostic step."""
        self._compute_cloud_param()
        return {}

    @property
    def cloud_dict(self):
        """Updated cloud parameter dict (recomputed on access)."""
        self._compute_cloud_param()
        return self._cloud_dict

    # -----------------------------------------------------------------------
    # Running-average management
    # -----------------------------------------------------------------------

    def initialize_avg(self):
        """Reset the running-average state."""
        self._steps0 = self.time['steps']
        self._prev_dT  = 0.0 * self.Tatm
        self._prev_drh = 0.0 * self.Tatm
        self._steps_current = 0

    @property
    def n_avg(self):
        """Running-average window length."""
        return self._navg

    @n_avg.setter
    def n_avg(self, value):
        self._navg = value
        self.initialize_avg()
