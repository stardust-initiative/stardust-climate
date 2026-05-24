# ---------------------------------------------------------------------------
# climate_runs_ext.model_factory
#
# Core model assembly and high-level model generator.
#
# Modules
# -------
# model_builder    – get_rce_sbm_model_annual_avg (process-level assembly)
# model_generator  – model_generator (high-level factory with defaults)
# ---------------------------------------------------------------------------

from climate_runs_ext.model_factory.model_builder import get_rce_sbm_model_annual_avg
from climate_runs_ext.model_factory.model_generator import model_generator
