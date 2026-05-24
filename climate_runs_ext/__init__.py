"""climate_runs_ext -- lean climate model runner.

Built on ``climlab`` + ``climlab-stardust-extension``. Provides the
model-factory, reference-model, and post-processing primitives used by
the Stardust paper's SARF pipeline.

Public API
----------
load_project_config(config_path)
    Parse the project JSON config and return a dict of resolved URLs,
    commit hashes, and aerosol table metadata.
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import json
import os

from climlab_stardust_extension.utils.file_handling import _get_latest_commit_hash


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_project_config(config_path=None):
    """Load and resolve the project configuration file.

    The JSON config contains GitHub credentials, repository names, aerosol
    table paths, and (optionally) pinned commit hashes.  This function
    resolves any missing commit hashes via the GitHub API and builds the
    fully-qualified raw-content URLs.

    Parameters
    ----------
    config_path : str or None
        Path to the JSON config file.  Defaults to ``config.json`` in the
        repository root (i.e. one level up from ``climate_runs_ext/``).

    Returns
    -------
    dict
        Keys:

        * ``proj_name``  -- project cache name
        * ``aerosols_token`` / ``climate_database_token``
        * ``optical_table_commit`` / ``climate_database_commit``
        * ``aerosols_tables_dict``  -- material → path mapping
        * ``aerosols_opt_tables_http``  -- base URL for optical tables
        * ``climate_database_files_http``  -- base URL for ERA5 data
        * ``aerosols_input_dict``  -- convenience sub-dict for
          ``AerosolsOptDepTables``
    """
    # --- locate the config file -------------------------------------------
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'config.json',
        )

    with open(config_path, 'r') as f:
        raw = json.load(f)

    # --- extract raw fields -----------------------------------------------
    gh = raw['github']
    proj_name = raw['project_name']
    token = gh['token']
    org = gh['organization_name']

    # --- resolve commit hashes (use pinned value or query GitHub) ----------
    optical_table_commit = gh.get('optical_tables_commit')
    if optical_table_commit is None:
        optical_table_commit = _get_latest_commit_hash(
            org, gh['materials_repository_name'], token, proj_name,
        )
    climate_database_commit = gh.get('climate_database_commit')
    if climate_database_commit is None:
        climate_database_commit = _get_latest_commit_hash(
            org, gh['climate_database_name'], token, proj_name,
        )

    # --- build fully-qualified raw-content URLs ----------------------------
    aerosols_opt_tables_http = (
        f"https://raw.githubusercontent.com/{org}/"
        f"{gh['materials_repository_name']}/{optical_table_commit}"
    )
    climate_database_files_http = (
        f"https://raw.githubusercontent.com/{org}/"
        f"{gh['climate_database_name']}/{climate_database_commit}"
    )

    # --- convenience sub-dict for AerosolsOptDepTables --------------------
    aerosols_tables_dict = raw['aerosols_table_dict']
    aerosols_input_dict = {
        'aerosols_opt_tables_http': aerosols_opt_tables_http,
        'aerosols_tables_dict': aerosols_tables_dict,
        'aerosols_token': token,
        'proj_name': proj_name,
    }

    # --- assemble the full config dict ------------------------------------
    return {
        'proj_name': proj_name,
        'aerosols_token': token,
        'climate_database_token': token,
        'optical_table_commit': optical_table_commit,
        'climate_database_commit': climate_database_commit,
        'aerosols_tables_dict': aerosols_tables_dict,
        'aerosols_opt_tables_http': aerosols_opt_tables_http,
        'climate_database_files_http': climate_database_files_http,
        'aerosols_input_dict': aerosols_input_dict
    }
