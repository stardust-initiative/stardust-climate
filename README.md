# climate_runs_ext

Lean climate model runner built on
[`climlab`](https://climlab.readthedocs.io/) and
[`climlab-stardust-extension`](https://github.com/stardust-initiative/climlab-stardust-extension).
Provides the model-factory, reference-model, and post-processing
primitives used by the Stardust paper's SARF pipeline.

> **Reproducing the paper.** The SARF pipeline that reproduces the
> headline result of Lederer et al. (2026) lives in
> [`solid-sai-2d-paper`](https://github.com/stardust-initiative/solid-sai-2d-paper),
> which uses this library. **Start there** if your goal is paper
> reproduction.
>
> **Custom experiments.** The example in `examples/` shows how to build a
> column climate model using this library. Sweep automation and
> exploratory tooling we use internally are out of scope for the public
> release.

## Architecture

```
climate_runs_ext/          Core library
  __init__.py                 load_project_config()
  model_factory/              Model builder & generator (RRTMG column models)
  reference_model/            Build / spin-up the reference climate (rev6)
  diagnostics/                Radiation helpers, cloud feedback, relative humidity
  utils/                      ERA5 data loading, state I/O, model control, transport

examples/
  climatic_response_annual_avg.py   Full experiment workflow (CLI entry point)

tests/                        Unit + integration tests (pytest, mocked — no network)
```

## Installation

```bash
conda create -n climate_runs_ext_env python=3.11 -y
conda activate climate_runs_ext_env
conda install -c conda-forge climlab compilers meson meson-python -y
pip install -e .
```

`pyproject.toml` pins every Stardust-modified dependency at a public
release tag — `climlab-rrtmg @ v0.5.0`, `climlab-sbm-convection @ v0.3.0`,
`climlab-stardust-extension @ v0.1.0` — so a single `pip install`
resolves the whole chain without credentials. The Fortran components
(rrtmg, sbm-convection) compile from source during install, hence the
`compilers` + `meson` conda deps.

## Configuration

Most users don't need any configuration. The example workflow loads
input data via the
[`stardust-2d-inputs`](https://github.com/stardust-initiative/stardust-2d-inputs)
engine, which fetches files from the public Zenodo deposit
([10.5281/zenodo.20271742](https://doi.org/10.5281/zenodo.20271742))
on demand and verifies each file by SHA-256. No credentials required.

The engine's backend (local cache / GCS / Zenodo) is selected by a
config file pointed to by `STARDUST_2D_INPUTS_CONFIG`; see the
[engine's README](https://github.com/stardust-initiative/stardust-2d-inputs#configuration)
for details. The default (Zenodo) is what most external users want.

A legacy-format `config.json.example` is also shipped for the older
GitHub-PAT-based loader path; new users should ignore it.

## Running simulations

### Reference model only

Build and spin-up the reference annual-average climate:

```bash
python examples/climatic_response_annual_avg.py \
    -do_ref_calc True \
    -do_just_ref True \
    -n_cycle 36 \
    -t_cycle_days 30 \
    -t_avg_days 365 \
    -base_folder ./output/ref_run
```

### Quick test run (short spin-up)

```bash
python examples/climatic_response_annual_avg.py \
    -do_ref_calc True \
    -do_just_ref True \
    -n_cycle 2 \
    -t_cycle_days 5 \
    -t_avg_days 10 \
    -base_folder ./output/test_run
```

### Full climatic response (aerosol layer + fxCO2)

```bash
python examples/climatic_response_annual_avg.py \
    -do_ref_calc True \
    -do_layer_inst_rf True \
    -do_layer_erf True \
    -do_layer_steady True \
    -do_fxco2_inst_rf True \
    -do_fxco2_erf True \
    -do_fxco2_steady True \
    -do_stabilization_problem True \
    -n_cycle 36 \
    -base_folder ./output/full_run
```

### Module invocation

The workflow can also be run as a Python module:

```bash
python -m examples.climatic_response_annual_avg --help
```

## Command-line reference

### Run flags

| Flag | Default | Description |
|------|---------|-------------|
| `-do_ref_calc` | `False` | Build and spin-up the reference model |
| `-do_just_ref` | `False` | Stop after reference calculation (skip perturbations) |
| `-do_layer_inst_rf` | `False` | Compute instantaneous RF for aerosol layer |
| `-do_layer_erf` | `False` | Compute ERF for aerosol layer (fixed Ts) |
| `-do_layer_steady` | `False` | Compute steady-state response for aerosol layer |
| `-do_fxco2_inst_rf` | `False` | Compute instantaneous RF for CO2 perturbation |
| `-do_fxco2_erf` | `False` | Compute ERF for CO2 perturbation (fixed Ts) |
| `-do_fxco2_steady` | `False` | Compute steady-state response for CO2 perturbation |
| `-do_stabilization_problem` | `False` | Compute optimal aerosol burden to offset CO2 |

### Aerosol layer parameters

| Flag | Default | Description |
|------|---------|-------------|
| `-rho` | `2196.0` | Particle density (kg/m^3) |
| `-r_m` | `250e-9` | Particle radius (m) |
| `-material_name` | `silica` | Aerosol material name (must match the engine's optical-table key) |
| `-M_tot_Tg` | `10.0` | Total layer mass (Tg) |
| `-p_min_layer` | `20.0` | Layer top pressure (hPa) |
| `-p_max_layer` | `80.0` | Layer bottom pressure (hPa) |

### Integration parameters

| Flag | Default | Description |
|------|---------|-------------|
| `-n_cycle` | `36` | Number of short spin-up cycles |
| `-t_cycle_days` | `30.0` | Duration of each short cycle (days) |
| `-t_avg_days` | `365.0` | Duration of final averaging period (days) |
| `-fxco2` | `2.0` | CO2 multiplier for perturbation |
| `-n_rrtmg_repeat` | `100` | RRTMG repeat count for RF diagnostics |
| `-season` | `Annual` | Season selection (Annual, DJF, MAM, JJA, SON, or month name) |

### Paths

| Flag | Default | Description |
|------|---------|-------------|
| `-config_path` | `None` | Path to config.json (defaults to repo root) |
| `-base_folder` | `./output/climatic_response` | Output directory for results |

## Data requirements

The workflow needs ERA5 zonal-mean monthly fields, transport drivers
(eddy diffusivity, tropopause climatology), and RRTMG-banded aerosol
optical-property tables. These are fetched on demand from the
[Zenodo deposit](https://doi.org/10.5281/zenodo.20271742) via the
`stardust-2d-inputs` engine, content-addressed and SHA-256-verified.

No GitHub credentials, no local data clone, no manual downloads
required.

## Testing

```bash
conda activate climate_runs_ext_env
python -m pytest tests/ -v
```

All tests use mocks and do not require network access or real RRTMG
computation.

## Dependencies

- [`climlab`](https://climlab.readthedocs.io/) — core column model framework.
- [`climlab-stardust-extension @ v0.1.0`](https://github.com/stardust-initiative/climlab-stardust-extension) — extended radiation/convection/dynamics classes.
- [`climlab-rrtmg @ v0.5.0`](https://github.com/stardust-initiative/climlab-rrtmg_stardust) — Stardust fork of the Fortran RRTMG wrappers.
- [`climlab-sbm-convection @ v0.3.0`](https://github.com/stardust-initiative/climlab-sbm-convection_stardust) — Stardust fork of the Fortran Simplified Betts–Miller scheme.
- [`stardust-2d-inputs @ v0.1.0`](https://github.com/stardust-initiative/stardust-2d-inputs) — input-data loader (used at runtime via `core.loader`); data via [Zenodo `10.5281/zenodo.20271742`](https://doi.org/10.5281/zenodo.20271742).
- numpy, xarray, scipy, scikit-image, pooch, requests.

All Stardust packages are URL-pinned in `pyproject.toml`; `pip install`
resolves the chain without credentials.
