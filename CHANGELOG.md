# Changelog

All notable changes to this project are documented here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-05-24

Initial public release.

### Added

- Climate-model runner library `climate_runs_ext`:
  - `model_factory/` — RRTMG column-model builder + model generator.
  - `reference_model/` — reference annual-average climate (rev6) builder
    and spin-up.
  - `diagnostics/` — radiation helpers, cloud-feedback diagnostics,
    relative-humidity utilities.
  - `utils/` — ERA5 data loading, state I/O, model control (limiters),
    transport helpers.
- One curated example: `examples/climatic_response_annual_avg.py` —
  full experiment workflow (reference + aerosol layer + fxCO2 + ERF /
  steady-state response / stabilization problem), CLI entry point.
- Mocked test suite (108 tests, no network or real RRTMG required).
- URL-pinned dependencies on every Stardust-modified package:
  - [`climlab-rrtmg` v0.5.0](https://github.com/stardust-initiative/climlab-rrtmg_stardust)
  - [`climlab-sbm-convection` v0.3.0](https://github.com/stardust-initiative/climlab-sbm-convection_stardust)
  - [`climlab-stardust-extension` v0.1.0](https://github.com/stardust-initiative/climlab-stardust-extension)
  - [`stardust-2d-inputs` v0.1.0](https://github.com/stardust-initiative/stardust-2d-inputs)
    (input data via Zenodo
    [10.5281/zenodo.20271742](https://doi.org/10.5281/zenodo.20271742)).
- Public, non-credentialed install path: a single `pip install` resolves
  the whole chain.
