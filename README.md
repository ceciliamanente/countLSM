# countLSM

This repository makes the results from *A Dynamic Latent Space Model for Healthcare Mobility Networks: the Italian National Health Service case* (Manente, C., Alfò, M., and D'Angelo, S.) reproducible.

## Repository structure

The repository contains two directories:

- `cpp/` — C++ source code for the MCMC samplers (interfaced with R via RcppArmadillo)
- `R/` — R scripts to fit each model variant and reproduce the analyses in the paper

## cpp

Each file implements a Metropolis-within-Gibbs sampler for a different model specification:

| File | Description |
|------|-------------|
| `mcmc_base.cpp` | Baseline model |
| `mcmc_base_geo.cpp` | Baseline + geographical distance |
| `mcmc_base_offset.cpp` | Baseline + exposure offset |
| `mcmc_base_geo_offset.cpp` | Baseline + geographical distance + exposure offset |
| `mcmc_molt.cpp` | Multiplicative sender/receiver effects |
| `mcmc_molt_geo.cpp` | Multiplicative + geographical distance |
| `mcmc_molt_offset.cpp` | Multiplicative + exposure offset |
| `mcmc_molt_geo_offset.cpp` | Multiplicative + geographical distance + exposure offset |

## R

Each script loads the data, initialises the parameters, calls the corresponding C++ sampler via Rcpp, and saves the posterior samples:

| File | Description |
|------|-------------|
| `run_base.R` | Baseline model |
| `run_base_geo.R` | Baseline + geographical distance |
| `run_base_offset.R` | Baseline + exposure offset |
| `run_base_geo_offset.R` | Baseline + geographical distance + exposure offset |
| `run_molt.R` | Multiplicative sender/receiver effects |
| `run_molt_geo.R` | Multiplicative + geographical distance |
| `run_molt_offset.R` | Multiplicative + exposure offset |
| `run_molt_geo_offset.R` | Multiplicative + geographical distance + exposure offset |

## Dependencies

The C++ code requires [RcppArmadillo](https://cran.r-project.org/package=RcppArmadillo). The R scripts require the following packages: `Rcpp`, `RcppArmadillo`, `MASS`, `ggplot2`, `dplyr`, `vegan`.

## Data availability

The data used in the paper are administrative hospital discharge records held by the Italian Ministry of Health and AGENAS. Access is subject to institutional data agreements and cannot be shared publicly.
