# countLSM
This repository makes the results from "A Dynamic Latent Space Model for Healthcare Mobility Networks: the Italian National Health Service case" reproducible.

There are two directories: `cpp` contains the C++ source code for the MCMC samplers, and `R` contains the R scripts used to run the models and reproduce the analyses.

The two subdirectories are described in more detail below.

## cpp
In this directory, the C++ source code for all MCMC samplers can be found. Each file corresponds to a different model variant:

- `mcmc_base.cpp` — baseline model
- `mcmc_base_geo.cpp` — baseline model with geographical distance 
- `mcmc_base_offset.cpp` — baseline model with exposure adjustment
- `mcmc_base_geo_offset.cpp` — baseline model with geographical distance and exposure adjustment term
- `mcmc_molt.cpp` — multiplicative sender/receiver effects model
- `mcmc_molt_geo.cpp` — multiplicative model with geographical distance 
- `mcmc_molt_offset.cpp` — multiplicative model with exposure adjustment term
- `mcmc_molt_geo_offset.cpp` — multiplicative model with geographical distance and exposure adjustment term

## R
In this directory, the R scripts used to fit each model to the Italian healthcare mobility data can be found:

- `run_base.R` — fits the baseline model
- `run_base_geo.R` — fits the baseline model with geographical distance
- `run_base_offset.R` — fits the baseline model with exposure adjustment term 
- `run_base_geo_offset.R` — fits the baseline model with geographical distance and exposure adjustment term
- `run_molt.R` — fits the multiplicative model
- `run_molt_geo.R` — fits the multiplicative model with geographical distance
- `run_molt_offset.R` — fits the multiplicative model with exposure adjustment term 
- `run_molt_geo_offset.R` — fits the multiplicative model with geographical distance and exposure adjustment term
