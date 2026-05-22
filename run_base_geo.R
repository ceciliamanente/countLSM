# ==============================================================================
# ==============================================================================
# ================ Baseline + Geographical Distance Model ======================
# ==============================================================================
# ==============================================================================

# ---------------------------------------------------------------
# PATHS 
# ---------------------------------------------------------------
cpp_path      <- "path/to/cpp/mcmc_base_geo.cpp"
data_path     <- "path/to/data/ANCA"
geo_matrix_path <- "path/to/data/dist_mat.rds"
output_dir_mcmc <- "path/to/output/mcmc"
output_dir_ppc  <- "path/to/output/ppc"
ppc_script    <- "path/to/ppc.R"

# ---------------------------------------------------------------
# 0). SETUP
# ---------------------------------------------------------------

library(MASS)
library(pscl)
library(ggplot2)
library(dplyr)
library(reshape2)
library(vegan)  # per procrustes
library(readxl)
library(stringi)
library(stringr)

Rcpp::sourceCpp(cpp_path)

# ---------------------------------------------------------------
# 1). DATA: load OD matrices 
# ---------------------------------------------------------------

years <- 2018:2024

Y_list <- setNames(vector("list", length(years)), years)

for (yr in years) {
  fname <- file.path(data_path, paste0("anca_OD_", yr, "_nomi.rds"))
  Y_list[[as.character(yr)]] <- readRDS(fname)
}

common_names <- Reduce(intersect, lapply(Y_list, rownames))
Y_list <- lapply(Y_list, function(m) {
  m <- m[common_names, common_names]
  storage.mode(m) <- "double"
  m
})

years <- names(Y_list)
W_list <- list()

for (yr in years) {
  Y <- Y_list[[yr]]
  nodes <- rownames(Y)
  diag(Y) <- 0
  
  isolated <- nodes[rowSums(Y) == 0 & colSums(Y) == 0]
  
  active <- rep(TRUE, length(nodes))
  active[nodes %in% isolated] <- FALSE
  
  W <- outer(active, active, function(a, b) as.numeric(a & b))
  dimnames(W) <- dimnames(Y)
  W_list[[yr]] <- W
}

n_nodes <- nrow(Y_list[[1]])
n_years <- length(Y_list)

# ---------------------------------------------------------------
# 1b). GEOGRAPHICAL DISTANCES MATRIX 
# ---------------------------------------------------------------

G <- readRDS(geo_matrix_path)
G <- G[rownames(Y_list[[1]]), rownames(Y_list[[1]])]
diag(G) <- 0

# ---------------------------------------------------------------
# 2). INITIALISATIONS: X, beta, alpha, r
# ---------------------------------------------------------------

make_dissimilarity <- function(Y) {
  eps <- 1e-3
  S <- (Y + t(Y)) / 2
  D <- -log(S + eps)
  D - min(D)
}

init_X_mds <- function(Y_list, p = 2) {
  TT <- length(Y_list)
  X_list <- vector("list", TT)
  
  for (tt in seq_len(TT)) {
    D <- make_dissimilarity(Y_list[[tt]])
    X <- cmdscale(D, k = p, add = TRUE)$points
    
    if (any(!is.finite(X))) {
      set.seed(11 + tt)
      X <- matrix(rnorm(nrow(D) * p, sd = 0.1), nrow(D), p)
    }
    
    X <- sweep(X, 2, colMeans(X))
    sc <- sqrt(mean(rowSums(X^2)))
    if (sc > 0) X <- X / sc
    
    if (tt > 1) X <- vegan::procrustes(X_list[[tt - 1]], X, scale = FALSE)$Yrot
    
    X_list[[tt]] <- X
  }
  X_list
}

X_list <- init_X_mds(Y_list)

init_beta <- function(Y_list) {
  sapply(Y_list, function(Y) {
    nz <- mean(Y > 0)
    nz <- pmax(pmin(nz, 1 - 1e-6), 1e-6)
    qlogis(nz)
  })
}

beta_vec <- init_beta(Y_list)

init_glm_values <- function(Y_list, X_list) {
  alpha <- beta <- r <- numeric(length(Y_list))
  
  for (i in seq_along(Y_list)) {
    Y <- Y_list[[i]]
    diag(Y) <- 0
    distZ <- as.matrix(dist(X_list[[i]]))
    
    fit <- glm.nb(c(Y) ~ c(distZ))
    alpha[i] <- coef(fit)[1]
    beta[i] <- coef(fit)[2]
    r[i] <- fit$theta
  }
  
  list(alpha = alpha, beta = beta, r = r)
}

init_params <- init_glm_values(Y_list, X_list)
alpha_vec <- init_params$alpha
r_vec <- init_params$r

mu_alpha <- mean(sapply(Y_list, function(Y) log(mean(Y[Y > 0]) + 1e-3))) + 3
mu_beta <- mean(sapply(Y_list, function(Y) {
  p <- mean(Y > 0)
  p <- pmax(pmin(p, 1 - 1e-6), 1e-6)
  qlogis(p)
}))

rho_vec <- rep(0, n_years)
mu_rho <- 0

# ---------------------------------------------------------------
# 3). HYPERPARAMETERS 
# ---------------------------------------------------------------

n_iter <- 50000
burn_in <- 20000

sigma2_alpha <- 5.0
sigma2_beta <- 1.0
sigma2_rho <- 1.0    
sigma2_rwX <- 1.0
sigma_a <- 0.5

prop_sd_X <- 0.15
prop_sd_alpha <- 0.3
prop_sd_beta <- 0.3
prop_sd_rho <- 0.05  
prop_sd_r <- 0.05

# ---------------------------------------------------------------
# 4). MCMC
# ---------------------------------------------------------------

set.seed(123)

res_base_geo <- mcmc_base_geo(
  Y_list_R = Y_list,
  X_list_R = X_list,
  W_list_R = W_list,
  
  G = G,                    
  
  alpha_vec = alpha_vec,
  beta_vec = beta_vec,
  rho_vec = rho_vec,        
  r_vec = r_vec,
  
  sigma2_alpha = sigma2_alpha,
  sigma2_beta = sigma2_beta,
  sigma2_rho = sigma2_rho,  
  
  n_iter = n_iter,
  burn_in = burn_in,
  
  prop_sd_X = prop_sd_X,
  prop_sd_alpha = prop_sd_alpha,
  prop_sd_beta = prop_sd_beta,
  prop_sd_rho = prop_sd_rho,  
  prop_sd_r = prop_sd_r,
  
  sigma2 = sigma2_rwX,
  sigma_a = sigma_a,
  
  mu_alpha = mu_alpha,
  mu_beta = mu_beta,
  mu_rho = mu_rho,          
  
  include_diagonal = FALSE,
  joint_update_r = TRUE,
  verbose = TRUE
)

saveRDS(res_base_geo, file = file.path(output_dir_mcmc, "res_base_geo.rds"))

# ---------------------------------------------------------------
# 5). DIAGNOSTIC
# ---------------------------------------------------------------

res_base_geo$DIC

alpha_samples <- res_base_geo$alpha_samples
beta_samples <- res_base_geo$beta_samples
rho_samples <- res_base_geo$rho_samples
r_samples <- res_base_geo$r_samples

plot_traceplots <- function(samples, param_name, year_names, color = "darkred") {
  n_years <- ncol(samples)
  par(mfrow = c(ceiling(n_years/3), 3), mar = c(4, 4, 2, 1))
  
  for (t in 1:n_years) {
    plot(samples[, t], type = "l", col = color, lwd = 0.5,
         main = paste(param_name, "year", year_names[t]),
         ylab = bquote(.(param_name)[.(t)]),
         xlab = "Iteration")
    abline(h = mean(samples[, t]), col = "blue", lty = 2, lwd = 2)
  }
}

# Alpha
plot_traceplots(alpha_samples, "alpha", names(Y_list), "darkred")

# Beta
plot_traceplots(beta_samples, "beta", names(Y_list), "purple")

# R
plot_traceplots(r_samples, "r", names(Y_list), "steelblue")

# Rho
plot_traceplots(rho_samples, "rho", names(Y_list), "darkgreen")

# ---------------------------------------------------------------
# 6). PPC
# ---------------------------------------------------------------

source(ppc_script)

ppc_results_base_geo <- ppc_all_years_BASE_GEO(
  Y_list = Y_list,
  res = res_base_geo,
  W_list = W_list,
  G = G,
  n_sim = 500,
  verbose = TRUE
)

# Save
saveRDS(ppc_results_base_geo, file = file.path(output_dir_ppc, "ppc_results_base_geo.rds"))

# ---------------------------------------------------------------
# 7). PLOT PPC
# ---------------------------------------------------------------

# Loop over all years
for (year in names(Y_list)) {
  
  cat(sprintf("  Year %s... ", year))
  
  # 1. Statistiche aggregate (con p-values)
  plot_ppc_stats(
    ppc_results = ppc_results_base_geo,
    year_name = year,
    model = "base"
  )
  
  # 2. Distribution
  plot_ppc_distribution(
    ppc_results = ppc_results_base_geo,
    Y_list = Y_list,
    W_list = W_list,
    year_name = year
  )
  
  # 3. Quantiles
  plot_ppc_quantiles(
    ppc_results = ppc_results_base_geo,
    Y_list = Y_list,
    W_list = W_list,
    year_name = year
  )
  
  # 4. MSE heatmap
  ppc_mse_heatmap(
    ppc_results = ppc_results_base_geo,
    year_name = year,
    Y_list = Y_list,
    highlight_top = 20
  )
}

