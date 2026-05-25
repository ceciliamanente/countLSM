# ==============================================================================
# ==============================================================================
# =========================== Multiplicative Model =============================
# ==============================================================================
# ==============================================================================

# ---------------------------------------------------------------
# PATHS 
# ---------------------------------------------------------------
cpp_path      <- "path/to/cpp/mcmc_molt.cpp"
data_path     <- "path/to/data/ANCA"
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
library(vegan) 

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

mu_alpha <- mean(sapply(Y_list, function(Y) log(mean(Y[Y > 0]) + 1e-3))) + 5
mu_beta <- mean(sapply(Y_list, function(Y) {
  p <- mean(Y > 0)
  p <- pmax(pmin(p, 1 - 1e-6), 1e-6)
  qlogis(p)
}))

init_gamma_theta_from_Y <- function(Y_list, include_diagonal = FALSE) {
  Tn <- length(Y_list)
  n  <- nrow(Y_list[[1]])
  
  out_deg <- rep(0, n)
  in_deg  <- rep(0, n)
  
  for (t in seq_len(Tn)) {
    Y <- Y_list[[t]]
    if (!include_diagonal) diag(Y) <- 0
    out_deg <- out_deg + rowSums(Y)
    in_deg  <- in_deg  + colSums(Y)
  }
  
  out_deg <- out_deg / Tn
  in_deg  <- in_deg  / Tn
  
  eps <- 0.5
  g_raw <- log(out_deg + eps)
  t_raw <- log(in_deg  + eps)
  
  g_sd <- sd(g_raw); if (!is.finite(g_sd) || g_sd == 0) g_z <- rep(0, n) else g_z <- (g_raw - mean(g_raw)) / g_sd
  t_sd <- sd(t_raw); if (!is.finite(t_sd) || t_sd == 0) t_z <- rep(0, n) else t_z <- (t_raw - mean(t_raw)) / t_sd
  
  squash <- function(z) z / (1 + abs(z))
  
  gamma_init <- squash(g_z) - mean(squash(g_z))
  theta_init <- squash(t_z) - mean(squash(t_z))
  
  list(gamma_init = gamma_init, theta_init = theta_init)
}

init_gt <- init_gamma_theta_from_Y(Y_list, include_diagonal = FALSE)

gamma_vec <- init_gt$gamma_init
theta_vec <- init_gt$theta_init

# ---------------------------------------------------------------
# 3). HYPERPARAMETERS 
# ---------------------------------------------------------------

n_iter <- 50000
burn_in <- 20000

# Variance prior
sigma2_alpha <- 5
sigma2_beta  <- 1.0
sigma2_rwX   <- 1      

sigma_a <- 0.5

sigma2_gamma <- 1
sigma2_theta <- 1

# Proposal standard deviations
prop_sd_X     <- 0.15
prop_sd_alpha <- 0.3
prop_sd_beta  <- 0.3
prop_sd_r   <- 0.05

prop_sd_gamma <- 0.1 
prop_sd_theta <- 0.3 

Y_mean <- Reduce("+", Y_list) / length(Y_list)

out_degree <- rowSums(Y_mean)
in_degree  <- colSums(Y_mean)

sender_ref   <- which.min(abs(out_degree - median(out_degree))) - 1 
receiver_ref <- which.min(abs(in_degree  - median(in_degree ))) - 1


# ---------------------------------------------------------------
# 4). MCMC
# ---------------------------------------------------------------

res_molt <- mcmc_mult(
  Y_list_R = Y_list,
  X_list_R = X_list,
  W_list_R = W_list,    
  
  alpha_vec = alpha_vec,        
  beta_vec  = beta_vec,  
  r_vec   = r_vec,
  gamma_vec = gamma_vec,
  theta_vec = theta_vec,
  
  sigma2_alpha = sigma2_alpha,
  sigma2_beta = sigma2_beta,
  sigma2_gamma = sigma2_gamma, 
  sigma2_theta = sigma2_theta,
  
  n_iter = n_iter,
  burn_in = burn_in,
  
  prop_sd_X = prop_sd_X,
  prop_sd_alpha = prop_sd_alpha,
  prop_sd_beta = prop_sd_beta,
  prop_sd_r = prop_sd_r,
  prop_sd_gamma = prop_sd_gamma,
  prop_sd_theta = prop_sd_theta,
  
  sigma2 = sigma2_rwX,
  
  sigma_a = sigma_a,
  
  mu_alpha = mu_alpha,          
  mu_beta = mu_beta, 
  
  sender_ref = sender_ref,
  receiver_ref = receiver_ref,
  
  include_diagonal = FALSE,
  joint_update_r = TRUE,
  verbose = TRUE)

# Save
saveRDS(res_base, file = file.path(output_dir_mcmc, "res_base.rds"))

# ---------------------------------------------------------------
# 5). DIAGNOSTIC
# ---------------------------------------------------------------

res_molt$DIC

alpha_samples <- res_molt$alpha_samples
beta_samples <- res_molt$beta_samples
r_samples <- res_molt$r_samples
X_last <- res_molt$X_last
X_samples <- res_molt$X_samples
gamma_samples <- res_molt$gamma_samples
theta_samples <- res_molt$theta_samples

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


par(mfrow = c(3,3))
for (i in 1:6) {
  plot(gamma_samples[, i], type = "l", col = "orange", lwd = 0.5,
       main = paste("Gamma node", i),
       ylab = expression(gamma[i]))
  abline(h = mean(gamma_samples[, i]), col = "blue", lty = 2, lwd = 2)
}
dev.off()

par(mfrow = c(3,3))
for (i in 1:6) {
  plot(theta_samples[, i], type = "l", col = "darkgreen", lwd = 0.5,
       main = paste("Theta node", i),
       ylab = expression(theta[i]))
  abline(h = mean(theta_samples[, i]), col = "blue", lty = 2, lwd = 2)
}
dev.off()

# ---------------------------------------------------------------
# 6). PPC
# ---------------------------------------------------------------

source(ppc_script)

ppc_results_molt <- ppc_all_years_MOLT(
  Y_list = Y_list,
  res = res_molt,
  W_list = W_list,
  n_sim = 500
)

# Save
saveRDS(ppc_results_molt, file = file.path(output_dir_ppc, "ppc_results_molt.rds"))

# ---------------------------------------------------------------
# 7). PLOT PPC
# ---------------------------------------------------------------

# Loop over all years
for (year in names(Y_list)) {
  
  cat(sprintf("  Year %s... ", year))
  
  # 1. Stats
  plot_ppc_stats(
    ppc_results = ppc_results_molt,
    year_name = year,
    model = "base"
  )
  
  # 2. Distributions
  plot_ppc_distribution(
    ppc_results = ppc_results_molt,
    Y_list = Y_list,
    W_list = W_list,
    year_name = year
  )
  
  # 3. Quantiles
  plot_ppc_quantiles(
    ppc_results = ppc_results_molt,
    Y_list = Y_list,
    W_list = W_list,
    year_name = year
  )
  
  # 4. MSE heatmap
  ppc_mse_heatmap(
    ppc_results = ppc_results_molt,
    year_name = year,
    Y_list = Y_list,
    highlight_top = 20
  )
}


