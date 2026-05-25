/* ==============================================================
 Hurdle NB DLS Model - Baseline
 ============================================================== */

// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
using namespace Rcpp;
using namespace arma;

/* ========
Functions
=========== */

// Returns log(Gamma(x))
inline double lgamma_safe(double x) {
  return std::lgamma(x);
}

// Calculate all Euclidean distances between the rows of X. Returns a matrix D.
arma::mat pairwise_dist(const arma::mat& X) {
  int n = X.n_rows; 
  arma::mat D(n, n, fill::zeros); 
  
  for (int i = 0; i < n; ++i) {
    for (int j = i + 1; j < n; ++j) {
      double dx = X(i,0) - X(j,0); 
      double dy = X(i,1) - X(j,1); 
      double dij = std::sqrt(dx*dx + dy*dy); 
      D(i,j) = dij; 
      D(j,i) = dij; 
    }
  }
  return D; 
}

// Center the latent configuration by subtracting the mean row by row.
void enforce_center(arma::mat& X) {
  arma::rowvec meanRow = mean(X, 0); 
  X.each_row() -= meanRow; 
}

// Align X with Y using optimal rotation (SVD).
arma::mat procrustes(const arma::mat& X, const arma::mat& Y) {
  arma::mat Xc = X.each_row() - mean(X, 0); 
  arma::mat Yc = Y.each_row() - mean(Y, 0); 
  
  arma::mat M = Xc.t() * Yc; 
  arma::mat U, V; 
  arma::vec s; 
  
  arma::svd(U, s, V, M); 
  
  arma::mat R = U * V.t(); 
  arma::mat X_aligned = Xc * R; 
  
  return X_aligned; 
}

// Simulates a truncated Normal. 
double rtruncnorm_left0(double mu, double sd) {
  double a  = (0.0 - mu) / sd; 
  double Fa = R::pnorm(a, 0.0, 1.0, true, false); 
  double u  = R::runif(Fa, 1.0); 
  double z  = R::qnorm(u, 0.0, 1.0, true, false);
  return mu + sd * z; 
}

// Probability for the hurdle threshold.
inline void logistic_hurdle(double beta_t,
                            double &pi_ij,
                            double &log_pi,
                            double &log1m_pi) {
  
  pi_ij = 1.0 / (1.0 + std::exp(-beta_t)); 
  if (pi_ij < 1e-12)        pi_ij = 1e-12; 
  if (pi_ij > 1 - 1e-12)    pi_ij = 1 - 1e-12; 
  
  log_pi    = std::log(pi_ij); // Log(pi)
  log1m_pi  = std::log(1.0 - pi_ij); // Log(1 - pi)
}

// Log-likelihood for a single cell (i,j).
double hurdle_loglik_cell(double y,
                          double d_ij,
                          double alpha_t,
                          double r_t,
                          double log_pi,
                          double log1m_pi) {
  
  if (!std::isfinite(y) || y < 0) return 0.0;
  
  double logM = alpha_t - d_ij; 
  double M    = std::exp(logM); 
  
  // log density NB 
  auto logf_nb = [&](double yval)->double{
    return lgamma_safe(yval + r_t) 
    - lgamma_safe(r_t)
    - lgamma_safe(yval + 1.0)
    + r_t * std::log(r_t)
    - r_t * std::log(r_t + M)
    + yval * std::log(M)
    - yval * std::log(r_t + M);
  };
  
  // log f_NB(0)
  double logf0 = r_t * (std::log(r_t) - std::log(r_t + M)); 
  double one_minus_f0 = 1.0 - std::exp(logf0); // 1 - f_NB(0)
  if (one_minus_f0 < 1e-12) one_minus_f0 = 1e-12; 
  
  if (y == 0.0) {
    // For y=0: 
    return log1m_pi;
  } else {
    // For y>0: log pi + log f_NB(y) - log(1 - f_NB(0))
    return log_pi + logf_nb(y) - std::log(one_minus_f0);
  }
}


// Log-likelihood across all pairs (i,j) for one year.
double hurdle_loglik_one_year(const arma::mat& Y,
                              const arma::mat& X,
                              const arma::mat& D,
                              const arma::mat& W,
                              double alpha_t,
                              double beta_t,
                              double r_t,
                              bool include_diagonal){
  
  int n = Y.n_rows; 
  double ll = 0.0; 
  
  double pi_ij, log_pi, log1m_pi;
  logistic_hurdle(beta_t, pi_ij, log_pi, log1m_pi); 
  
  for (int i = 0; i < n; ++i) {
    
    for (int j = 0; j < n; ++j) {
      
      if(!include_diagonal && i==j) continue;
      
      if(W(i,j)==0) continue;
      
      double y   = Y(i,j); 
      double dij = D(i,j);
      
      ll += hurdle_loglik_cell(y, dij,
                               alpha_t, r_t,
                               log_pi, log1m_pi);
    }
  }
  return ll; 
}

// MH update alpha_t.
double mh_update_alpha(const arma::mat& Y,
                       const arma::mat& X,
                       const arma::mat& D,
                       const arma::mat& W, 
                       double alpha_t,
                       double beta_t,
                       double r_t,
                       double mu_alpha,
                       double sigma2_alpha,
                       double prop_sd,
                       bool include_diagonal) {
  
  double ll_old = hurdle_loglik_one_year(Y, X, D, W,
                                         alpha_t, beta_t, r_t, include_diagonal);
  
  double lp_old = -0.5 * std::pow(alpha_t - mu_alpha, 2) / sigma2_alpha;
  
  double alpha_new = alpha_t + R::rnorm(0.0, prop_sd);
  
  double ll_new = hurdle_loglik_one_year(Y, X, D, W,
                                         alpha_new, beta_t, r_t, include_diagonal);
  
  double lp_new = -0.5 * std::pow(alpha_new - mu_alpha, 2) / sigma2_alpha;
  
  double log_acc = (ll_new + lp_new) - (ll_old + lp_old);
  
  if (std::log(R::runif(0.0,1.0)) < log_acc)
    return alpha_new;
  return alpha_t;
}

// MH update beta_t.
double mh_update_beta(const arma::mat& Y,
                      const arma::mat& X,
                      const arma::mat& D,
                      const arma::mat& W, 
                      double beta_t,
                      double alpha_t,
                      double r_t,
                      double mu_beta,
                      double sigma2_beta,
                      double prop_sd, 
                      bool include_diagonal) {
  
  double ll_old = hurdle_loglik_one_year(Y, X, D, W,
                                         alpha_t, beta_t, r_t, include_diagonal);
  
  double lp_old = -0.5 * std::pow(beta_t - mu_beta, 2) / sigma2_beta;
  
  double beta_new = beta_t + R::rnorm(0.0, prop_sd);
  
  double ll_new = hurdle_loglik_one_year(Y, X, D, W,
                                         alpha_t, beta_new, r_t, include_diagonal);
  
  double lp_new = -0.5 * std::pow(beta_new - mu_beta, 2) / sigma2_beta;
  
  double log_acc = (ll_new + lp_new) - (ll_old + lp_old);
  
  if (std::log(R::runif(0.0,1.0)) < log_acc)
    return beta_new;
  return beta_t;
}

// MH update a = 1/sqrt(r)
double mh_update_r_halfnormal(const arma::mat& Y,
                              const arma::mat& X,
                              const arma::mat& D,
                              const arma::mat& W,
                              double alpha_t,
                              double beta_t,
                              double r_t,
                              double prop_sd_a,
                              double sigma_a,
                              bool include_diagonal) {
  
  double a_old = 1.0 / std::sqrt(r_t);
  
  double a_new = std::fabs(a_old + R::rnorm(0.0, prop_sd_a));
  
  double r_new = 1.0 / (a_new * a_new);
  
  double ll_old = hurdle_loglik_one_year(Y, X, D, W,
                                         alpha_t, beta_t, r_t,
                                         include_diagonal);
  
  double ll_new = hurdle_loglik_one_year(Y, X, D, W,
                                         alpha_t, beta_t, r_new,
                                         include_diagonal);
  
  double lp_old = -0.5 * (a_old*a_old)/(sigma_a*sigma_a);
  double lp_new = -0.5 * (a_new*a_new)/(sigma_a*sigma_a);
  
  double logJ_old = std::log(2.0) - 3.0*std::log(a_old);
  double logJ_new = std::log(2.0) - 3.0*std::log(a_new);
  
  double log_acc =
    (ll_new + lp_new + logJ_new) -
    (ll_old + lp_old + logJ_old);
  
  if (std::log(R::runif(0.0,1.0)) < log_acc)
    return r_new;
  
  return r_t;
}

// MH update latent coords node i year t.
bool mh_update_Xi(const arma::mat& Y,
                  std::vector<arma::mat>& X_list,
                  arma::mat& D_t,
                  int t_idx,
                  int i,
                  double alpha_t,
                  double beta_t,
                  double r_t,
                  const arma::mat& W,
                  double prop_sd,
                  double sigma2) {
  
  arma::mat& X_t = X_list[t_idx];
  int n = X_t.n_rows;
  
  double pi_ij, log_pi, log1m_pi;
  logistic_hurdle(beta_t, pi_ij, log_pi, log1m_pi); 
  
  double ll_old = 0.0;
  
  for (int j = 0; j < n; ++j) {
    if (j == i) continue;
    
    if(W(i,j)==0) continue;          
    if(W(j,i)==0) continue;    
    
    double y_ij = Y(i,j);
    double y_ji = Y(j,i);
    
    double d_ij = D_t(i,j); 
    double d_ji = D_t(j,i); 
    
    ll_old += hurdle_loglik_cell(y_ij, d_ij,
                                 alpha_t, r_t,
                                 log_pi, log1m_pi);
    
    ll_old += hurdle_loglik_cell(y_ji, d_ji,
                                 alpha_t, r_t,
                                 log_pi, log1m_pi);
  }
  
  arma::rowvec Xi_old = X_t.row(i); 
  
  double lp_old = 0.0;
  
  if (t_idx == 0) {
    lp_old += -0.5 * arma::dot(Xi_old, Xi_old) / sigma2;
    
    if (X_list.size() > 1)
      lp_old += -0.5 * accu(square(X_list[1].row(i) - Xi_old)) / sigma2;
    
  } else {
    lp_old += -0.5 * accu(square(Xi_old - X_list[t_idx-1].row(i))) / sigma2;
    
    if (t_idx + 1 < (int)X_list.size())
      lp_old += -0.5 * accu(square(X_list[t_idx+1].row(i) - Xi_old)) / sigma2;
  }
  
  arma::rowvec Xi_new = Xi_old;
  Xi_new(0) += R::rnorm(0.0, prop_sd); 
  Xi_new(1) += R::rnorm(0.0, prop_sd); 
  
  double ll_new = 0.0;
  
  for (int j = 0; j < n; ++j) {
    if (j == i) continue;
    
    if(W(i,j)==0) continue;          
    if(W(j,i)==0) continue;
    
    arma::rowvec Xj = X_t.row(j); 
    
    double dx = Xi_new(0) - Xj(0); 
    double dy = Xi_new(1) - Xj(1); 
    double dij_new = std::sqrt(dx*dx + dy*dy); 
    
    double y_ij = Y(i,j);
    double y_ji = Y(j,i);
    
    ll_new += hurdle_loglik_cell(y_ij, dij_new,
                                 alpha_t, r_t,
                                 log_pi, log1m_pi);
    
    ll_new += hurdle_loglik_cell(y_ji, dij_new,
                                 alpha_t, r_t,
                                 log_pi, log1m_pi);
  }
  
  double lp_new = 0.0;
  
  if (t_idx == 0) {
    
    lp_new += -0.5 * dot(Xi_new, Xi_new) / sigma2; 
    
    if (X_list.size() > 1)
      lp_new += -0.5 * accu(square(X_list[1].row(i) - Xi_new)) / sigma2;
    
  } else {
    
    lp_new += -0.5 * accu(square(Xi_new - X_list[t_idx-1].row(i))) / sigma2;
    
    if (t_idx + 1 < (int)X_list.size())
      lp_new += -0.5 * accu(square(X_list[t_idx+1].row(i) - Xi_new)) / sigma2;
  }
  
  double log_acc = (ll_new + lp_new) - (ll_old + lp_old);
  
  if (std::log(R::runif(0.0, 1.0)) < log_acc) {
    
    X_t.row(i) = Xi_new; 
    
    for (int j = 0; j < n; ++j) {
      if (j == i) {
        D_t(i,j) = 0.0; 
      } else {
        arma::rowvec Xj = X_t.row(j);
        double dx = Xi_new(0) - Xj(0);
        double dy = Xi_new(1) - Xj(1);
        double dij = std::sqrt(dx*dx + dy*dy);
        
        D_t(i,j) = dij; 
        D_t(j,i) = dij; 
      }
    }
    return true; 
  }
  return false;
}

// Update all coords of X_i,t.
int mh_update_X_year(const arma::mat& Y,
                     std::vector<arma::mat>& X_list,
                     arma::mat& D_t,
                     const arma::mat& W,
                     int t_idx,
                     double alpha_t,
                     double beta_t,
                     double r_t,
                     double prop_sd,
                     double sigma2) {
  
  int n = X_list[t_idx].n_rows;
  int acc = 0; 
  
  for (int i = 0; i < n; ++i) {
    bool ok = mh_update_Xi(Y,
                           X_list,
                           D_t,
                           t_idx,
                           i,
                           alpha_t,
                           beta_t,
                           r_t,
                           W,
                           prop_sd,
                           sigma2);
    if (ok) acc++; 
  }
  
  if (t_idx > 0)
    X_list[t_idx] = procrustes(X_list[t_idx], X_list[t_idx - 1]); 
  else
    enforce_center(X_list[t_idx]); 
  
  D_t = pairwise_dist(X_list[t_idx]);
  
  return acc; 
}

// Joint update r_t
arma::vec mh_update_r_joint_halfnormal(
    const std::vector<arma::mat>& Y_list,
    const std::vector<arma::mat>& X_list,
    const std::vector<arma::mat>& D_list,
    const std::vector<arma::mat>& W_list,
    const arma::vec& r_vec,
    const arma::vec& alpha_vec,
    const arma::vec& beta_vec,
    double prop_sd_a,
    double sigma_a,
    bool include_diagonal
) {
  int Tn = r_vec.size();
  
  arma::vec r_new(Tn);
  arma::vec a_old(Tn);
  arma::vec a_new(Tn);
  
  for (int t=0; t<Tn; t++) {
    a_old(t) = 1.0 / std::sqrt(r_vec(t));
    a_new(t) = std::fabs(a_old(t) + R::rnorm(0.0, prop_sd_a));
    r_new(t) = 1.0 / (a_new(t)*a_new(t));
  }
  
  double ll_old = 0.0, ll_new = 0.0;
  
  for (int t=0; t<Tn; t++) {
    ll_old += hurdle_loglik_one_year(
      Y_list[t], X_list[t], D_list[t], W_list[t],
                                             alpha_vec(t), beta_vec(t), r_vec(t),
                                             include_diagonal);
    
    ll_new += hurdle_loglik_one_year(
      Y_list[t], X_list[t], D_list[t], W_list[t],
                                             alpha_vec(t), beta_vec(t), r_new(t),
                                             include_diagonal);
  }
  
  double lp_old = 0.0, lp_new = 0.0;
  for (int t=0; t<Tn; t++) {
    lp_old += -0.5 * (a_old(t)*a_old(t))/(sigma_a*sigma_a);
    lp_new += -0.5 * (a_new(t)*a_new(t))/(sigma_a*sigma_a);
  }
  
  double logJ_old = 0.0, logJ_new = 0.0;
  for (int t=0; t<Tn; t++) {
    logJ_old += std::log(2.0) - 3.0*std::log(a_old(t));
    logJ_new += std::log(2.0) - 3.0*std::log(a_new(t));
  }
  
  double log_acc = (ll_new + lp_new + logJ_new)
    - (ll_old + lp_old + logJ_old);
  
  if (std::log(R::runif(0.0,1.0)) < log_acc)
    return r_new;
  
  return r_vec;
}


/* ==============================================================
 WRAPPER MCMC
 ============================================================== */
// [[Rcpp::export]]
List mcmc_base(List Y_list_R,
               List X_list_R,
               List W_list_R,
               
               arma::vec alpha_vec,
               arma::vec beta_vec,
               arma::vec r_vec,
               
               double sigma2_alpha,
               double sigma2_beta,
               
               int n_iter,
               int burn_in,
               
               double prop_sd_X,
               double prop_sd_alpha,
               double prop_sd_beta,
               double prop_sd_r,
               
               double sigma2,
               
               double sigma_a, 
               
               double mu_alpha,          
               double mu_beta,           
               
               bool include_diagonal = false,
               bool joint_update_r = false,
               bool verbose = true) {
  
  int Tn = Y_list_R.size(); 
  
  std::vector<arma::mat> Y_list(Tn); 
  std::vector<arma::mat> X_list(Tn); 
  std::vector<arma::mat> W_list(Tn);
  
  for (int t = 0; t < Tn; ++t) {
    Y_list[t] = as<arma::mat>(Y_list_R[t]); 
    X_list[t] = as<arma::mat>(X_list_R[t]); 
    W_list[t] = as<arma::mat>(W_list_R[t]);
  }
  
  int n_nodes = Y_list[0].n_rows; 
  int n_save  = n_iter - burn_in; 
  
  std::vector<arma::mat> D_list(Tn); 
  for (int t = 0; t < Tn; ++t) {
    D_list[t] = pairwise_dist(X_list[t]); 
  }
  
  /* --------------------------------------------------------------
   Storage
   -------------------------------------------------------------- */
  arma::mat alpha_save(n_save, Tn, fill::zeros);
  arma::mat beta_save (n_save, Tn, fill::zeros);
  arma::mat r_save    (n_save, Tn, fill::zeros);
  
  arma::vec mu_alpha_save(n_save, fill::zeros);
  arma::vec mu_beta_save (n_save, fill::zeros);
  
  List X_save(n_save);
  
  arma::vec loglik_save(n_save, fill::zeros);   
  
  arma::ivec acc_X(Tn, fill::zeros),     trials_X(Tn, fill::zeros);
  arma::ivec acc_alpha(Tn, fill::zeros), trials_alpha(Tn, fill::zeros);
  arma::ivec acc_beta(Tn, fill::zeros),  trials_beta(Tn, fill::zeros);
  arma::ivec acc_r(Tn, fill::zeros),     trials_r(Tn, fill::zeros);
  
  /* --------------------------------------------------------------
    MCMC
   -------------------------------------------------------------- */
  for (int iter = 1; iter <= n_iter; ++iter) {
    
    /* ------------------------------
     Update X_t 
     ------------------------------ */
    for (int t = 0; t < Tn; ++t) {
      int acc = mh_update_X_year(Y_list[t],
                                 X_list,
                                 D_list[t],
                                       W_list[t],
                                             t,
                                             alpha_vec(t),
                                             beta_vec(t),
                                             r_vec(t),
                                             prop_sd_X,
                                             sigma2);
      
      acc_X(t)     += acc; 
      trials_X(t)  += n_nodes; 
    }
    
    
    /* ------------------------------
     Update alpha_t
     ------------------------------ */
    for (int t = 0; t < Tn; ++t) {
      
      double alpha_old = alpha_vec(t); 
      double alpha_new = mh_update_alpha(Y_list[t],
                                         X_list[t],
                                               D_list[t],
                                                     W_list[t],
                                                           alpha_old,
                                                           beta_vec(t),
                                                           r_vec(t),
                                                           mu_alpha,
                                                           sigma2_alpha,
                                                           prop_sd_alpha,
                                                           include_diagonal);
      
      if (alpha_new != alpha_old) acc_alpha(t)++; 
      
      trials_alpha(t)++; 
      alpha_vec(t) = alpha_new; 
    }
    
    
    /* ------------------------------
     Update beta_t
     ------------------------------ */
    for (int t = 0; t < Tn; ++t) {
      
      double beta_old = beta_vec(t); 
      
      double beta_new = mh_update_beta(Y_list[t],
                                       X_list[t],
                                             D_list[t],
                                                   W_list[t],
                                                         beta_old,
                                                         alpha_vec(t),
                                                         r_vec(t),
                                                         mu_beta,
                                                         sigma2_beta,
                                                         prop_sd_beta,
                                                         include_diagonal);
      
      if (beta_new != beta_old) acc_beta(t)++; 
      
      trials_beta(t)++; 
      beta_vec(t) = beta_new; 
    }
    
    
    /* ------------------------------
     Update r_t via a = 1/sqrt(r)
     ------------------------------ */
    
    if (joint_update_r) {
      
      arma::vec r_old_vec = r_vec;
      
      arma::vec r_new_vec =
        mh_update_r_joint_halfnormal(
          Y_list, X_list, D_list, W_list,
          r_vec,
          alpha_vec,
          beta_vec,
          prop_sd_r,
          sigma_a,
          include_diagonal);
      
      if (!approx_equal(r_new_vec, r_old_vec, "absdiff", 0)) {
        for (int t=0; t<Tn; ++t) acc_r(t)++;
      }
      for (int t=0; t<Tn; ++t) trials_r(t)++;
      
      r_vec = r_new_vec;
      
    } else {
      
      for (int t=0; t<Tn; ++t) {
        
        double r_old = r_vec(t);
        
        double r_new = mh_update_r_halfnormal(
          Y_list[t],
                X_list[t],
                      D_list[t],
                            W_list[t],
                                  alpha_vec(t),
                                  beta_vec(t),
                                  r_old,
                                  prop_sd_r,
                                  sigma_a,
                                  include_diagonal);
        
        if (r_new != r_old) acc_r(t)++;
        trials_r(t)++;
        
        r_vec(t) = r_new;
      }
    }
    
    
    
    /* --------------------------------------------------------------
     Tuning proposals  
     -------------------------------------------------------------- */
    if (iter <= burn_in && (iter % 100 == 0)) { 
      
      for (int t = 0; t < Tn; ++t) {
        
        double acc_rate_alpha = (double)acc_alpha(t) / trials_alpha(t); 
        if (acc_rate_alpha < 0.3) prop_sd_alpha *= 0.8; 
        else if (acc_rate_alpha > 0.6) prop_sd_alpha *= 1.2; 
        
        double acc_rate_beta = (double)acc_beta(t) / trials_beta(t); 
        if (acc_rate_beta < 0.3) prop_sd_beta *= 0.8;
        else if (acc_rate_beta > 0.6) prop_sd_beta *= 1.2;
        
        double acc_rate_X = (double)acc_X(t) / trials_X(t); 
        if (acc_rate_X < 0.2) prop_sd_X *= 0.8;
        else if (acc_rate_X > 0.5) prop_sd_X *= 1.2;
        
        double acc_rate_r = (double)acc_r(t) / trials_r(t);  
        if (acc_rate_r < 0.2) prop_sd_r *= 0.8;
        else if (acc_rate_r > 0.5) prop_sd_r *= 1.2;
        
        acc_alpha(t) = trials_alpha(t) = 0;
        acc_beta(t)  = trials_beta(t)  = 0;
        acc_X(t)     = trials_X(t)     = 0;
        acc_r(t)     = trials_r(t)     = 0;
      }
    }
    
    
    /* --------------------------------------------------------------
     Save
     -------------------------------------------------------------- */
    
    if (iter > burn_in) { 
      int idx = iter - burn_in - 1; 
      
      double total_ll = 0.0;   
      for (int t = 0; t < Tn; ++t) {
        total_ll += hurdle_loglik_one_year(
          Y_list[t], X_list[t], D_list[t], W_list[t],
                                                 alpha_vec(t), beta_vec(t), r_vec(t),
                                                 include_diagonal
        );
      }
      loglik_save(idx) = total_ll; 
      
      alpha_save.row(idx) = alpha_vec.t();
      beta_save.row(idx)  = beta_vec.t();
      r_save.row(idx)     = r_vec.t();
      
      mu_alpha_save(idx)  = mu_alpha;
      mu_beta_save(idx)   = mu_beta;
      
      List X_this_iter(Tn);
      for (int tt = 0; tt < Tn; ++tt)
        X_this_iter[tt] = X_list[tt];
      
      X_save[idx] = X_this_iter;
      
    }
    
    
    /* --------------------------------------------------------------
     Verbose
     -------------------------------------------------------------- */
    
    if (verbose && (iter % 200 == 0)) {
      Rcpp::Rcout
      << "Iter " << iter
      << " | prop_sd_X=" << prop_sd_X
      << " | prop_sd_alpha=" << prop_sd_alpha
      << " | prop_sd_beta="  << prop_sd_beta
      << " | prop_sd_r="     << prop_sd_r
      << std::endl;
    }
  }
  
  /* --------------------------------------------------------------
   DIC 
  -------------------------------------------------------------- */
  
  arma::vec alpha_hat = arma::mean(alpha_save, 0).t();
  arma::vec beta_hat  = arma::mean(beta_save, 0).t();
  arma::vec r_hat     = arma::mean(r_save, 0).t();
  
  double Dbar = -2.0 * arma::mean(loglik_save);
  
  double loglik_hat = 0.0;
  
  for (int t = 0; t < Tn; ++t) {
    loglik_hat += hurdle_loglik_one_year(
      Y_list[t],
            X_list[t],      
                  D_list[t],
                        W_list[t],
                              alpha_hat(t),
                              beta_hat(t),
                              r_hat(t),
                              include_diagonal
    );
  }
  
  double D_hat = -2.0 * loglik_hat;
  double DIC = 2.0 * Dbar - D_hat;
  

  /* --------------------------------------------------------------
   OUTPUT
   -------------------------------------------------------------- */
  
  List X_out(Tn);
  for (int t = 0; t < Tn; ++t)
    X_out[t] = X_list[t];
  
  return List::create(
    _["alpha_samples"]     = alpha_save,
    _["beta_samples"]      = beta_save,
    _["r_samples"]         = r_save,
    _["mu_alpha_samples"]  = mu_alpha_save,
    _["mu_beta_samples"]   = mu_beta_save,
    
    _["X_last"]            = X_out,
    _["X_samples"]     = X_save,
    
    _["sigma2_alpha_last"] = sigma2_alpha,
    _["sigma2_beta_last"]  = sigma2_beta,
    
    _["mu_alpha_last"]     = mu_alpha,
    _["mu_beta_last"]      = mu_beta,
    
    _["prop_sd_X_last"]    = prop_sd_X,
    _["prop_sd_alpha_last"]= prop_sd_alpha,
    _["prop_sd_beta_last"] = prop_sd_beta,
    _["prop_sd_r_last"]    = prop_sd_r,
    _["loglik_samples"] = loglik_save, 
    _["Dbar"] = Dbar,
    _["D_hat"] = D_hat,
    _["DIC"] = DIC
  );
}
