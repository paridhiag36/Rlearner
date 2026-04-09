# =============================================================
# FILE: learners.R
# PURPOSE: S, T, X, R learner functions that accept any base model
#
# The learner structure remains the same regardless of base model. Only the fitting and predicting lines change.
# Eg:
#   lasso <- make_lasso()
#   tau_hat <- r_learner(X_train, Y_train, W_train, X_test, base_model = lasso)
#   rf <- make_rf()
#   tau_hat <- r_learner(X_train, Y_train, W_train, X_test, base_model = rf)
# =============================================================


source("code/base_models.R")


# =============================================================
# S-LEARNER
# Single model: W is just another covariate
# =============================================================
s_learner <- function(X_train, Y_train, W_train, X_test, base_model = make_lasso()) {
  # stack W as extra column — it's just another feature here
  XW_train <- cbind(X_train, W = W_train)
  XW_test1 <- cbind(X_test,  W = rep(1, nrow(X_test)))  # everyone treated
  XW_test0 <- cbind(X_test,  W = rep(0, nrow(X_test)))  # nobody treated
  
  # fit ONE model on all data
  model <- base_model$fit(XW_train, Y_train)
  
  # predict under treatment and control, take difference
  y1 <- base_model$pred(model, XW_test1)
  y0 <- base_model$pred(model, XW_test0)
  
  return(y1 - y0)
}


# =============================================================
# T-LEARNER
# Two separate models, one per treatment arm
# =============================================================
t_learner <- function(X_train, Y_train, W_train, X_test, base_model = make_lasso()) {
  treated <- W_train == 1
  control <- W_train == 0
  
  # fit separate model for treated and control groups
  model1 <- base_model$fit(X_train[treated, ], Y_train[treated])
  model0 <- base_model$fit(X_train[control, ], Y_train[control])
  
  # predict for everyone in test set from both models
  mu1 <- base_model$pred(model1, X_test)
  mu0 <- base_model$pred(model0, X_test)
  
  return(mu1 - mu0)
}


# =============================================================
# X-LEARNER
# Cross-fitted pseudo-outcomes, propensity-weighted combination
# =============================================================
x_learner <- function(X_train, Y_train, W_train, X_test, base_model = make_lasso()) {
  treated <- W_train == 1
  control <- W_train == 0
  
  # stage 1: base outcome models (same as T-learner)
  model1 <- base_model$fit(X_train[treated, ], Y_train[treated])
  model0 <- base_model$fit(X_train[control, ], Y_train[control])
  
  # cross-predictions: control model on treated people, and vice versa
  mu0_on_treated <- base_model$pred(model0, X_train[treated, ])
  mu1_on_control <- base_model$pred(model1, X_train[control, ])
  
  # stage 2: pseudo-outcomes
  # for treated: actual - what they'd have gotten without treatment
  # for control: what they'd have gotten with treatment - actual
  D1 <- Y_train[treated] - mu0_on_treated
  D0 <- mu1_on_control   - Y_train[control]
  
  # stage 3: fit tau models on pseudo-outcomes
  tau_model1 <- base_model$fit(X_train[treated, ], D1)
  tau_model0 <- base_model$fit(X_train[control, ], D0)
  
  tau1 <- base_model$pred(tau_model1, X_test)
  tau0 <- base_model$pred(tau_model0, X_test)
  
  # stage 4: propensity-weighted combination
  # always use lasso for propensity regardless of base_model
  # (propensity is a classification problem — keep it separate)
  e_fit <- cv.glmnet(X_train, W_train, alpha = 1, family = "binomial")
  e_hat <- as.numeric(predict(e_fit, X_test, s = "lambda.min", type = "response"))
  e_hat <- pmin(pmax(e_hat, 0.05), 0.95)
  
  return((1 - e_hat) * tau1 + e_hat * tau0)
}


# =============================================================
# R-LEARNER
# Robinson residualisation with cross-fitting
# Our main model
# =============================================================
r_learner <- function(X_train, Y_train, W_train, X_test, base_model = make_lasso()) {
  n     <- nrow(X_train)
  folds <- sample(rep(1:5, length.out = n))
  
  m_hat <- numeric(n)   # E[Y|X] estimates
  e_hat <- numeric(n)   # E[W|X] estimates
  
  # --- cross-fitted nuisance estimation ---
  # for each fold: train on the other 4 folds, predict on this fold
  # this keeps nuisance estimation and tau estimation independent
  for (k in 1:5) {
    in_k  <- folds == k
    out_k <- folds != k
    
    # m(X): expected outcome ignoring treatment
    m_mod       <- base_model$fit(X_train[out_k, ], Y_train[out_k])
    m_hat[in_k] <- base_model$pred(m_mod, X_train[in_k, ])
    
    # e(X): propensity score — always use lasso with binomial
    # WHY always lasso here? because e(X) is a probability (0-1)
    # and lasso with family="binomial" handles this correctly.
    # RF and XGBoost need extra work for calibrated probabilities.
    e_mod       <- cv.glmnet(X_train[out_k, ], W_train[out_k],
                             alpha = 1, family = "binomial")
    e_hat[in_k] <- as.numeric(predict(e_mod, X_train[in_k, ],
                                      s    = "lambda.min",
                                      type = "response"))
  }
  
  e_hat <- pmin(pmax(e_hat, 0.05), 0.95)
  
  # --- form residuals ---
  Y_tilde <- Y_train - m_hat   # outcome residual
  W_tilde <- W_train - e_hat   # treatment residual
  
  # --- weighted regression for tau ---
  # pseudo_Y = Y_tilde / W_tilde approximates tau(X)
  # weights  = W_tilde^2 downweights near-deterministic treatment
  pseudo_Y <- Y_tilde / W_tilde
  weights  <- W_tilde^2
  
  # fit tau model — this IS where the base model matters
  # for RF and XGBoost we pass weights through
  tau_mod <- tryCatch(
    base_model$fit(X_train, pseudo_Y, weights = weights),
    error = function(e) base_model$fit(X_train, pseudo_Y)
  )
  
  return(base_model$pred(tau_mod, X_test))
}