# =============================================================
# FILE: base_models.R
# PURPOSE: Define lasso, Random Forest, and XGBoost as interchangeable base models
#
# The Key Idea — why we need this file?
# The R-learner (and S/T/X learners) are just frameworks. They describe how to structure the estimation problem, not which model to use inside.
# Think of it like a sandwich recipe that says "spread your choice of filling" — the bread and structure are fixed, but you can swap tuna for chicken.
# Here, the "bread" is the learner structure (S/T/X/R). The "filling" is the base model (lasso/RF/XGBoost).
#
# How the interface works:
# Every base model in this file is a function that returns a list with exactly two things:
#   $fit  — a function(X, y) that trains a model and returns it
#   $pred — a function(model, X_new) that predicts from the model
# The learners only ever call $fit and $pred. They never look inside. This is called "abstraction."
# This means: to add a new model (e.g. neural network), we only need to add it here, not change any learner code.

# =============================================================
library(glmnet)   # lasso
library(ranger)   # random forest (faster than grf for prediction)
library(xgboost)  # gradient boosting



# =============================================================
# BASE MODEL 1: LASSO
# What It Is:
# Linear regression with a penalty that shrinks small coefficients to exactly zero — effectively selecting the most important features. This is what the paper uses in Section 6.2.
# Strengths: fast, interpretable, works well when tau(X) is roughly linear in X.
# Weaknesses: can't capture nonlinear relationships or interactions between covariates unless we manually add them.
# 
# alpha=1 means pure lasso (vs alpha=0 which is ridge).
# cv.glmnet automatically picks the best penalty strength via cross-validation.
# =============================================================

make_lasso <- function() {
  list(
    # fit: train a lasso model on (X, y)
    # returns the fitted cv.glmnet object
    fit = function(X, y, ...) {
      # family argument handles binary outcomes (W) vs continuous (Y)
      # we pass it through ... so the learner can specify it
      cv.glmnet(X, y, alpha = 1, ...)
    },
    
    # pred: predict from a fitted lasso model on new X
    # always uses lambda.min = the penalty that minimised CV error
    pred = function(model, X_new, ...) {
      as.numeric(predict(model, X_new, s = "lambda.min", ...))
    }
    
  )
}



# =============================================================
# BASE MODEL 2: RANDOM FOREST (via ranger)
# What It Is:
# Grows many decision trees on random subsets of data, then averages their predictions. Each tree captures different patterns — averaging reduces overfitting.
# Strengths: naturally captures nonlinear effects and interactions without any manual feature engineering. Works well out of the box without much tuning.
# Also closer to what the paper uses in Section 6.3 (KRR was the original, but RF is the modern practical replacement).
# Weaknesses: slower than lasso, less interpretable, can overfit on small samples.
#
# num.trees=500: grow 500 trees (standard default)
# min.node.size=5: each leaf has at least 5 observations (prevents overfitting on tiny subgroups)
# =============================================================

make_rf <- function(num.trees = 500, min.node.size = 5) {
  list(
    fit = function(X, y, ...) {
      # ranger needs a data frame, not a matrix
      # we combine X and y into one data frame for training
      df <- as.data.frame(X)
      df$y_outcome <- y
      
      ranger(
        y_outcome ~ .,          # predict y_outcome from all other columns
        data          = df,
        num.trees     = num.trees,
        min.node.size = min.node.size,
        # case.weights handles the weighted regression needed by R-learner
      )
    },
    
    pred = function(model, X_new, ...) {
      df_new <- as.data.frame(X_new)
      as.numeric(predict(model, df_new)$predictions)
    }
    
  )
}



# =============================================================
# BASE MODEL 3: XGBOOST
# What It Is:
# Builds trees sequentially — each new tree corrects the errors of all previous trees. Very powerful in practice. This is what the paper uses in Section 6.4.
# Strengths: often the most accurate on tabular data, handles nonlinearity and interactions automatically, faster than RF for large datasets.
# Weaknesses: more hyperparameters to tune, can overfit badly if not regularised properly.
#
# Our settings (conservative, works well for n=500):
#   nrounds=200      — number of trees to build
#   max_depth=4      — how deep each tree can grow (shallow = less overfit)
#   eta=0.1          — learning rate (how much each tree contributes)
#   subsample=0.8    — use 80% of data per tree (reduces overfitting)
#   early_stopping=20 — stop if no improvement after 20 rounds
# =============================================================

make_xgb <- function(nrounds      = 200,
                     max_depth    = 4,
                     eta          = 0.1,
                     subsample    = 0.8) {
  list(
    
    fit = function(X, y, weights = NULL, ...) {
      # xgboost needs its own special data format called DMatrix
      dtrain <- xgb.DMatrix(
        data   = as.matrix(X),
        label  = y,
        weight = if (!is.null(weights)) weights else rep(1, length(y))
      )
      
      xgb.train(
        params = list(
          objective = "reg:squarederror",  # regression task
          max_depth = max_depth,
          eta       = eta,
          subsample = subsample
        ),
        data    = dtrain,
        nrounds = nrounds,
        verbose = 0   # silent — don't print progress every round
      )
    },
    
    pred = function(model, X_new, ...) {
      dtest <- xgb.DMatrix(data = as.matrix(X_new))
      as.numeric(predict(model, dtest))
    }
    
  )
}

# Tuning Status:
#   Lasso  — DONE: cv.glmnet does 10-fold CV internally
#   RF     — TODO: currently uses fixed defaults (num.trees=500, min.node.size=5). Need to add CV over min.node.size in c(1, 5, 10, 20)
#   XGBoost — TODO: currently uses fixed defaults (max_depth=4,eta=0.1). Need to add CV over max_depth in c(3,4,6) and eta in c(0.05, 0.1, 0.2)
#
# Why?
# The paper (Section 6.2) explicitly uses cross-validated penalty selection. Our RF and XGBoost use sensible defaults but not formally tuned values. We can note this as a limitation in the report or even work on (if time allows) before final submission.