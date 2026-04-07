# cvkern.R
# Kernel ridge regression with Gaussian kernel and cross-validated hyperparameter search.
# Rewritten to use kernlab::gausspr / kernlab::ksvm instead of the defunct KRLS2 package.
#
# kernlab is on CRAN and actively maintained. Install via: install.packages("kernlab")
#
# The Gaussian kernel in kernlab is parameterised by `sigma` (= 1 / (2 * b^2) in some
# conventions). Here we keep the paper's convention and search over bandwidth b, converting
# internally: sigma_kernlab = 1 / b.

#' Kernel ridge regression (Gaussian kernel) with cross-validated hyperparameters
#'
#' Fits KRR using \code{kernlab::gausspr} over a grid of kernel bandwidth values \code{b}
#' and ridge penalty values \code{lambda}.  Hyperparameters are selected by k-fold
#' cross-validation on (weighted) MSE.
#'
#' @param x       Numeric feature matrix (n x p).
#' @param y       Numeric response vector (length n).
#' @param weights Optional numeric weight vector (length n). If NULL, OLS weighting.
#' @param k_folds Number of CV folds.
#' @param b_range Grid of Gaussian kernel bandwidth values (search over these).
#' @param lambda_range Grid of ridge penalty values (search over these).
#'
#' @return A list with elements:
#'   \item{b}{Best bandwidth.}
#'   \item{lambda}{Best ridge penalty.}
#'   \item{fit}{Cross-validated in-sample predictions (length n).}
#'   \item{model}{Final model object (kernlab ksvm) fitted on all data with best params.}
#'
#' @examples
#' \dontrun{
#' n <- 200; p <- 5
#' x <- matrix(rnorm(n * p), n, p)
#' y <- x[,1] + x[,2]^2 + rnorm(n)
#' fit <- cv_klrs(x, y, k_folds = 5)
#' pred <- predict(fit$model, x)
#' }
#'
#' @importFrom kernlab ksvm predict
#' @export
cv_klrs <- function(x,
                    y,
                    weights    = NULL,
                    k_folds    = 5,
                    b_range    = 10^(seq(-3, 3, 0.5)),
                    lambda_range = 10^(seq(-3, 3, 0.5))) {
  
  if (!requireNamespace("kernlab", quietly = TRUE)) {
    stop("Package 'kernlab' is required. Install it with: install.packages('kernlab')")
  }
  
  n <- length(y)
  if (is.null(k_folds) || k_folds < 2) k_folds <- 5
  
  # Create balanced fold IDs
  foldid <- sample(rep(seq_len(k_folds), length.out = n))
  
  # ---- Helper: fit one (b, lambda) combination ----
  fit_one <- function(b, lambda, train_idx, pred_idx, w_train = NULL) {
    x_tr <- x[train_idx, , drop = FALSE]
    y_tr <- y[train_idx]
    x_te <- x[pred_idx, , drop = FALSE]
    
    # kernlab::ksvm with rbfdot kernel.
    # sigma in kernlab = 1/b (inverse bandwidth).
    sigma_val <- 1 / b
    
    if (is.null(w_train)) {
      model <- kernlab::ksvm(
        x_tr, y_tr,
        kernel   = "rbfdot",
        kpar     = list(sigma = sigma_val),
        type     = "eps-svr",      # epsilon-SVR ~ ridge with L2 loss
        C        = 1 / lambda,     # C = 1/lambda maps ridge penalty
        epsilon  = 0,
        scaled   = FALSE
      )
    } else {
      # kernlab ksvm does not natively support observation weights for regression.
      # We approximate weighted KRR by solving the normal equations directly.
      model <- .weighted_krr(x_tr, y_tr, sigma_val, lambda, w_train)
    }
    
    preds <- as.numeric(kernlab::predict(model, x_te))
    list(model = model, preds = preds)
  }
  
  best_mse  <- Inf
  best_b    <- b_range[1]
  best_lam  <- lambda_range[1]
  best_fit  <- rep(0, n)
  
  for (b in b_range) {
    for (lambda in lambda_range) {
      cv_preds <- rep(NA_real_, n)
      
      tryCatch({
        for (f in seq_len(k_folds)) {
          tr  <- which(foldid != f)
          te  <- which(foldid == f)
          w_f <- if (!is.null(weights)) weights[tr] else NULL
          out <- fit_one(b, lambda, tr, te, w_f)
          cv_preds[te] <- out$preds
        }
        
        if (!is.null(weights)) {
          mse <- mean(weights * (y - cv_preds)^2, na.rm = TRUE)
        } else {
          mse <- mean((y - cv_preds)^2, na.rm = TRUE)
        }
        
        if (is.finite(mse) && mse < best_mse) {
          best_mse   <- mse
          best_b     <- b
          best_lam   <- lambda
          best_fit   <- cv_preds
        }
      }, error = function(e) {
        # Skip this (b, lambda) combination silently if fitting fails
        NULL
      })
    }
  }
  
  # Fit final model on all data with best (b, lambda)
  sigma_best <- 1 / best_b
  w_all <- if (!is.null(weights)) weights else NULL
  
  if (is.null(w_all)) {
    final_model <- kernlab::ksvm(
      x, y,
      kernel  = "rbfdot",
      kpar    = list(sigma = sigma_best),
      type    = "eps-svr",
      C       = 1 / best_lam,
      epsilon = 0,
      scaled  = FALSE
    )
  } else {
    final_model <- .weighted_krr(x, y, sigma_best, best_lam, w_all)
  }
  
  list(
    b      = best_b,
    lambda = best_lam,
    fit    = best_fit,
    model  = final_model
  )
}


# ---------------------------------------------------------------------------
# Internal: Weighted kernel ridge regression via normal equations
# ---------------------------------------------------------------------------
# Solves (K W K + lambda * K)^{-1} K W y  =  (W K + lambda I)^{-1} W y
# where K is the n x n Gaussian kernel matrix.
# Returns a minimal object that supports predict().
.weighted_krr <- function(x_tr, y_tr, sigma, lambda, weights) {
  n <- nrow(x_tr)
  
  # Build kernel matrix
  K <- .gauss_kernel_matrix(x_tr, x_tr, sigma)
  
  W  <- diag(weights)
  A  <- W %*% K + lambda * diag(n)
  alpha <- tryCatch(
    solve(A, W %*% y_tr),
    error = function(e) {
      # Fallback: use pseudoinverse via svd
      sv  <- svd(A)
      tol <- max(dim(A)) * .Machine$double.eps * sv$d[1]
      inv_d <- ifelse(sv$d > tol, 1 / sv$d, 0)
      sv$v %*% diag(inv_d) %*% t(sv$u) %*% (W %*% y_tr)
    }
  )
  
  structure(
    list(alpha   = alpha,
         x_train = x_tr,
         sigma   = sigma,
         lambda  = lambda),
    class = "weighted_krr"
  )
}

#' @export
predict.weighted_krr <- function(object, newdata, ...) {
  K_new <- .gauss_kernel_matrix(newdata, object$x_train, object$sigma)
  as.numeric(K_new %*% object$alpha)
}

# Build n1 x n2 Gaussian kernel matrix between rows of x1 and x2
# K(x, z) = exp(-sigma * ||x - z||^2)
.gauss_kernel_matrix <- function(x1, x2, sigma) {
  # Efficient squared-distance computation
  x1 <- as.matrix(x1)
  x2 <- as.matrix(x2)
  sq1 <- rowSums(x1^2)
  sq2 <- rowSums(x2^2)
  # ||x - z||^2 = ||x||^2 - 2<x,z> + ||z||^2
  D2 <- outer(sq1, sq2, "+") - 2 * tcrossprod(x1, x2)
  D2 <- pmax(D2, 0)   # numerical safety
  exp(-sigma * D2)
}

