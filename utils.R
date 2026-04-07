# utils.R
# Utility functions replicated from the rlearner package (Nie & Wager, 2021)
# These replace the @include utils.R dependency in the original source files.

#' Sanitize and validate (x, w, y) inputs
#'
#' Converts x to a numeric matrix, w and y to numeric vectors, and checks
#' for compatible dimensions.
#'
#' @param x feature matrix (n x p)
#' @param w treatment vector (binary, 0/1)
#' @param y outcome vector (real-valued)
#' @return list with validated x, w, y
sanitize_input <- function(x, w, y) {
  if (is.data.frame(x)) x <- model.matrix(~. - 1, x)
  if (!is.matrix(x))    x <- as.matrix(x)
  if (!is.numeric(x))   x <- apply(x, 2, as.numeric)
  
  w <- as.numeric(w)
  y <- as.numeric(y)
  
  n <- nrow(x)
  if (length(w) != n) stop("Length of w must equal nrow(x).")
  if (length(y) != n) stop("Length of y must equal nrow(x).")
  if (any(is.na(x)))   stop("x contains NA values.")
  if (any(is.na(w)))   stop("w contains NA values.")
  if (any(is.na(y)))   stop("y contains NA values.")
  
  list(x = x, w = w, y = y)
}

#' Sanitize a new covariate matrix for prediction
#'
#' @param newx feature matrix for prediction
#' @return numeric matrix
sanitize_x <- function(newx) {
  if (is.data.frame(newx)) newx <- model.matrix(~. - 1, newx)
  if (!is.matrix(newx))    newx <- as.matrix(newx)
  if (!is.numeric(newx))   newx <- apply(newx, 2, as.numeric)
  newx
}