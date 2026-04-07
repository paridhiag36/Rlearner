# setwd("~/DSE4231/Rlearner") ALWAYS

source("utils.R")
source("code/KRR/cvkern.R")


# Test 1: Kernel Matrix Properties
set.seed(42)
x <- matrix(rnorm(20), 10, 2)
K <- .gauss_kernel_matrix(x, x, sigma = 1)

all(K >= 0 & K <= 1)       # TRUE
all(abs(diag(K) - 1) < 1e-10) # correct way to check since values are 0.9999999 type
all(abs(K - t(K)) < 1e-10) # TRUE — symmetric


# Test 2: Weighted KRR recovers a known function
set.seed(1)
n <- 100
x <- matrix(runif(n * 2), n, 2)
y_true <- x[,1] + x[,2]        # simple linear truth
y      <- y_true + rnorm(n, sd = 0.1)
w      <- rep(1, n)             # uniform weights

m    <- .weighted_krr(x, y, sigma = 1, lambda = 0.01, weights = w)
pred <- predict(m, x)

cor(pred, y_true)               # should be very close to 1, we get 0.9988093
mean((pred - y_true)^2)         # should be small, ~0.01 or less (0.0003663138)


# Test 3: cv_klrs() picks a finite, sensible parameter
set.seed(1)
n <- 150; p <- 4
x <- matrix(rnorm(n * p), n, p)
y <- x[,1]^2 + x[,2] + rnorm(n, sd = 0.5)

fit <- cv_klrs(x, y, k_folds = 3,
               b_range      = 10^(-1:1),   # coarse grid, fast
               lambda_range = 10^(-1:1))

# Check outputs exist and are sensible
fit$b                          # should be a finite positive number, we got 10
fit$lambda                     # same, we got 0.1
length(fit$fit) == n           # TRUE — one CV prediction per obs
cor(fit$fit, y) > 0.5          # TRUE — should fit better than noise


# Test 4: Sanity check on cv_klrs()
# Very smooth function → large b should win over tiny b
set.seed(42)
x <- matrix(runif(200), 200, 1)
y <- sin(2 * pi * x[,1]) + rnorm(200, sd = 0.05)

fit <- cv_klrs(x, y, k_folds = 5,
               b_range      = 10^(seq(-2, 2, 1)),
               lambda_range = 10^(seq(-2, 2, 1)))

cat("Best b:", fit$b, "\n")       # expect moderate value, not 0.01 (we got Best b: 0.1)
cat("Best lambda:", fit$lambda, "\n") # we got best lambda = 1
plot(x[,1], y)
points(x[,1], fit$fit, col = "red", pch = 20)  # should trace the sine curve

