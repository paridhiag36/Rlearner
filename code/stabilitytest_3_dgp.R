source("code/utils.R")
source("code/3_dgp.R")
source("code/learners_lasso.R")

set.seed(42)
reps <- 10
results_store <- matrix(NA, nrow = reps, ncol = 4)
colnames(results_store) <- c("S", "T", "X", "R")

for (i in 1:reps) {
  dat  <- gen_setup_3(n = 1000)
  n    <- nrow(dat$X)
  tr   <- 1:800; te <- 801:1000
  
  results_store[i, "S"] <- mean((s_learner_lasso(dat$X[tr,], dat$Y[tr], dat$W[tr], dat$X[te,]) - dat$tau[te])^2)
  results_store[i, "T"] <- mean((t_learner_lasso(dat$X[tr,], dat$Y[tr], dat$W[tr], dat$X[te,]) - dat$tau[te])^2)
  results_store[i, "X"] <- mean((x_learner_lasso(dat$X[tr,], dat$Y[tr], dat$W[tr], dat$X[te,]) - dat$tau[te])^2)
  results_store[i, "R"] <- mean((r_learner_lasso(dat$X[tr,], dat$Y[tr], dat$W[tr], dat$X[te,]) - dat$tau[te])^2)
  
  cat("Rep", i, "done\n")
}

cat("\n=== Setup 3: Average MSE over 10 reps ===\n")
cat(sprintf("S: %.4f | T: %.4f | X: %.4f | R: %.4f\n",
            mean(results_store[,"S"]), mean(results_store[,"T"]),
            mean(results_store[,"X"]), mean(results_store[,"R"])))
cat(sprintf("Winner: %s-learner\n",
            colnames(results_store)[which.min(colMeans(results_store))]))
