# DSE4231 Group Project 
# Section 6.3 replication: Nie & Wager (2021)

## Project Structure

```
RLearner/
├── utils.R          # Input sanitisation helpers (replaces rlearner utils.R)
├── data/   
|   ├── data_clean.csv
├── code/
│   ├── cvkern.R         # KRR cross-validation engine (replaces KRLS2)
│   ├── skern.R          # S-learner via KRR
│   ├── tkern.R          # T-learner via KRR
│   ├── xkern.R          # X-learner via KRR
│   ├── ukern.R          # U-learner via KRR
│   ├── rkern.R          # R-learner via KRR
│   └── sim_setups.R     # DGPs A, B, C, D (Section 6.1)
│
```

## Installation

Install required R packages (run once):
```r
install.packages(c("kernlab", "ggplot2"))
```
## Kernel Ridge Regression Experiments
# Assigned to Paridhi Agarwal
> **Why `kernlab` instead of `KRLS2`?**
> The original rlearner package used `KRLS2`, which is no longer on CRAN and
> no longer maintained. `kernlab` is the modern, actively maintained alternative.
> The Gaussian kernel parameterisation is equivalent; we convert the bandwidth `b`
> to kernlab's `sigma = 1/b` internally.