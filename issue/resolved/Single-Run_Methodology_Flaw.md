# Issue: Single-Run Evaluation Methodology Flaw

## Context
Currently, the benchmark generates a dataset using a single configuration (e.g., `experiment_26_06.py`) and evaluates the estimators on that specific dataset realization. For example, it generates 1500 trajectories for the pendulum or Lorenz system using a single base random seed. 

## The Core Problem
While evaluating across 1500 trajectories is good for statistical significance *within* a single dataset, the dataset itself is fundamentally stochastic. The process noise, observation noise, and initial states are all governed by random variables. Furthermore, systems like `LorenzBenchmark` are chaotic, meaning they possess a positive Lyapunov exponent where minute differences compound exponentially.

Because of this, a single dataset realization—no matter how many trajectories it contains—is ultimately a high-variance point estimate. The RMSE score an estimator achieves might be skewed by a "lucky" or "unlucky" global sequence of noise. 

In the state-estimation literature, comparing two estimators based on one stochastic run is considered methodologically flawed because performance deltas might simply fall within the margin of noise.

## Required Solution: Monte-Carlo Evaluation
To scientifically prove that one estimator outperforms another, we must evaluate them across multiple independent dataset realizations.

1. **Seed Looping:** Wrap the entire experiment runner in an outer loop that iterates over an array of base random seeds (e.g., $N \ge 10$).
2. **Full Pipeline Execution:** For *each* seed:
   - Generate a completely new dataset (train/val/test).
   - Initialize and (if applicable) fit the estimators.
   - Run predictions and calculate metrics.
3. **Aggregate Reporting:** Instead of reporting a single scalar RMSE, aggregate the results across all $N$ runs and report the **Mean $\pm$ Standard Deviation** (or 95% confidence intervals) for every metric. 

## Acceptance Criteria
- [ ] Experiment runner accepts a list of seeds rather than a single `RANDOM_SEED`.
- [ ] SQLite storage / tracking is updated to log the run ID or seed.
- [ ] Visualizations and summary tables are updated to render error bars or $\pm$ standard deviation text.