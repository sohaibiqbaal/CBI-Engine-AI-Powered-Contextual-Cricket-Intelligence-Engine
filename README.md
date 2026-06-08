# Contextual Batting Intelligence (CBI) Framework

**Author:** Muhammad Sohaib Iqbal
**Dashboard:** [CBI Interactive Dashboard](https://cbidashboard.netlify.app/) 

## Overview

Traditional batting metrics treat all runs identically, regardless of the match state in which they were scored. A batter accumulating runs in a comfortable powerplay and one manufacturing runs under death-over pressure receive equivalent credit, despite categorically different decision environments

The **Contextual Batting Intelligence (CBI) framework** reconceptualises batting as a sequential state-dependent optimisation problem.Drawing on Markov Decision Process (MDP) formalisation and Boltzmann rationality theory, it models batting behaviour as a risk-sensitive sequential decision problem, evaluating policy alignment relative to match context.

## Mathematical Foundation

### 1. State-Dependent Risk Weight
The cost of losing a wicket scales with resource scarcity
**Innings 1:** `λ(s) = θ1 × (W_t + 1) / (B_t + 1)`
* **Innings 2:** `λ(s) = θ2 × (W_t + 1) / (B_t + 1) × exp(clip[(RRR_t - CRR_t)/(CRR_t + 1), -2, 2])`

### 2. Contextual Utility
Opponent quality is treated as an exogenous contextual modifier
U(a|s) = Ω(b,t) × E[runs|s,a] - λ(s) × P(out|s,a)
Ω(b,t) = 1.0 + α × [(rank_bowler + rank_team) / 2000]

### 3. Boltzmann Rationality Policy Alignment
π(a|s) = exp(β × U(a|s)) / Σ exp(β × U(a'|s))

### 4. Player-Level CBI Index
A player's CBI Index is the arithmetic mean of per-delivery CBI probabilities across all legal deliveries

## Repository Architecture

**`cbi_advanced_suite.py`**: Multi-tournament pipeline, CBIEngine, and ExcelReportGenerator
**`cbi_ranker.py`**: Single-tournament CBI pipeline
* **`cbi_bootstrap_ci_module.py`**: Stratified BCa bootstrap validator.
* **`cbi_vs_icc_comparator.py`**: ICC rating simulation and paradigm comparison.
* **`cbi_validation_extensions.py`**: Revised validation tests (post-hoc)
* **`cbi_validation.py`**: Four pre-registered statistical tests
* **`cbi_benchmark.py`**: Compare CBI to SR, Runs, Avg, WCI, and ICC ratings
## Validation & Empirical Findings

* **Robustness (Sensitivity Analysis):** The aggregate mean rank standard deviation across 282 qualified players is 1.80 positions, showing strong stabilit
**Null Hypothesis Inversion:** Shuffled data produced higher scores (Z = -69.65), establishing that CBI currently measures proximity to the dataset's empirical average policy rather than independent optimal decisions
* **Predictive Validity:** Using a cumulative training design (2016+2021+2022) to predict 2024, the framework achieves a Spearman correlation of ρ = 0.421, exceeding the predefined threshold
## License & Data
Ball-by-ball delivery data were sourced from Cricsheet.org under a Creative Commons licence
This study uses exclusively publicly available sports performance data
