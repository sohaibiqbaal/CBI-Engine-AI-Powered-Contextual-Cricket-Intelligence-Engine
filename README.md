# Contextual Batting Intelligence (CBI) Framework

[cite_start]**Author:** Muhammad Sohaib Iqbal [cite: 3]
[cite_start]**Dashboard:** [CBI Interactive Dashboard](https://cbidashboard.netlify.app/) [cite: 104]

## Overview

[cite_start]Traditional batting metrics treat all runs identically, regardless of the match state in which they were scored[cite: 8]. [cite_start]A batter accumulating runs in a comfortable powerplay and one manufacturing runs under death-over pressure receive equivalent credit, despite categorically different decision environments[cite: 9]. 

[cite_start]The **Contextual Batting Intelligence (CBI) framework** reconceptualises batting as a sequential state-dependent optimisation problem[cite: 10]. [cite_start]Drawing on Markov Decision Process (MDP) formalisation and Boltzmann rationality theory, it models batting behaviour as a risk-sensitive sequential decision problem, evaluating policy alignment relative to match context[cite: 11].

## Mathematical Foundation

### 1. State-Dependent Risk Weight
[cite_start]The cost of losing a wicket scales with resource scarcity[cite: 74].
* [cite_start]**Innings 1:** `λ(s) = θ1 × (W_t + 1) / (B_t + 1)` [cite: 77]
* [cite_start]**Innings 2:** `λ(s) = θ2 × (W_t + 1) / (B_t + 1) × exp(clip[(RRR_t - CRR_t)/(CRR_t + 1), -2, 2])` [cite: 79]

### 2. Contextual Utility
[cite_start]Opponent quality is treated as an exogenous contextual modifier[cite: 69].
* [cite_start]`U(a|s) = Ω(b,t) × E[runs|s,a] - λ(s) × P(out|s,a)` [cite: 84]
* [cite_start]`Ω(b,t) = 1.0 + α × [(rank_bowler + rank_team) / 2000]` [cite: 85]

### 3. Boltzmann Rationality Policy Alignment
* [cite_start]`π(a|s) = exp(β × U(a|s)) / Σ exp(β × U(a'|s))` [cite: 88]

### 4. Player-Level CBI Index
[cite_start]A player's CBI Index is the arithmetic mean of per-delivery CBI probabilities across all legal deliveries[cite: 94].

## Repository Architecture

* [cite_start]**`cbi_advanced_suite.py`**: Multi-tournament pipeline, CBIEngine, and ExcelReportGenerator[cite: 245].
* [cite_start]**`cbi_ranker.py`**: Single-tournament CBI pipeline[cite: 245].
* **`cbi_bootstrap_ci_module.py`**: Stratified BCa bootstrap validator[cite: 245].
* **`cbi_vs_icc_comparator.py`**: ICC rating simulation and paradigm comparison[cite: 245].
* **`cbi_validation_extensions.py`**: Revised validation tests (post-hoc)[cite: 245].
* [cite_start]**`cbi_validation.py`**: Four pre-registered statistical tests[cite: 245].
* **`cbi_benchmark.py`**: Compare CBI to SR, Runs, Avg, WCI, and ICC ratings[cite: 245].

## Validation & Empirical Findings

* **Robustness (Sensitivity Analysis):** The aggregate mean rank standard deviation across 282 qualified players is 1.80 positions, showing strong stability[cite: 119].
* [cite_start]**Null Hypothesis Inversion:** Shuffled data produced higher scores (Z = -69.65), establishing that CBI currently measures proximity to the dataset's empirical average policy rather than independent optimal decisions[cite: 130, 131, 133].
* [cite_start]**Predictive Validity:** Using a cumulative training design (2016+2021+2022) to predict 2024, the framework achieves a Spearman correlation of ρ = 0.421, exceeding the predefined threshold[cite: 147, 151, 156].

## License & Data
[cite_start]Ball-by-ball delivery data were sourced from Cricsheet.org under a Creative Commons licence[cite: 231]. [cite_start]This study uses exclusively publicly available sports performance data[cite: 235].
