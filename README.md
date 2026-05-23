# Contextual Batting Intelligence (CBI) Engine

The **Contextual Batting Intelligence (CBI) Engine** is a production-ready, decision-theoretic framework designed for advanced cricket analytics. It leverages principles from **Maximum Entropy (MaxEnt) Inverse Reinforcement Learning** to evaluate batting performance, accounting for situational game pressure, opponent strength, and historical performance context.

## Overview

Traditional cricket metrics (like batting average and strike rate) often overlook the context of a run. A run scored in a high-pressure death-over scenario against a top-tier bowler is objectively more valuable than a run scored in a low-pressure environment against a weaker opponent. The CBI Engine bridges this gap by calculating the probability that an optimal, professional agent would make a given batting choice (dot, rotate, or attack) under specific match conditions.



## Key Features

- **Dynamic State Reconstruction:** Ingests raw ball-by-ball data and reconstructs the game state (overs remaining, wickets lost, required run rate, etc.) for every delivery.
- **Context-Aware Utility Mapping:** Assigns utility to batting actions based on environmental factors, including bowler quality and team strength rankings.
- **Maximum Entropy Modeling:** Utilizes softmax policy alignment to measure how well a batsman’s observed actions align with optimal "intelligence" under pressure.
- **Paradigm Comparison:** Includes a benchmarking tool to compare CBI-based rankings against traditional ICC-style ratings.
- **Automated Validation:** Features a comprehensive suite for stability, predictive power, and sensitivity analysis.

## Project Structure

- `cbi_advanced_suite.py`: The central engine containing the data pipeline, CBI logic, and validation suite.
- `cbi_ranker.py`: A streamlined version focused on producing comparative leaderboards (CBI vs. Traditional Rankings) in Excel.
- `cbi_vs_icc_comparator.py`: An auditing tool that simulates official ICC T20 ratings and computes correlation metrics against CBI rankings.

## Getting Started

### Prerequisites

Ensure you have Python installed and the necessary libraries:

```bash
pip install pandas numpy scipy openpyxl
