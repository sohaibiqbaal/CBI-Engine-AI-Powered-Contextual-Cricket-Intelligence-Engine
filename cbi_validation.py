"""
             1. Sensitivity Analysis   — ranking stability across hyper-parameter sweep
             2. Bootstrap Confidence   — 95% CIs for every player's CBI Index
             3. Null Hypothesis Test   — shuffled-delivery control experiment
             4. Predictive Validity    — cross-tournament train → test correlation

Run from the same directory as cbi_advanced_suite.py with:
    python cbi_validation_extensions.py

Outputs:
    CBI_Validation_Suite.xlsx   — all four validation sheets
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

# ── Try to import the production pipeline ────────────────────────────────────
try:
    from cbi_advanced_suite import (
        TournamentDataPipeline,
        CBIEngine,
        CONFIG,
        HISTORICAL_BOWLER_RANKINGS,
        TEAM_RANKINGS_BY_YEAR,
    )
except ImportError:
    logging.critical(
        "cbi_advanced_suite.py not found in this directory. "
        "Place this file alongside cbi_advanced_suite.py and retry."
    )
    sys.exit(1)


# 
# HELPER: run a single CBI pass with custom hyper-parameters
# 

def _run_cbi_with_params(df: pd.DataFrame, alpha: float, theta1: float,
                          theta2: float, beta: float,
                          min_balls: int = 40) -> pd.DataFrame:
    """
    Lightweight wrapper: re-runs the CBIEngine with override parameters
    and returns a leaderboard DataFrame (batsman, CBI_Index, CBI_Rank).
    """
    # Temporarily patch CONFIG so CBIEngine picks up new values
    original = {k: CONFIG[k] for k in ("alpha", "theta1", "theta2", "beta")}
    CONFIG.update(dict(alpha=alpha, theta1=theta1, theta2=theta2, beta=beta))

    engine = CBIEngine()
    processed = engine.evaluate_policy(df.copy())
    leaderboard = engine.generate_leaderboard(processed)

    # Restore CONFIG
    CONFIG.update(original)
    return leaderboard[leaderboard["Balls"] >= min_balls].copy()



# 1. SENSITIVITY ANALYSIS
#    Sweeps β ∈ [0.5, 3.0] (10 steps) and θ₁/θ₂/α ± 20%.
#    Records standard deviation of each player's rank across all runs.

def run_sensitivity_analysis(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("=== SENSITIVITY ANALYSIS: hyper-parameter sweep ===")

    base = dict(
        alpha=CONFIG["alpha"],
        theta1=CONFIG["theta1"],
        theta2=CONFIG["theta2"],
        beta=CONFIG["beta"],
    )

    # Build sweep grid
    beta_values   = np.linspace(0.5, 3.0, 10)
    alpha_values  = [base["alpha"] * f for f in (0.8, 1.0, 1.2)]
    theta1_values = [base["theta1"] * f for f in (0.8, 1.0, 1.2)]
    theta2_values = [base["theta2"] * f for f in (0.8, 1.0, 1.2)]

    all_rank_frames = []
    run_labels = []

    # --- β sweep (primary parameter) ---
    for bv in beta_values:
        lb = _run_cbi_with_params(df, base["alpha"], base["theta1"],
                                  base["theta2"], bv)
        lb = lb.set_index("batsman")["CBI_Rank"].rename(f"beta={bv:.2f}")
        all_rank_frames.append(lb)
        run_labels.append(f"β={bv:.2f}")

    # --- α perturbation ---
    for av in [alpha_values[0], alpha_values[2]]:
        lb = _run_cbi_with_params(df, av, base["theta1"], base["theta2"],
                                  base["beta"])
        lb = lb.set_index("batsman")["CBI_Rank"].rename(f"alpha={av:.3f}")
        all_rank_frames.append(lb)

    # --- θ₁ perturbation ---
    for tv in [theta1_values[0], theta1_values[2]]:
        lb = _run_cbi_with_params(df, base["alpha"], tv, base["theta2"],
                                  base["beta"])
        lb = lb.set_index("batsman")["CBI_Rank"].rename(f"theta1={tv:.2f}")
        all_rank_frames.append(lb)

    # --- θ₂ perturbation ---
    for tv in [theta2_values[0], theta2_values[2]]:
        lb = _run_cbi_with_params(df, base["alpha"], base["theta1"], tv,
                                  base["beta"])
        lb = lb.set_index("batsman")["CBI_Rank"].rename(f"theta2={tv:.2f}")
        all_rank_frames.append(lb)

    rank_matrix = pd.concat(all_rank_frames, axis=1).dropna()

    summary = pd.DataFrame(index=rank_matrix.index)
    summary["Rank_Mean"]  = rank_matrix.mean(axis=1)
    summary["Rank_StdDev"]= rank_matrix.std(axis=1)
    summary["Rank_Min"]   = rank_matrix.min(axis=1)
    summary["Rank_Max"]   = rank_matrix.max(axis=1)
    summary["Rank_Range"] = summary["Rank_Max"] - summary["Rank_Min"]
    summary = summary.sort_values("Rank_Mean").reset_index()
    summary.rename(columns={"index": "batsman"}, inplace=True)

    # Spearman correlation between β-sweep endpoints
    base_run = rank_matrix.iloc[:, 0]  # β=0.5
    high_run = rank_matrix.iloc[:, -1] # β=3.0
    rho, p = spearmanr(base_run, high_run)
    logging.info(
        f"Sensitivity: Spearman ρ(β=0.5, β=3.0) = {rho:.4f} — "
        + ("STABLE" if rho > 0.85 else "SENSITIVE — review β calibration")
    )

    # Append meta row
    meta = pd.DataFrame([{
        "batsman": "[AGGREGATE STABILITY]",
        "Rank_Mean": summary["Rank_Mean"].mean(),
        "Rank_StdDev": summary["Rank_StdDev"].mean(),
        "Rank_Min": np.nan,
        "Rank_Max": np.nan,
        "Rank_Range": summary["Rank_Range"].mean(),
    }])
    summary = pd.concat([summary, meta], ignore_index=True)

    # Also keep the raw matrix for the Excel output
    rank_matrix_out = rank_matrix.reset_index().rename(columns={"batsman": "batsman"})
    return summary, rank_matrix_out


# 2. BOOTSTRAP CONFIDENCE INTERVALS
#    For each qualified player, resample their delivery pool 500 times
#    and compute 2.5th / 97.5th percentile of CBI_Index.

def run_bootstrap_confidence_intervals(processed_df: pd.DataFrame,
                                       n_bootstrap: int = 500,
                                       min_balls: int = 40) -> pd.DataFrame:
    logging.info("=== BOOTSTRAP CONFIDENCE INTERVALS (n=500 per player) ===")

    engine = CBIEngine()
    results = []

    player_groups = {
        name: grp for name, grp in processed_df.groupby("batsman")
        if len(grp) >= min_balls
    }

    for player, pdata in player_groups.items():
        boot_indices = np.array([
            pdata["cbi_probability"].sample(
                n=len(pdata), replace=True, random_state=seed
            ).mean()
            for seed in range(n_bootstrap)
        ])
        results.append({
            "batsman":    player,
            "CBI_Index":  pdata["cbi_probability"].mean(),
            "CI_Lower":   np.percentile(boot_indices, 2.5),
            "CI_Upper":   np.percentile(boot_indices, 97.5),
            "CI_Width":   np.percentile(boot_indices, 97.5) - np.percentile(boot_indices, 2.5),
            "Balls":      len(pdata),
        })

    ci_df = pd.DataFrame(results).sort_values("CBI_Index", ascending=False)
    ci_df["CBI_Rank"] = range(1, len(ci_df) + 1)

    # Flag players whose CI overlaps with adjacent rank
    ci_df["Overlaps_Next_Rank"] = False
    for i in range(len(ci_df) - 1):
        if ci_df.iloc[i]["CI_Lower"] < ci_df.iloc[i + 1]["CI_Upper"]:
            ci_df.at[ci_df.index[i], "Overlaps_Next_Rank"] = True

    logging.info(
        f"Bootstrap complete. Mean CI width = {ci_df['CI_Width'].mean():.4f}. "
        f"Players with statistically distinct top rank: "
        f"{(~ci_df['Overlaps_Next_Rank']).sum()}"
    )
    return ci_df.reset_index(drop=True)


# 3. NULL HYPOTHESIS (CONTROL EXPERIMENT)
#    Shuffle delivery outcomes within each player's pool and recompute CBI.
#    A valid model should produce significantly lower scores on shuffled data.

def run_null_hypothesis_test(processed_df: pd.DataFrame,
                              n_shuffles: int = 30,
                              min_balls: int = 40) -> pd.DataFrame:
    logging.info("=== NULL HYPOTHESIS TEST: shuffled delivery control ===")

    engine_real = CBIEngine()
    real_lb = engine_real.generate_leaderboard(processed_df)
    real_lb = real_lb[real_lb["Balls"] >= min_balls].copy()
    real_mean_cbi = real_lb["CBI_Index"].mean()

    shuffle_means = []
    for seed in range(n_shuffles):
        rng = np.random.default_rng(seed)
        shuffled = processed_df.copy()
        # Shuffle runs_batter and is_out columns independently within each match
        for match_id, mgrp in shuffled.groupby("match_id"):
            idx = mgrp.index
            shuffled.loc[idx, "runs_batter"] = rng.permutation(
                mgrp["runs_batter"].values
            )
            shuffled.loc[idx, "is_out"] = rng.permutation(
                mgrp["is_out"].values
            )

        engine_sh = CBIEngine()
        sh_processed = engine_sh.evaluate_policy(shuffled)
        sh_lb = engine_sh.generate_leaderboard(sh_processed)
        sh_lb = sh_lb[sh_lb["Balls"] >= min_balls]
        shuffle_means.append(sh_lb["CBI_Index"].mean())

    shuffle_mean   = np.mean(shuffle_means)
    shuffle_std    = np.std(shuffle_means)
    z_score        = (real_mean_cbi - shuffle_mean) / (shuffle_std + 1e-10)

    logging.info(
        f"Real mean CBI: {real_mean_cbi:.4f} | "
        f"Shuffled mean: {shuffle_mean:.4f} ± {shuffle_std:.4f} | "
        f"Z-score: {z_score:.2f} — "
        + ("MODEL IS VALID (signal ≠ noise)" if z_score > 2.0
           else "WARNING: low separation from null distribution")
    )

    result_df = pd.DataFrame({
        "Metric": [
            "Real Data — Mean CBI Index",
            "Shuffled Data — Mean CBI Index",
            "Shuffled Data — Std Dev",
            "Z-Score (Real vs Null)",
            "Interpretation",
            "Number of Shuffle Iterations",
        ],
        "Value": [
            round(real_mean_cbi, 4),
            round(shuffle_mean, 4),
            round(shuffle_std, 4),
            round(z_score, 2),
            "Model captures genuine signal (Z > 2)" if z_score > 2.0
            else "Insufficient separation — revisit feature design",
            n_shuffles,
        ]
    })

    per_shuffle = pd.DataFrame({
        "Shuffle_Iteration": range(n_shuffles),
        "Shuffled_Mean_CBI": shuffle_means
    })
    return result_df, per_shuffle


# 4. PREDICTIVE VALIDITY
#    Train on tournament year N, evaluate Spearman ρ against year N+1.

def run_predictive_validity(df_all: pd.DataFrame,
                             min_balls: int = 40) -> pd.DataFrame:
    logging.info("=== PREDICTIVE VALIDITY: cross-tournament train → test ===")

    years = sorted(df_all["tournament_year"].unique())
    records = []

    for i in range(len(years) - 1):
        yr_train = years[i]
        yr_test  = years[i + 1]

        train_df = df_all[df_all["tournament_year"] == yr_train]
        test_df  = df_all[df_all["tournament_year"] == yr_test]

        engine_tr = CBIEngine()
        train_processed = engine_tr.evaluate_policy(train_df.copy())
        train_lb = engine_tr.generate_leaderboard(train_processed)
        train_lb = train_lb[train_lb["Balls"] >= min_balls][["batsman", "CBI_Index"]].rename(
            columns={"CBI_Index": "CBI_Train"}
        )

        engine_te = CBIEngine()
        test_processed = engine_te.evaluate_policy(test_df.copy())
        test_lb = engine_te.generate_leaderboard(test_processed)
        test_lb = test_lb[test_lb["Balls"] >= min_balls][["batsman", "CBI_Index"]].rename(
            columns={"CBI_Index": "CBI_Test"}
        )

        merged = pd.merge(train_lb, test_lb, on="batsman")

        if len(merged) < 5:
            logging.warning(
                f"Only {len(merged)} overlapping players between {yr_train}→{yr_test}. "
                "Skipping pair."
            )
            continue

        rho, p_val = spearmanr(merged["CBI_Train"], merged["CBI_Test"])
        r_pearson, _ = pearsonr(merged["CBI_Train"], merged["CBI_Test"])

        records.append({
            "Train_Year":       yr_train,
            "Test_Year":        yr_test,
            "Overlapping_Players": len(merged),
            "Spearman_rho":     round(rho, 4),
            "Pearson_r":        round(r_pearson, 4),
            "P_Value":          round(p_val, 4),
            "Interpretation":   (
                "Strong predictive signal" if rho > 0.5
                else ("Moderate predictive signal" if rho > 0.3
                      else "Weak — expand sample or revisit features")
            ),
        })

        logging.info(
            f"  {yr_train} → {yr_test}: ρ = {rho:.4f}, "
            f"n = {len(merged)} players"
        )

    if not records:
        logging.warning("No valid train/test pairs found. Need multi-year data.")
        return pd.DataFrame(columns=["Train_Year", "Test_Year",
                                     "Overlapping_Players", "Spearman_rho",
                                     "Pearson_r", "P_Value", "Interpretation"])

    return pd.DataFrame(records)



# MASTER EXECUTION: compile all 4 tests into one Excel workbook

def run_all_validations(data_dir: str = "t20_json_data",
                         output_file: str = "CBI_Validation_Suite.xlsx"):
    logging.info("Initialising CBI Validation Suite…")

    pipeline = TournamentDataPipeline(data_dir)
    raw_data = pipeline.ingest_all_tournaments()

    if raw_data.empty:
        logging.critical(
            f"No CSV files found in '{data_dir}'. "
            "Place your Cricsheet tournament CSVs there and retry."
        )
        return

    # Pre-compute base processed dataset (used across tests)
    engine_base = CBIEngine()
    processed_base = engine_base.evaluate_policy(raw_data.copy())

    logging.info("Running Test 1: Sensitivity Analysis…")
    sens_summary, sens_matrix = run_sensitivity_analysis(raw_data)

    logging.info("Running Test 2: Bootstrap Confidence Intervals…")
    bootstrap_df = run_bootstrap_confidence_intervals(processed_base)

    logging.info("Running Test 3: Null Hypothesis Test…")
    null_summary, null_per_run = run_null_hypothesis_test(processed_base)

    logging.info("Running Test 4: Predictive Validity…")
    predictive_df = run_predictive_validity(raw_data)

    # ── Write to Excel ────────────────────────────────────────────────────────
    logging.info(f"Writing results to '{output_file}'…")
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:

        # Sheet 1 — Sensitivity summary
        sens_summary.to_excel(writer, sheet_name="1_Sensitivity_Summary", index=False)
        sens_matrix.to_excel(writer, sheet_name="1_Sensitivity_RawMatrix", index=False)

        # Sheet 2 — Bootstrap CIs
        bootstrap_df.to_excel(writer, sheet_name="2_Bootstrap_CIs", index=False)

        # Sheet 3 — Null hypothesis
        null_summary.to_excel(writer, sheet_name="3_Null_Hypothesis", index=False)
        null_per_run.to_excel(writer, sheet_name="3_Null_PerIteration", index=False)

        # Sheet 4 — Predictive validity
        if not predictive_df.empty:
            predictive_df.to_excel(writer, sheet_name="4_Predictive_Validity", index=False)
        else:
            pd.DataFrame({"Note": ["Multi-year data required for this test."]}).to_excel(
                writer, sheet_name="4_Predictive_Validity", index=False
            )

        # Sheet 5 — Validation summary card
        spearman_sens = (
            "Run sensitivity to get value"
            if sens_summary.empty
            else f"Mean rank StdDev = {sens_summary['Rank_StdDev'].iloc[:-1].mean():.2f}"
        )
        summary_card = pd.DataFrame({
            "Validation Test": [
                "1. Sensitivity Analysis",
                "2. Bootstrap 95% CI",
                "3. Null Hypothesis (Shuffle)",
                "4. Predictive Validity",
            ],
            "Key Metric": [
                "Rank StdDev across 14 parameter configurations",
                "Per-player 95% CI width for CBI Index",
                "Z-Score: Real CBI vs Shuffled CBI",
                "Spearman ρ: Train-year CBI → Test-year performance",
            ],
            "Threshold for Acceptance": [
                "StdDev < 5 rank positions → model stable",
                "CI width < 0.05 for top-10 players → reliable",
                "Z > 2.0 → model captures real signal, not noise",
                "ρ > 0.40 → CBI is predictive, not descriptive",
            ],
            "Result Location": [
                "Sheet: 1_Sensitivity_Summary",
                "Sheet: 2_Bootstrap_CIs",
                "Sheet: 3_Null_Hypothesis",
                "Sheet: 4_Predictive_Validity",
            ],
        })
        summary_card.to_excel(writer, sheet_name="0_Validation_Overview", index=False)

    logging.info(f"✓ All validation tests complete → '{output_file}'")
    return output_file


if __name__ == "__main__":
    DATA_DIR   = "t20_json_data"
    OUT_FILE   = "CBI_Validation_Suite.xlsx"

    run_all_validations(data_dir=DATA_DIR, output_file=OUT_FILE)
