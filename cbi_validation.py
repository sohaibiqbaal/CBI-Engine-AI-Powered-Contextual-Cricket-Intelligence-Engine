"""
CBI Validation Extensions v2.0 — Fixed Statistical Tests
Author: Muhammad Sohaib Iqbal
Fixes:
  - Null Hypothesis: lookup tables now recomputed FROM shuffled data each iteration
    (previously used real-data tables → structural inversion / negative Z-score)
  - Predictive Validity: cumulative training (2016+2021+2022 → 2024) rather than
    pairwise year-on-year, giving larger overlap and cleaner single holdout test
  - All four tests write full results to CBI_Validation_Suite_v2.xlsx

Run:  python cbi_validation_extensions_v2.py
Requires: cbi_advanced_suite.py in same directory, t20_json_data/ folder
"""

import os, sys, logging, warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge
import joblib

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

try:
    from cbi_advanced_suite import (
        TournamentDataPipeline, CBIEngine, CONFIG,
        HISTORICAL_BOWLER_RANKINGS, TEAM_RANKINGS_BY_YEAR,
    )
except ImportError:
    logging.critical("cbi_advanced_suite.py not found. Place this file alongside it.")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: run CBI with custom hyperparameters
# ─────────────────────────────────────────────────────────────────────────────
def _run_cbi_with_params(df, alpha, theta1, theta2, beta, min_balls=40):
    orig = {k: CONFIG[k] for k in ("alpha", "theta1", "theta2", "beta")}
    CONFIG.update(dict(alpha=alpha, theta1=theta1, theta2=theta2, beta=beta))
    engine = CBIEngine()
    proc = engine.evaluate_policy(df.copy())
    lb = engine.generate_leaderboard(proc)
    CONFIG.update(orig)
    return lb[lb["Balls"] >= min_balls].copy()


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: build leaderboard from a DataFrame
# ─────────────────────────────────────────────────────────────────────────────
def _make_leaderboard(df, min_balls=40):
    engine = CBIEngine()
    proc = engine.evaluate_policy(df.copy())
    lb = engine.generate_leaderboard(proc)
    return lb[lb["Balls"] >= min_balls].copy()


# ─────────────────────────────────────────────────────────────────────────────
# TEST 1 — SENSITIVITY ANALYSIS (unchanged, already passes)
# ─────────────────────────────────────────────────────────────────────────────
def run_sensitivity_analysis(df):
    logging.info("=== TEST 1: SENSITIVITY ANALYSIS ===")
    base = {k: CONFIG[k] for k in ("alpha", "theta1", "theta2", "beta")}
    beta_values   = np.linspace(0.5, 3.0, 10)
    alpha_vals    = [base["alpha"] * f for f in (0.8, 1.2)]
    theta1_vals   = [base["theta1"] * f for f in (0.8, 1.2)]
    theta2_vals   = [base["theta2"] * f for f in (0.8, 1.2)]

    frames, labels = [], []
    for bv in beta_values:
        lb = _run_cbi_with_params(df, base["alpha"], base["theta1"], base["theta2"], bv)
        frames.append(lb.set_index("batsman")["CBI_Rank"].rename(f"beta={bv:.2f}"))
        labels.append(f"beta={bv:.2f}")
    for av in alpha_vals:
        lb = _run_cbi_with_params(df, av, base["theta1"], base["theta2"], base["beta"])
        frames.append(lb.set_index("batsman")["CBI_Rank"].rename(f"alpha={av:.3f}"))
    for tv in theta1_vals:
        lb = _run_cbi_with_params(df, base["alpha"], tv, base["theta2"], base["beta"])
        frames.append(lb.set_index("batsman")["CBI_Rank"].rename(f"theta1={tv:.2f}"))
    for tv in theta2_vals:
        lb = _run_cbi_with_params(df, base["alpha"], base["theta1"], tv, base["beta"])
        frames.append(lb.set_index("batsman")["CBI_Rank"].rename(f"theta2={tv:.2f}"))

    rank_matrix = pd.concat(frames, axis=1).dropna()
    summary = pd.DataFrame({
        "batsman":    rank_matrix.index,
        "Rank_Mean":  rank_matrix.mean(axis=1).values,
        "Rank_StdDev": rank_matrix.std(axis=1).values,
        "Rank_Min":   rank_matrix.min(axis=1).values,
        "Rank_Max":   rank_matrix.max(axis=1).values,
        "Rank_Range": (rank_matrix.max(axis=1) - rank_matrix.min(axis=1)).values,
    }).sort_values("Rank_Mean").reset_index(drop=True)

    rho, _ = spearmanr(rank_matrix.iloc[:, 0], rank_matrix.iloc[:, -1])
    logging.info(f"  Spearman ρ(β=0.5, β=3.0) = {rho:.4f} | Mean StdDev = {summary['Rank_StdDev'].mean():.2f}")

    meta = pd.DataFrame([{
        "batsman": "[AGGREGATE STABILITY]",
        "Rank_Mean": summary["Rank_Mean"].mean(),
        "Rank_StdDev": summary["Rank_StdDev"].mean(),
        "Rank_Min": np.nan, "Rank_Max": np.nan,
        "Rank_Range": summary["Rank_Range"].mean(),
    }])
    summary = pd.concat([summary, meta], ignore_index=True)
    return summary, rank_matrix.reset_index()


# ─────────────────────────────────────────────────────────────────────────────
# TEST 2 — BOOTSTRAP CONFIDENCE INTERVALS (unchanged)
# ─────────────────────────────────────────────────────────────────────────────
def run_bootstrap_confidence_intervals(processed_df, n_bootstrap=500, min_balls=40):
    logging.info("=== TEST 2: BOOTSTRAP CONFIDENCE INTERVALS ===")
    results = []
    for player, pdata in processed_df.groupby("batsman"):
        if len(pdata) < min_balls:
            continue
        probs = pdata["cbi_probability"].values
        boot = [np.random.choice(probs, size=len(probs), replace=True).mean()
                for _ in range(n_bootstrap)]
        results.append({
            "batsman": player,
            "CBI_Index": probs.mean(),
            "CI_Lower": np.percentile(boot, 2.5),
            "CI_Upper": np.percentile(boot, 97.5),
            "CI_Width": np.percentile(boot, 97.5) - np.percentile(boot, 2.5),
            "Balls": len(pdata),
        })
    ci_df = pd.DataFrame(results).sort_values("CBI_Index", ascending=False).reset_index(drop=True)
    ci_df["CBI_Rank"] = range(1, len(ci_df) + 1)
    ci_df["Overlaps_Next_Rank"] = [
        ci_df.iloc[i]["CI_Lower"] < ci_df.iloc[i+1]["CI_Upper"]
        if i < len(ci_df)-1 else False
        for i in range(len(ci_df))
    ]
    logging.info(f"  Mean CI width = {ci_df['CI_Width'].mean():.4f} | "
                 f"Non-overlapping pairs: {(~ci_df['Overlaps_Next_Rank']).sum()}")
    return ci_df


# ─────────────────────────────────────────────────────────────────────────────
# TEST 3 — NULL HYPOTHESIS (FIXED)
# Key fix: for each shuffle, rebuild lookup tables FROM the shuffled data,
# not from real data. This is the correct implementation of the shuffle test.
# A valid model: real CBI > shuffled CBI (positive Z-score).
# ─────────────────────────────────────────────────────────────────────────────
def run_null_hypothesis_test_fixed(processed_df, raw_df, n_shuffles=30, min_balls=40):
    logging.info("=== TEST 3: NULL HYPOTHESIS (FIXED — lookup tables rebuilt each shuffle) ===")

    # Real data CBI (leaderboard already computed)
    engine_real = CBIEngine()
    real_lb = engine_real.generate_leaderboard(processed_df)
    real_lb = real_lb[real_lb["Balls"] >= min_balls]
    real_mean = real_lb["CBI_Index"].mean()
    logging.info(f"  Real mean CBI Index = {real_mean:.4f}")

    shuffle_means = []
    for seed in range(n_shuffles):
        rng = np.random.default_rng(seed)
        shuffled_raw = raw_df.copy()

        # Permute runs_batter and is_out within each match
        for match_id, mgrp in shuffled_raw.groupby("match_id"):
            idx = mgrp.index
            shuffled_raw.loc[idx, "runs_batter"] = rng.permutation(mgrp["runs_batter"].values)
            shuffled_raw.loc[idx, "is_out"]      = rng.permutation(mgrp["is_out"].values)

        # Recompute derived columns that depend on runs_batter / is_out
        shuffled_raw["runs_total"] = shuffled_raw["runs_batter"] + shuffled_raw.get("extras", 0)
        # action categories
        shuffled_raw["action"] = np.select(
            [shuffled_raw["runs_batter"] == 0,
             shuffled_raw["runs_batter"].isin([1, 2, 3]),
             shuffled_raw["runs_batter"] >= 4],
            [0, 1, 2], default=0
        )

        # CRITICAL FIX: evaluate_policy on shuffled data so lookup tables are
        # built from the shuffled distribution, not the real one
        engine_sh = CBIEngine()
        sh_proc = engine_sh.evaluate_policy(shuffled_raw)
        sh_lb   = engine_sh.generate_leaderboard(sh_proc)
        sh_lb   = sh_lb[sh_lb["Balls"] >= min_balls]
        shuffle_means.append(sh_lb["CBI_Index"].mean())

        if seed % 5 == 0:
            logging.info(f"  Shuffle {seed+1}/{n_shuffles}: mean={shuffle_means[-1]:.4f}")

    shuffle_mean = np.mean(shuffle_means)
    shuffle_std  = np.std(shuffle_means)
    z_score      = (real_mean - shuffle_mean) / (shuffle_std + 1e-10)

    logging.info(f"  Shuffled mean = {shuffle_mean:.4f} ± {shuffle_std:.4f}")
    logging.info(f"  Z-score = {z_score:.2f} — {'PASS (Z > 2)' if z_score > 2.0 else 'FAIL'}")

    result_df = pd.DataFrame({
        "Metric": [
            "Real Data — Mean CBI Index",
            "Shuffled Data — Mean CBI Index",
            "Shuffled Data — Std Dev",
            "Z-Score (Real vs Null)",
            "Interpretation",
            "Fix Applied",
            "Number of Shuffle Iterations",
        ],
        "Value": [
            round(real_mean, 4),
            round(shuffle_mean, 4),
            round(shuffle_std, 4),
            round(z_score, 2),
            "Model captures genuine signal (Z > 2)" if z_score > 2.0
            else "Insufficient separation — check lookup table construction",
            "Lookup tables rebuilt from shuffled data each iteration (v2 fix)",
            n_shuffles,
        ]
    })
    per_iter = pd.DataFrame({
        "Shuffle_Iteration": range(n_shuffles),
        "Shuffled_Mean_CBI": shuffle_means
    })
    return result_df, per_iter, z_score


# ─────────────────────────────────────────────────────────────────────────────
# TEST 4 — PREDICTIVE VALIDITY (FIXED: cumulative training set)
# Train on 2016+2021+2022 → predict 2024.
# Also runs year-by-year pairs for reference.
# ─────────────────────────────────────────────────────────────────────────────
def run_predictive_validity_fixed(df_all, min_balls=40):
    logging.info("=== TEST 4: PREDICTIVE VALIDITY (FIXED — cumulative training) ===")

    years = sorted(df_all["tournament_year"].unique())
    records = []

    # ── MAIN TEST: cumulative train → holdout 2024 ──────────────────────────
    holdout_year = max(years)
    train_years  = [y for y in years if y != holdout_year]

    logging.info(f"  Cumulative train years: {train_years} → holdout: {holdout_year}")

    train_df   = df_all[df_all["tournament_year"].isin(train_years)]
    test_df    = df_all[df_all["tournament_year"] == holdout_year]

    train_lb = _make_leaderboard(train_df, min_balls)[["batsman","CBI_Index"]].rename(columns={"CBI_Index":"CBI_Train"})
    test_lb  = _make_leaderboard(test_df,  min_balls)[["batsman","CBI_Index"]].rename(columns={"CBI_Index":"CBI_Test"})
    merged   = pd.merge(train_lb, test_lb, on="batsman")

    rho_cum, p_cum = spearmanr(merged["CBI_Train"], merged["CBI_Test"])
    r_cum, _       = pearsonr(merged["CBI_Train"],  merged["CBI_Test"])

    logging.info(f"  Cumulative: n={len(merged)}, Spearman ρ={rho_cum:.4f}, p={p_cum:.4f}")

    records.append({
        "Train_Set": f"{'+'.join(str(y) for y in train_years)} (cumulative)",
        "Test_Year": holdout_year,
        "Overlapping_Players": len(merged),
        "Spearman_rho": round(rho_cum, 4),
        "Pearson_r":    round(r_cum, 4),
        "P_Value":      round(p_cum, 4),
        "Interpretation": (
            "Strong predictive signal"   if rho_cum > 0.50 else
            "Moderate predictive signal" if rho_cum > 0.35 else
            "Weak predictive signal"
        ),
        "Test_Type": "Cumulative (PRIMARY)",
    })

    # ── PAIRWISE (reference) ─────────────────────────────────────────────────
    for i in range(len(years) - 1):
        yr_tr, yr_te = years[i], years[i+1]
        t_lb = _make_leaderboard(df_all[df_all["tournament_year"]==yr_tr], min_balls)[["batsman","CBI_Index"]].rename(columns={"CBI_Index":"CBI_Train"})
        e_lb = _make_leaderboard(df_all[df_all["tournament_year"]==yr_te], min_balls)[["batsman","CBI_Index"]].rename(columns={"CBI_Index":"CBI_Test"})
        m    = pd.merge(t_lb, e_lb, on="batsman")
        if len(m) < 5:
            continue
        rho, p = spearmanr(m["CBI_Train"], m["CBI_Test"])
        r, _   = pearsonr(m["CBI_Train"],  m["CBI_Test"])
        records.append({
            "Train_Set": str(yr_tr),
            "Test_Year": yr_te,
            "Overlapping_Players": len(m),
            "Spearman_rho": round(rho, 4),
            "Pearson_r":    round(r, 4),
            "P_Value":      round(p, 4),
            "Interpretation": (
                "Strong" if rho > 0.5 else
                "Moderate" if rho > 0.3 else "Weak"
            ),
            "Test_Type": "Pairwise (reference)",
        })
        logging.info(f"  Pairwise {yr_tr}→{yr_te}: n={len(m)}, ρ={rho:.4f}, p={p:.4f}")

    return pd.DataFrame(records), merged


# ─────────────────────────────────────────────────────────────────────────────
# ML PREDICTIVE MODEL — GradientBoosting trained on 2016+2021+2022 features
# ─────────────────────────────────────────────────────────────────────────────
def build_ml_predictive_model(df_all, min_balls=40):
    """
    Builds and evaluates a ML model that predicts 2024 CBI from career features.
    Features: mean stats per player from all prior tournaments.
    Returns: model, scaler, feature_cols, predictions_df
    """
    logging.info("=== ML PREDICTIVE MODEL: GradientBoosting ===")

    years = sorted(df_all["tournament_year"].unique())
    holdout_year = max(years)
    train_years  = [y for y in years if y != holdout_year]

    def extract_features(df, min_balls=40):
        """Per-player features from delivery-level data."""
        feats = df.groupby("batsman").apply(lambda g: pd.Series({
            "balls":          len(g[g["is_legal"]==1]),
            "runs":           g["runs_batter"].sum(),
            "outs":           g["is_out"].sum(),
            "boundaries":     (g["runs_batter"] >= 4).sum(),
            "dots":           (g["runs_batter"] == 0).sum(),
            "strike_rate":    g["runs_batter"].sum() / max(len(g[g["is_legal"]==1]), 1) * 100,
            "avg":            g["runs_batter"].sum() / max(g["is_out"].sum(), 1),
            "boundary_pct":   (g["runs_batter"] >= 4).sum() / max(len(g[g["is_legal"]==1]), 1),
            "dot_pct":        (g["runs_batter"] == 0).sum() / max(len(g[g["is_legal"]==1]), 1),
            "pp_sr":          _phase_sr(g, "pp"),
            "mid_sr":         _phase_sr(g, "mid"),
            "death_sr":       _phase_sr(g, "death"),
            "pp_boundary_pct": _phase_boundary(g, "pp"),
            "mean_bowler_rank": g["bowler_rank"].mean() if "bowler_rank" in g.columns else 480,
            "mean_opp_rank":  g["opp_team_rank"].mean() if "opp_team_rank" in g.columns else 130,
            "n_matches":      g["match_id"].nunique(),
            "inn2_sr":        _innings_sr(g, 2),
        })).reset_index()
        feats = feats[feats["balls"] >= min_balls]
        return feats

    def _phase_sr(g, phase):
        if "state_phase" not in g.columns:
            return 100.0
        pmap = {"pp": 0, "mid": 1, "death": 2}
        sub = g[g["state_phase"] == pmap[phase]]
        legal = sub[sub["is_legal"]==1] if "is_legal" in sub.columns else sub
        if len(legal) == 0: return 100.0
        return sub["runs_batter"].sum() / len(legal) * 100

    def _phase_boundary(g, phase):
        if "state_phase" not in g.columns:
            return 0.1
        pmap = {"pp": 0, "mid": 1, "death": 2}
        sub = g[g["state_phase"] == pmap[phase]]
        legal = sub[sub["is_legal"]==1] if "is_legal" in sub.columns else sub
        if len(legal) == 0: return 0.1
        return (sub["runs_batter"] >= 4).sum() / len(legal)

    def _innings_sr(g, inn):
        sub = g[g["innings"] == inn] if "innings" in g.columns else g
        legal = sub[sub["is_legal"]==1] if "is_legal" in sub.columns else sub
        if len(legal) == 0: return 100.0
        return sub["runs_batter"].sum() / len(legal) * 100

    # Get processed data with CBI probabilities for training years
    logging.info("  Processing training data (2016+2021+2022)...")
    train_raw = df_all[df_all["tournament_year"].isin(train_years)]
    engine_tr = CBIEngine()
    train_proc = engine_tr.evaluate_policy(train_raw.copy())
    train_lb = engine_tr.generate_leaderboard(train_proc)
    train_lb = train_lb[train_lb["Balls"] >= min_balls][["batsman","CBI_Index"]].rename(columns={"CBI_Index":"CBI_Train"})

    # Extract features from processed training data
    train_feats = extract_features(train_proc, min_balls)
    train_data = pd.merge(train_feats, train_lb, on="batsman")

    # Get 2024 CBI (target)
    logging.info("  Processing test data (2024)...")
    test_raw = df_all[df_all["tournament_year"] == holdout_year]
    engine_te = CBIEngine()
    test_proc = engine_te.evaluate_policy(test_raw.copy())
    test_lb = engine_te.generate_leaderboard(test_proc)
    test_lb = test_lb[test_lb["Balls"] >= min_balls][["batsman","CBI_Index"]].rename(columns={"CBI_Index":"CBI_2024"})
    test_feats = extract_features(test_proc, min_balls)

    # Players appearing in both
    test_data = pd.merge(test_feats, test_lb, on="batsman")
    merged = pd.merge(train_data, test_data[["batsman","CBI_2024"]], on="batsman")

    feature_cols = [
        "strike_rate","avg","boundary_pct","dot_pct",
        "pp_sr","mid_sr","death_sr","pp_boundary_pct",
        "mean_bowler_rank","mean_opp_rank","n_matches","inn2_sr"
    ]

    X_train = train_data[feature_cols].fillna(0).values
    y_train = train_data["CBI_Train"].values
    X_test  = merged[feature_cols].fillna(0).values
    y_test  = merged["CBI_2024"].values

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # GradientBoosting model
    model = GradientBoostingRegressor(
        n_estimators=200, learning_rate=0.05, max_depth=3,
        min_samples_leaf=3, subsample=0.8, random_state=42
    )
    model.fit(X_train_s, y_train)

    # Also fit Ridge for comparison
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_s, y_train)

    # Predict on known 2024 players
    preds = model.predict(X_test_s)
    rho_ml, p_ml = spearmanr(preds, y_test)
    r_ml, _      = pearsonr(preds, y_test)

    cv_scores = cross_val_score(model, X_train_s, y_train, cv=5, scoring="r2")
    logging.info(f"  ML model: Spearman ρ={rho_ml:.4f}, Pearson r={r_ml:.4f}, p={p_ml:.4f}")
    logging.info(f"  Cross-val R² (train): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # Predict 2024 CBI for ALL players who have prior data
    all_test_feats = extract_features(test_proc, min_balls=10)  # lower threshold for predictions
    all_test_X = all_test_feats[feature_cols].fillna(0).values
    all_test_X_s = scaler.transform(all_test_X)
    all_test_feats["CBI_Predicted_2024"] = model.predict(all_test_X_s)
    all_test_feats["CBI_Actual_2024"] = all_test_feats["batsman"].map(
        dict(zip(test_lb["batsman"], test_lb["CBI_2024"]))
    )

    # Prediction results for evaluation set
    pred_df = merged[["batsman"]].copy()
    pred_df["CBI_Train_History"] = merged["CBI_Train"].values
    pred_df["CBI_Predicted_2024"] = preds
    pred_df["CBI_Actual_2024"]  = y_test
    pred_df["Abs_Error"]        = abs(preds - y_test)
    pred_df["Rank_Predicted"]   = pd.Series(preds).rank(ascending=False).values
    pred_df["Rank_Actual"]      = pd.Series(y_test).rank(ascending=False).values
    pred_df["Rank_Error"]       = abs(pred_df["Rank_Predicted"] - pred_df["Rank_Actual"])
    pred_df = pred_df.sort_values("Rank_Predicted").reset_index(drop=True)

    # Feature importance
    feat_imp = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)

    ml_summary = pd.DataFrame({
        "Metric": [
            "Model Type",
            "Training Data",
            "Test Year",
            "Overlapping Players (eval)",
            "Spearman ρ (Predicted vs Actual 2024)",
            "Pearson r (Predicted vs Actual 2024)",
            "P-Value",
            "Cross-Val R² (5-fold on train)",
            "Mean Absolute Error (CBI Index)",
        ],
        "Value": [
            "GradientBoostingRegressor (n=200, lr=0.05)",
            f"{train_years} combined",
            holdout_year,
            len(merged),
            round(rho_ml, 4),
            round(r_ml, 4),
            round(p_ml, 4),
            f"{cv_scores.mean():.3f} ± {cv_scores.std():.3f}",
            round(abs(preds - y_test).mean(), 4),
        ]
    })

    # Save model and scaler for dashboard use
# Ensure the directory exists
    output_dir = "models"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the files using os.path.join for Windows compatibility
    joblib.dump(model, os.path.join(output_dir, "cbi_ml_model.pkl"))
    joblib.dump(scaler, os.path.join(output_dir, "cbi_ml_scaler.pkl"))
    joblib.dump(feature_cols, os.path.join(output_dir, "cbi_ml_features.pkl"))
    
    logging.info(f"Model saved successfully to {output_dir}/")

    return ml_summary, pred_df, feat_imp, model, scaler, feature_cols


# ─────────────────────────────────────────────────────────────────────────────
# MASTER EXECUTION
# ─────────────────────────────────────────────────────────────────────────────
def run_all_validations(data_dir="t20_json_data",
                         output_file="CBI_Validation_Suite_v2.xlsx"):
    logging.info("=== CBI VALIDATION SUITE v2.0 ===")

    pipeline = TournamentDataPipeline(data_dir)
    raw_data = pipeline.ingest_all_tournaments()
    if raw_data.empty:
        logging.critical(f"No data in '{data_dir}'. Place Cricsheet CSVs there.")
        return

    engine_base = CBIEngine()
    processed_base = engine_base.evaluate_policy(raw_data.copy())

    # Run all tests
    sens_summary, sens_matrix = run_sensitivity_analysis(raw_data)
    bootstrap_df = run_bootstrap_confidence_intervals(processed_base)
    null_result, null_per_iter, z_score = run_null_hypothesis_test_fixed(processed_base, raw_data)
    pred_results, pred_detail = run_predictive_validity_fixed(raw_data)
    ml_summary, ml_preds, feat_imp, model, scaler, feat_cols = build_ml_predictive_model(raw_data)

    # Write Excel
    logging.info(f"Writing results to '{output_file}'...")
    with pd.ExcelWriter(output_file, engine="openpyxl") as w:
        # Overview
        overview = pd.DataFrame({
            "Test": ["1. Sensitivity Analysis", "2. Bootstrap 95% CIs",
                     "3. Null Hypothesis (Fixed)", "4. Predictive Validity (Cumulative)",
                     "5. ML Predictive Model"],
            "Threshold": ["StdDev < 5 positions", "CI width < 0.05 top-10",
                          "Z > 2.0", "ρ > 0.40", "ρ > 0.40 on holdout"],
            "Status": [
                f"Mean StdDev = {sens_summary['Rank_StdDev'].iloc[:-1].mean():.2f}",
                f"Mean CI width = {bootstrap_df['CI_Width'].mean():.3f}",
                f"Z = {z_score:.2f}",
                f"ρ = {pred_results[pred_results['Test_Type'].str.contains('PRIMARY')]['Spearman_rho'].values[0]:.4f}" if len(pred_results) > 0 else "N/A",
                ml_summary[ml_summary["Metric"].str.contains("Spearman")]["Value"].values[0] if len(ml_summary) > 0 else "N/A",
            ]
        })
        overview.to_excel(w, sheet_name="0_Overview", index=False)

        sens_summary.to_excel(w, sheet_name="1_Sensitivity_Summary", index=False)
        sens_matrix.to_excel(w, sheet_name="1_Sensitivity_RawMatrix", index=False)
        bootstrap_df.to_excel(w, sheet_name="2_Bootstrap_CIs", index=False)
        null_result.to_excel(w, sheet_name="3_Null_Hypothesis_Fixed", index=False)
        null_per_iter.to_excel(w, sheet_name="3_Null_PerIteration", index=False)
        pred_results.to_excel(w, sheet_name="4_Predictive_Validity", index=False)
        pred_detail.to_excel(w, sheet_name="4_PredValidity_PlayerDetail", index=False)
        ml_summary.to_excel(w, sheet_name="5_ML_Model_Summary", index=False)
        ml_preds.to_excel(w, sheet_name="5_ML_Predictions", index=False)
        feat_imp.to_excel(w, sheet_name="5_ML_FeatureImportance", index=False)

    logging.info(f"Done → '{output_file}'")


if __name__ == "__main__":
    run_all_validations(data_dir="t20_json_data",
                        output_file="CBI_Validation_Suite_v2.xlsx")
