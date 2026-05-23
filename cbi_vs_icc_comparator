import os
import sys
import logging
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# Safely import components from your existing production file
try:
    from cbi_advanced_suite import (
        TournamentDataPipeline, 
        CBIEngine, 
        CONFIG, 
        HISTORICAL_BOWLER_RANKINGS, 
        TEAM_RANKINGS_BY_YEAR
    )
except ImportError:
    logging.critical("Could not find 'cbi_advanced_suite.py'. Ensure this script is placed in the same directory.")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# ==============================================================================
# ENGINE: EMULATING THE OFFICIAL ICC MEN'S T20 BATTING RATING ALGORITHM
# ==============================================================================
class ICCT20RatingEngine:
    @staticmethod
    def compute_match_ratings(raw_df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Analyzing match environments to determine dynamic ICC Match Ratings...")
        
        # 1. Deduce Match Winners and Total Innings Runs directly from data
        match_metrics = raw_df.groupby(['match_id', 'innings']).agg(
            inn_runs=('runs_total', 'sum'),
            batting_team=('batting_team', 'first')
        ).reset_index()
        
        match_winners = {}
        team_totals = {}
        
        for m_id, grp in match_metrics.groupby('match_id'):
            i1 = grp[grp['innings'] == 1]
            i2 = grp[grp['innings'] == 2]
            
            r1 = i1['inn_runs'].sum() if not i1.empty else 0
            r2 = i2['inn_runs'].sum() if not i2.empty else 0
            t1 = i1['batting_team'].iloc[0] if not i1.empty else None
            t2 = i2['batting_team'].iloc[0] if not i2.empty else None
            
            team_totals[(m_id, 1)] = max(r1, 1)
            team_totals[(m_id, 2)] = max(r2, 1)
            
            if r1 > r2:
                match_winners[m_id] = t1
            elif r2 > r1:
                match_winners[m_id] = t2
            else:
                match_winners[m_id] = None

        # 2. Extract individual match performance matrices for every batsman
        player_match_perf = raw_df.groupby(['match_id', 'tournament_year', 'batting_team', 'batsman']).agg(
            runs=('runs_batter', 'sum'),
            balls=('is_legal', 'sum'),
            is_out=('is_out', 'max'),
            avg_bowler_rank=('bowler_rank', 'mean'),
            opp_team_rank=('opp_team_rank', 'first'),
            innings=('innings', 'first')
        ).reset_index()

        icc_points_list = []
        
        # 3. Calculate the standard 0-1000 point allocation for each match performance
        for _, row in player_match_perf.iterrows():
            if row['balls'] == 0:
                icc_points_list.append(0.0)
                continue
                
            runs = row['runs']
            balls = row['balls']
            sr = (runs / balls) * 100.0
            t_total = team_totals.get((row['match_id'], row['innings']), 150)
            
            # Base run rating relative to team total
            base_perf = runs * (150.0 / t_total) * 5.0
            
            # Quality of opposition scaling factor
            opp_factor = (row['avg_bowler_rank'] + row['opp_team_rank']) / 650.0
            weighted_base = base_perf * opp_factor
            
            # T20 rapid scoring incentive curve
            if sr >= 130:
                sr_mod = (sr - 130) * 1.5
            elif sr >= 100:
                sr_mod = (sr - 100) * 2.0
            else:
                sr_mod = -60.0
                
            # Not Out bonus and Match Outcome premium
            not_out_bonus = 15.0 if row['is_out'] == 0 else 0.0
            win_bonus = 30.0 if match_winners.get(row['match_id']) == row['batting_team'] else 0.0
            
            # Compile final performance score out of 1000
            match_rating = np.clip(weighted_base + sr_mod + not_out_bonus + win_bonus, 0, 1000)
            icc_points_list.append(match_rating)
            
        player_match_perf['match_icc_rating'] = icc_points_list
        return player_match_perf

    @staticmethod
    def aggregate_global_ratings(match_perf_df: pd.DataFrame) -> pd.DataFrame:
        # Replicating the ICC rolling decay window using sequence-based averages
        agg_df = match_perf_df.groupby('batsman').agg(
            Matches=('match_id', 'count'),
            Total_Runs=('runs', 'sum'),
            ICC_Rating=('match_icc_rating', 'mean')
        ).reset_index()
        
        # Filter by minimum qualification base configured in existing system
        agg_df = agg_df[agg_df['Matches'] >= 3].copy()
        agg_df['ICC_Rank'] = agg_df['ICC_Rating'].rank(ascending=False, method='min').astype(int)
        return agg_df.sort_values(by='ICC_Rank').reset_index(drop=True)

# ==============================================================================
# COMPARATIVE VALIDATION EXECUTION ENGINE
# ==============================================================================
def execute_paradigm_comparison():
    DATA_DIR = "t20_json_data"
    OUTPUT_FILE = "CBI_vs_ICC_Comparison_Report.xlsx"
    
    # 1. Pipeline execution using existing codebase structures
    pipeline = TournamentDataPipeline(DATA_DIR)
    raw_data = pipeline.ingest_all_tournaments()
    
    if raw_data.empty:
        logging.error("No source tournament datasets identified.")
        return
        
    # 2. Compute Contextual Batting Intelligence metrics
    cbi_engine = CBIEngine()
    cbi_processed = cbi_engine.evaluate_policy(raw_data)
    cbi_leaderboard = cbi_engine.generate_leaderboard(cbi_processed)
    
    # 3. Compute Simulated Official ICC Rankings
    icc_match_data = ICCT20RatingEngine.compute_match_ratings(raw_data)
    icc_leaderboard = ICCT20RatingEngine.aggregate_global_ratings(icc_match_data)
    
    # 4. Consolidate systems to extract statistical delta variances
    comparison_master = pd.merge(
        cbi_leaderboard[['batsman', 'Balls', 'CBI_Index', 'CBI_Rank']],
        icc_leaderboard[['batsman', 'Matches', 'Total_Runs', 'ICC_Rating', 'ICC_Rank']],
        on='batsman'
    )
    
    comparison_master['Rank_Delta_CBI_minus_ICC'] = comparison_master['CBI_Rank'] - comparison_master['ICC_Rank']
    comparison_master = comparison_master.sort_values(by='CBI_Rank').reset_index(drop=True)
    
    # Calculate global parametric agreement metrics
    rho, _ = spearmanr(comparison_master['CBI_Rank'], comparison_master['ICC_Rank'])
    print(f"\n" + "="*60)
    print(f"PARADIGM CORRELATION ANALYSIS (Spearman Rank Rho): {rho:.4f}")
    print("="*60)
    print(" -> Strong structural alignment between metrics." if rho > 0.75 
          else " -> Significant divergence: CBI penalizes contextual risk where ICC rewards raw volume.")
    print("="*60 + "\n")
    
    # 5. Build multi-sheet Excel comparative workbook
    with pd.ExcelWriter(OUTPUT_FILE, engine='openpyxl') as writer:
        comparison_master.to_excel(writer, sheet_name="Global Paradigm Comparison", index=False)
        
        # Add a sheet tracking the divergence between metrics
        correlation_summary = pd.DataFrame({
            'Metric Feature': ['Total Analyzed Batsmen', 'Spearman Rank Correlation (Rho)', 'Mean Absolute Rank Divergence'],
            'Value': [len(comparison_master), round(rho, 4), round(np.abs(comparison_master['Rank_Delta_CBI_minus_ICC']).mean(), 2)]
        })
        correlation_summary.to_excel(writer, sheet_name="Statistical Insights", index=False)
        
        # Generate tournament-by-tournament breakdowns
        for yr in sorted(raw_data['tournament_year'].unique()):
            yr_raw = raw_data[raw_data['tournament_year'] == yr]
            
            cbi_yr = cbi_engine.generate_leaderboard(cbi_engine.evaluate_policy(yr_raw))
            icc_yr = ICCT20RatingEngine.aggregate_global_ratings(icc_match_data[icc_match_data['tournament_year'] == yr])
            
            yr_merged = pd.merge(
                cbi_yr[['batsman', 'CBI_Index', 'CBI_Rank']],
                icc_yr[['batsman', 'ICC_Rating', 'ICC_Rank']],
                on='batsman'
            ).sort_values(by='CBI_Rank')
            
            yr_merged.to_excel(writer, sheet_name=f"WC {yr} Comparison", index=False)

    logging.info(f"Comparison matrix compiled perfectly. Sheet generated -> '{OUTPUT_FILE}'")

if __name__ == "__main__":
    execute_paradigm_comparison()
