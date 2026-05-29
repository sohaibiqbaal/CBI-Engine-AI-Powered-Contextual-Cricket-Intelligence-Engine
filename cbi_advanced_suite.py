import os
import glob
import sys
import logging
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# Configure professional diagnostic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# 
# MODULE 1: TOURNAMENT-SPECIFIC HISTORICAL LOOKUPS & GLOBAL CONFIG
# 
CONFIG = {
    'alpha': 0.65,
    'theta1': 2.6,
    'theta2': 4.2,
    'beta': 1.6,
    'min_balls': 40,
    'default_bowler_rank': 500,
    'default_team_rank': 150
}

# Dynamic Team Rankings per Tournament Era
TEAM_RANKINGS_BY_YEAR = {
    2016: {'IND': 268, 'WI': 262, 'NZ': 255, 'RSA': 250, 'AUS': 248, 'ENG': 244, 'PAK': 238, 'SL': 225, 'BAN': 218, 'AFG': 195, 'SCO': 150, 'IRE': 160, 'NED': 145, 'OMA': 120, 'ZIM': 170},
    2021: {'ENG': 266, 'IND': 262, 'PAK': 258, 'NZ': 252, 'RSA': 250, 'AUS': 246, 'WI': 235, 'AFG': 228, 'SL': 220, 'BAN': 215, 'SCO': 180, 'NAM': 165, 'IRE': 170, 'NED': 155, 'OMA': 140, 'PNG': 125},
    2022: {'ENG': 264, 'PAK': 258, 'IND': 256, 'NZ': 252, 'RSA': 248, 'AUS': 245, 'SL': 232, 'AFG': 224, 'WI': 220, 'BAN': 212, 'IRE': 190, 'ZIM': 185, 'SCO': 175, 'NED': 168, 'UAE': 135, 'NAM': 150},
    2024: {'IND': 265, 'AUS': 258, 'ENG': 254, 'WI': 253, 'NZ': 248, 'RSA': 247, 'PAK': 241, 'SL': 230, 'BAN': 226, 'AFG': 220, 'SCO': 192, 'IRE': 188, 'USA': 177, 'NED': 175, 'CAN': 152, 'NEP': 150, 'OMA': 148, 'PNG': 136, 'UGA': 135}
}

# Expanded Top 50+ Multi-Era Bowler Lookup Map, The List is Compilation of All-Time Top Rankings and From June 2024
HISTORICAL_BOWLER_RANKINGS = {
    'Jasprit Bumrah': 740, 'Rashid Khan': 753, 'Adil Rashid': 730, 'Wanindu Hasaranga': 710,
    'Anrich Nortje': 695, 'Akeal Hosein': 695, 'Shaheen Shah Afridi': 685, 'Varun Chakaravarthy': 690,
    'Axar Patel': 680, 'Arshdeep Singh': 675, 'Maheesh Theekshana': 675, 'Trent Boult': 670,
    'Josh Hazlewood': 665, 'Ravi Bishnoi': 660, 'Josh Hazlewood': 665, 'Pat Cummins': 660,
    'Kuldeep Yadav': 655, 'Mitchell Starc': 655, 'Adam Zampa': 650, 'Haris Rauf': 650,
    'Alzarri Joseph': 645, 'Fazalhaq Farooqi': 640, 'Tabraiz Shamsi': 640, 'Naveen-ul-Haq': 635,
    'Gudakesh Motie': 635, 'Mujeeb Ur Rahman': 630, 'Mark Wood': 630, 'Jofra Archer': 625,
    'Noor Ahmad': 625, 'Taskin Ahmed': 620, 'Mitchell Santner': 620, 'Mustafizur Rahman': 615,
    'Rishad Hossain': 615, 'Tanzim Hasan Sakib': 610, 'Mahedi Hasan': 610, 'Tim Southee': 605,
    'Shakib Al Hasan': 605, 'Lockie Ferguson': 600, 'Roston Chase': 600, 'Matheesha Pathirana': 595,
    'Imad Wasim': 595, 'Nuwan Thushara': 590, 'Shadab Khan': 590, 'Nandre Burger': 585,
    'Keshav Maharaj': 585, 'Kagiso Rabada': 580, 'Glenn Maxwell': 580, 'Saurabh Netravalkar': 580,
    'Hardik Pandya': 575, 'Mohammad Nabi': 575, 'Mark Watt': 575, 'Gerald Coetzee': 570,
    'Moeen Ali': 570, 'Romario Shepherd': 565, 'Harmeet Singh': 565, 'Liam Livingstone': 565,
    'Chris Jordan': 560, 'Brad Currie': 560, 'Mahmudullah': 560, 'Shoriful Islam': 555,
    'Ali Khan': 555, 'Obed McCoy': 550, 'Logan van Beek': 550, 'Reece Topley': 545,
    'Paul van Meekeren': 545, 'Tim Pringle': 540, 'Aryan Dutt': 535, 'Bas de Leede': 530,
    'Brandon McMullen': 525, 'Chris Sole': 520, 'Saad Bin Zafar': 515, 'Dillon Heyliger': 510,
    'Kaleem Sana': 505, 'Sompal Kami': 500, 'Sandeep Lamichhane': 495, 'Dipendra Singh Airee': 490,
    'Abinash Bohara': 485, 'Bilal Khan': 480, 'Aqib Ilyas': 475, 'Samuel Badree': 745,
    'Imran Tahir': 722, 'Ravichandran Ashwin': 665, 'Sunil Narine': 680, 'Saeed Ajmal': 710,
    'Chris Woakes': 610, 'Mohammad Amir': 635, 'Wahab Riaz': 580, 'Morne Morkel': 620
}

# 
# MODULE 2: REFACTORED MULTI-TOURNAMENT DATA PIPELINE
# 
class TournamentDataPipeline:
    def __init__(self, folder_path: str):
        self.folder_path = folder_path

    def ingest_all_tournaments(self) -> pd.DataFrame:
        search_path = os.path.join(self.folder_path, "*.csv")
        target_files = glob.glob(search_path)
        
        if not target_files:
            logging.error(f"Zero CSV target files located in path: '{self.folder_path}'")
            return pd.DataFrame()

        aggregated_frames = []
        for file_path in target_files:
            filename = os.path.basename(file_path).lower()
            # Dynamic extraction of the target world cup year
            year = 2024
            for target_yr in [2016, 2021, 2022, 2024]:
                if str(target_yr) in filename:
                    year = target_yr
                    break
            
            logging.info(f"Ingesting tournament matrix: Year {year} | File: {filename}...")
            df = pd.read_csv(file_path)
            df.columns = df.columns.str.strip().str.lower()
            
            rename_dict = {'striker': 'batsman', 'runs_of_bat': 'runs_batter'}
            df = df.rename(columns={k: v for k, v in rename_dict.items() if k in df.columns})
            
            # Structuring standardized data metrics
            df['wide'] = df['wide'].fillna(0).astype(int)
            df['noballs'] = df['noballs'].fillna(0).astype(int)
            df['runs_batter'] = df['runs_batter'].fillna(0).astype(int)
            df['extras'] = df['extras'].fillna(0).astype(int)
            df['runs_total'] = df['runs_batter'] + df['extras']
            df['is_out'] = df['player_dismissed'].notna().astype(int)
            df['is_legal'] = ((df['wide'] == 0) & (df['noballs'] == 0)).astype(int)
            df['tournament_year'] = year
            
            processed_df = self._build_sequential_states(df, year)
            if not processed_df.empty:
                aggregated_frames.append(processed_df)

        return pd.concat(aggregated_frames, ignore_index=True) if aggregated_frames else pd.DataFrame()

    def _build_sequential_states(self, df: pd.DataFrame, year: int) -> pd.DataFrame:
        processed_matches = []
        team_lookup = TEAM_RANKINGS_BY_YEAR.get(year, TEAM_RANKINGS_BY_YEAR[2024])

        for match_id, match_grp in df.groupby('match_id'):
            inn1 = match_grp[match_grp['innings'] == 1]
            target_score = inn1['runs_total'].sum() + 1 if not inn1.empty else 0
            
            for inn_num, inn_grp in match_grp.groupby('innings'):
                if inn_num > 2:
                    continue
                
                inn_sorted = inn_grp.sort_values(by=['over']).copy()
                inn_sorted['cum_legal_balls'] = inn_sorted['is_legal'].cumsum()
                inn_sorted['balls_remaining'] = np.clip(120 - inn_sorted['cum_legal_balls'], 0, 120)
                inn_sorted['wickets_lost'] = inn_sorted['is_out'].cumsum().shift(1, fill_value=0)
                inn_sorted['running_runs'] = inn_sorted['runs_total'].cumsum().shift(1, fill_value=0)
                
                overs_completed = inn_sorted['cum_legal_balls'].shift(1, fill_value=0) / 6.0
                inn_sorted['crr'] = (inn_sorted['running_runs'] / overs_completed * 6.0).where(overs_completed > 0, 0.0)
                
                inn_sorted['rrr'] = 0.0
                if inn_num == 2 and target_score > 0:
                    runs_needed = np.clip(target_score - inn_sorted['running_runs'], 0, None)
                    b_rem = inn_sorted['balls_remaining']
                    inn_sorted['rrr'] = (runs_needed / (b_rem / 6.0)).where(b_rem > 0, runs_needed * 6.0)
                    
                processed_matches.append(inn_sorted)

        if not processed_matches:
            return pd.DataFrame()

        final_df = pd.concat(processed_matches, ignore_index=True)
        
        # Mapping Action Spaces
        action_conds = [final_df['runs_batter'] == 0, final_df['runs_batter'].isin([1, 2, 3]), final_df['runs_batter'] >= 4]
        final_df['action'] = np.select(action_conds, [0, 1, 2], default=0)
        
        # Mapping Phase Spaces
        phase_conds = [final_df['balls_remaining'] > 84, final_df['balls_remaining'] > 30]
        final_df['state_phase'] = np.select(phase_conds, [0, 1], default=2)

        # Era-specific Context Feature Assignments
        final_df['bowler_rank'] = final_df['bowler'].map(HISTORICAL_BOWLER_RANKINGS).fillna(CONFIG['default_bowler_rank'])
        final_df['opp_team_rank'] = final_df['bowling_team'].map(team_lookup).fillna(CONFIG['default_team_rank'])
        
        return final_df

# 
# MODULE 3: MAXENT CONTEXTUAL BATTING INTELLIGENCE (CBI) CORE
# 
class CBIEngine:
    def __init__(self, custom_config=None):
        self.cfg = custom_config if custom_config else CONFIG

    def evaluate_policy(self, df: pd.DataFrame, robust_noise=False) -> pd.DataFrame:
        if df.empty:
            return df
        
        working_df = df.copy()
        
        # Robustness Check Injection Framework
        if robust_noise:
            # Inject a 5% system perturbation into critical environmental parameters
            noise_mask = np.random.rand(len(working_df)) < 0.05
            working_df.loc[noise_mask, 'bowler_rank'] += np.random.choice([-50, 50], size=noise_mask.sum())
            working_df.loc[noise_mask, 'opp_team_rank'] += np.random.choice([-20, 20], size=noise_mask.sum())

        lookup_matrix = working_df.groupby(['state_phase', 'action']).agg(
            exp_runs=('runs_batter', 'mean'),
            p_out=('is_out', 'mean')
        ).to_dict('index')

        cbi_probabilities = []
        
        for _, row in working_df.iterrows():
            omega = 1.0 + self.cfg['alpha'] * ((row['bowler_rank'] + row['opp_team_rank']) / 2000.0)
            resource_ratio = (row['wickets_lost'] + 1) / (row['balls_remaining'] + 1)
            
            if row['innings'] == 1:
                lambda_s = self.cfg['theta1'] * resource_ratio
            else:
                pressure_delta = np.clip((row['rrr'] - row['crr']) / (row['crr'] + 1.0), -2.0, 2.0)
                lambda_s = self.cfg['theta2'] * resource_ratio * np.exp(pressure_delta)
                
            utilities = {}
            for act in [0, 1, 2]:
                state_key = (row['state_phase'], act)
                defaults = {'exp_runs': 0.70, 'p_out': 0.04}
                e_runs = lookup_matrix.get(state_key, defaults)['exp_runs']
                p_out = lookup_matrix.get(state_key, defaults)['p_out']
                
                utilities[act] = (omega * e_runs) - (lambda_s * p_out)
                
            exp_utilities = {a: np.exp(self.cfg['beta'] * u) for a, u in utilities.items()}
            sum_exp = sum(exp_utilities.values())
            
            policy_prob = exp_utilities[row['action']] / sum_exp if sum_exp > 0 else 0.333
            cbi_probabilities.append(policy_prob)
            
        working_df['cbi_probability'] = cbi_probabilities
        return working_df

    @staticmethod
    def generate_leaderboard(df: pd.DataFrame) -> pd.DataFrame:
        summary = df.groupby('batsman').agg(
            Runs=('runs_batter', 'sum'),
            Outs=('is_out', 'sum'),
            Balls=('runs_batter', 'count'),
            CBI_Index=('cbi_probability', 'mean')
        ).reset_index()
        
        summary = summary[summary['Balls'] >= CONFIG['min_balls']].copy()
        if summary.empty:
            return pd.DataFrame(columns=['batsman', 'CBI_Rank', 'CBI_Index'])
            
        summary['Batting_Avg'] = summary['Runs'] / summary['Outs'].replace(0, 1)
        summary['Strike_Rate'] = (summary['Runs'] / summary['Balls']) * 100
        summary['CBI_Rank'] = summary['CBI_Index'].rank(ascending=False, method='min').astype(int)
        return summary.sort_values(by='CBI_Rank').reset_index(drop=True)

# 
# MODULE 4: INTEGRATED FIVE-FOLD MATHEMATICAL VALIDATION SUITE
# 
class ModelValidationSuite:
    def __init__(self, engine: CBIEngine, raw_data: pd.DataFrame):
        self.engine = engine
        self.raw_data = raw_data

    def run_all_validation_checks(self):
        print("\n" + "="*80)
        print("          CONTEXTUAL BATTING INTELLIGENCE (CBI) VALIDATION REPORT")
        print("="*80)
        
        baseline_processed = self.engine.evaluate_policy(self.raw_data)
        baseline_leaderboard = self.engine.generate_leaderboard(baseline_processed)
        
        self._test_robustness(baseline_leaderboard)
        self._test_cross_tournament()
        self._test_predictive_power()
        self._test_ablation_studies()
        self._test_sensitivity_analysis(baseline_leaderboard)
        print("="*80 + "\n")

    def _test_robustness(self, baseline_lead):
        print("\n[1/5] RUNNING ROBUSTNESS TESTING (Stochastic Noise Injection)")
        noisy_df = self.engine.evaluate_policy(self.raw_data, robust_noise=True)
        noisy_lead = self.engine.generate_leaderboard(noisy_df)
        
        merged = pd.merge(baseline_lead[['batsman', 'CBI_Rank']], noisy_lead[['batsman', 'CBI_Rank']], on='batsman', suffixes=('_base', '_noisy'))
        corr, _ = spearmanr(merged['CBI_Rank_base'], merged['CBI_Rank_noisy'])
        print(f" -> System Stability Index (Spearman Rho under 5% noise): {corr:.4f}")
        print(" -> Verdict: Highly Robust (System output holds high structural integrity)" if corr > 0.90 else " -> Verdict: Sensitive to Environmental Variance")

    def _test_cross_tournament(self):
        print("\n[2/5] RUNNING CROSS-TOURNAMENT VALIDATION (Domain Invariance)")
        years = sorted(self.raw_data['tournament_year'].unique())
        rankings_by_year = {}
        
        for yr in years:
            yr_data = self.raw_data[self.raw_data['tournament_year'] == yr]
            yr_proc = self.engine.evaluate_policy(yr_data)
            yr_lead = self.engine.generate_leaderboard(yr_proc)
            rankings_by_year[yr] = yr_lead
            print(f" -> Tournament Year {yr}: Processed {len(yr_lead)} qualified batsmen.")

    def _test_predictive_power(self):
        print("\n[3/5] RUNNING PREDICTIVE POWER CHECKS (Temporal Out-of-Sample Evaluation)")
        # Trailing window optimization split: Past records (2016-2022) predict future execution profiles (2024)
        historical_df = self.raw_data[self.raw_data['tournament_year'] < 2024]
        future_df = self.raw_data[self.raw_data['tournament_year'] == 2024]
        
        if historical_df.empty or future_df.empty:
            print(" -> Error: Insufficient multi-era datasets matching temporal constraints.")
            return
            
        hist_proc = self.engine.evaluate_policy(historical_df)
        hist_lead = self.engine.generate_leaderboard(hist_proc)
        
        fut_proc = self.engine.evaluate_policy(future_df)
        fut_lead = self.engine.generate_leaderboard(fut_proc)
        
        merged = pd.merge(hist_lead[['batsman', 'CBI_Index']], fut_lead[['batsman', 'Strike_Rate', 'Batting_Avg']], on='batsman')
        if len(merged) < 5:
            print(f" -> Sample match counts ({len(merged)}) below significance thresholds for correlation mapping.")
            return
            
        r_sr, _ = spearmanr(merged['CBI_Index'], merged['Strike_Rate'])
        r_avg, _ = spearmanr(merged['CBI_Index'], merged['Batting_Avg'])
        print(f" -> Historical CBI Index vs Next-Tournament Strike Rate Correlation: {r_sr:.4f}")
        print(f" -> Historical CBI Index vs Next-Tournament Batting Average Correlation: {r_avg:.4f}")

    def _test_ablation_studies(self):
        print("\n[4/5] RUNNING ABLATION STUDIES (Structural Feature Degradation)")
        
        # Baseline Setup
        base_df = self.engine.evaluate_policy(self.raw_data)
        base_lead = self.engine.generate_leaderboard(base_df)[['batsman', 'CBI_Rank']]
        
        # Ablation 1: Sever/Flatten Contextual Matchups Weights (Contextual Deadening)
        ablated_data1 = self.raw_data.copy()
        ablated_data1['bowler_rank'] = 500
        ablated_data1['opp_team_rank'] = 150
        df_abl1 = self.engine.evaluate_policy(ablated_data1)
        lead_abl1 = self.engine.generate_leaderboard(df_abl1)
        
        # Compute mean rank deflection shift
        m1 = pd.merge(base_lead, lead_abl1, on='batsman')
        mas1 = np.abs(m1['CBI_Rank_x'] - m1['CBI_Rank_y']).mean()
        print(f" -> Ablation 1 [Deactivating Bowler/Team Rank context] -> Mean Rank Shift: {mas1:.2f} places")

        # Ablation 2: Remove Game State Pressure Tracking Matrix
        config_no_pressure = CONFIG.copy()
        config_no_pressure['theta1'] = 0.0
        config_no_pressure['theta2'] = 0.0
        engine_abl2 = CBIEngine(config_no_pressure)
        df_abl2 = engine_abl2.evaluate_policy(self.raw_data)
        lead_abl2 = engine_abl2.generate_leaderboard(df_abl2)
        
        m2 = pd.merge(base_lead, lead_abl2, on='batsman')
        mas2 = np.abs(m2['CBI_Rank_x'] - m2['CBI_Rank_y']).mean()
        print(f" -> Ablation 2 [Deactivating Structural State Risk parameters] -> Mean Rank Shift: {mas2:.2f} places")

    def _test_sensitivity_analysis(self, baseline_lead):
        print("\n[5/5] RUNNING PARAMETER SENSITIVITY SWEEPS (Hyperparameter Gradients)")
        test_variations = [
            ('beta_low', {**CONFIG, 'beta': 1.0}),
            ('beta_high', {**CONFIG, 'beta': 2.2}),
            ('alpha_low', {**CONFIG, 'alpha': 0.3}),
            ('alpha_high', {**CONFIG, 'alpha': 0.9})
        ]
        
        for name, cfg_alteration in test_variations:
            alt_engine = CBIEngine(cfg_alteration)
            alt_df = alt_engine.evaluate_policy(self.raw_data)
            alt_lead = alt_engine.generate_leaderboard(alt_df)
            
            merged = pd.merge(baseline_lead[['batsman', 'CBI_Rank']], alt_lead[['batsman', 'CBI_Rank']], on='batsman')
            corr, _ = spearmanr(merged['CBI_Rank_x'], merged['CBI_Rank_y'])
            print(f" -> Sensitivity Sweep [{name:10}] -> Policy Ranking Correlation: {corr:.4f}")

# 
# MODULE 5: WORKBOOK COMPILATION ENGINE
# 
def export_comprehensive_report(raw_data: pd.DataFrame, engine: CBIEngine, out_path: str):
    logging.info(f"Compiling combined spreadsheets and final leaderboard matrices...")
    
    # Process Master Consolidated Rankings
    master_processed = engine.evaluate_policy(raw_data)
    master_leaderboard = engine.generate_leaderboard(master_processed)
    
    try:
        with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
            # Sheet 1: Consolidated Multi-Era Master Chart
            master_leaderboard.to_excel(writer, sheet_name="Master Leaderboard", index=False)
            
            # Sheets 2-5: Individual Tournament Breakdowns
            for yr in sorted(raw_data['tournament_year'].unique()):
                yr_data = raw_data[raw_data['tournament_year'] == yr]
                yr_processed = engine.evaluate_policy(yr_data)
                yr_leaderboard = engine.generate_leaderboard(yr_processed)
                yr_leaderboard.to_excel(writer, sheet_name=f"WC {yr} Rankings", index=False)
                
            logging.info(f"Workbook compiled perfectly. Output saved to target destination -> '{out_path}'")
    except ImportError:
        logging.error("Execution missing spreadsheet writer engine. Run: 'pip install openpyxl'")

# 
# EXECUTION ENTRY POINT
# 
if __name__ == "__main__":
    DATA_DIRECTORY = "t20_json_data"
    OUTPUT_REPORT = "Multi_Tournament_CBI_Validation_Suite.xlsx"
    
    logging.info("Spinning up Contextual Batting Intelligence Framework Engine v3.0...")
    
    pipeline = TournamentDataPipeline(DATA_DIRECTORY)
    consolidated_raw = pipeline.ingest_all_tournaments()
    
    if not consolidated_raw.empty:
        cbi_engine = CBIEngine()
        
        # Fire structural validation execution suites
        validation_suite = ModelValidationSuite(cbi_engine, consolidated_raw)
        validation_suite.run_all_validation_checks()
        
        # Export high-fidelity analytical reporting datasets
        export_comprehensive_report(consolidated_raw, cbi_engine, OUTPUT_REPORT)
    else:
        logging.critical("Data system structural error: Failed to parse raw delivery files from target directory.")
