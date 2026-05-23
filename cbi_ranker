"""
Contextual Batting Intelligence (CBI) Engine v2.2
Author: Muhammad Sohaib Iqbal
Description: Production-ready decision-theoretic evaluation engine for cricket analytics.
             Fixes sorting KeyError: 'ball' by utilizing natural index arrays.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd

# Configure structured production logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# ==============================================================================
# MODULE 1: GLOBAL CONFIGURATIONS & HISTORICAL SEEDING
# ==============================================================================
CONFIG = {
    'alpha': 0.65,
    'theta1': 2.6,
    'theta2': 4.2,
    'beta': 1.6,
    'min_balls': 40,
    'default_bowler_rank': 480,
    'default_team_rank': 130
}

BOWLER_ICC_RANKINGS = {
    'Jasprit Bumrah': 740, 'Anrich Nortje': 695, 'Shaheen Shah Afridi': 685,
    'Arshdeep Singh': 675, 'Trent Boult': 670, 'Josh Hazlewood': 665,
    'Pat Cummins': 660, 'Mitchell Starc': 655, 'Haris Rauf': 650,
    'Alzarri Joseph': 645, 'Fazalhaq Farooqi': 640, 'Naveen-ul-Haq': 635,
    'Mark Wood': 630, 'Jofra Archer': 625, 'Taskin Ahmed': 620,
    'Mustafizur Rahman': 615, 'Tanzim Hasan Sakib': 610, 'Tim Southee': 605,
    'Lockie Ferguson': 600, 'Matheesha Pathirana': 595, 'Nuwan Thushara': 590,
    'Nandre Burger': 585, 'Kagiso Rabada': 580, 'Hardik Pandya': 575,
    'Gerald Coetzee': 570, 'Romario Shepherd': 565, 'Chris Jordan': 560,
    'Shoriful Islam': 555, 'Obed McCoy': 550, 'Reece Topley': 545,
    'Rashid Khan': 753, 'Adil Rashid': 730, 'Wanindu Hasaranga': 710,
    'Akeal Hosein': 695, 'Varun Chakaravarthy': 690, 'Axar Patel': 680,
    'Maheesh Theekshana': 675, 'Ravi Bishnoi': 660, 'Kuldeep Yadav': 655,
    'Adam Zampa': 650, 'Tabraiz Shamsi': 640, 'Gudakesh Motie': 635,
    'Mujeeb Ur Rahman': 630, 'Noor Ahmad': 625, 'Mitchell Santner': 620,
    'Rishad Hossain': 615, 'Mahedi Hasan': 610, 'Shakib Al Hasan': 605,
    'Roston Chase': 600, 'Imad Wasim': 595, 'Shadab Khan': 590,
    'Keshav Maharaj': 585, 'Glenn Maxwell': 580, 'Mohammad Nabi': 575,
    'Moeen Ali': 570, 'Liam Livingstone': 565, 'Mahmudullah': 560,
    'Saurabh Netravalkar': 580, 'Mark Watt': 575, 'Harmeet Singh': 565,
    'Brad Currie': 560, 'Ali Khan': 555, 'Logan van Beek': 550,
    'Paul van Meekeren': 545, 'Tim Pringle': 540, 'Aryan Dutt': 535,
    'Bas de Leede': 530, 'Brandon McMullen': 525, 'Chris Sole': 520,
    'Saad Bin Zafar': 515, 'Dillon Heyliger': 510, 'Kaleem Sana': 505,
    'Sompal Kami': 500, 'Sandeep Lamichhane': 495, 'Dipendra Singh Airee': 490,
    'Abinash Bohara': 485, 'Bilal Khan': 480, 'Aqib Ilyas': 475
}

OPPOSITION_TEAM_RANKINGS = {
    'IND': 265, 'AUS': 258, 'ENG': 254, 'WI': 253,
    'NZ': 248, 'RSA': 247, 'PAK': 241, 'SL': 230,
    'BAN': 226, 'AFG': 220, 'SCO': 192, 'IRE': 188,
    'USA': 177, 'NED': 175, 'CAN': 152, 'NEP': 150,
    'OMA': 148, 'PNG': 136, 'UGA': 135
}

# ==============================================================================
# MODULE 2: DATA PROCESSING & FEATURE ENGINEERING PIPELINE
# ==============================================================================
class CricketDataPipeline:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def process_dataset(self) -> pd.DataFrame:
        if not os.path.exists(self.file_path):
            logging.error(f"Target data missing from path: '{self.file_path}'")
            return pd.DataFrame()

        logging.info("Ingesting raw delivery source matrix...")
        raw_df = pd.read_csv(self.file_path)
        
        # Standardize header string cases
        raw_df.columns = raw_df.columns.str.strip().str.lower()
        
        # Map alternative Kaggle dataset naming styles securely
        rename_dict = {'striker': 'batsman', 'runs_of_bat': 'runs_batter'}
        raw_df = raw_df.rename(columns={k: v for k, v in rename_dict.items() if k in raw_df.columns})
        
        # Safe filling of null array types to protect downstream calculations
        raw_df['wide'] = raw_df['wide'].fillna(0).astype(int)
        raw_df['noballs'] = raw_df['noballs'].fillna(0).astype(int)
        raw_df['runs_batter'] = raw_df['runs_batter'].fillna(0).astype(int)
        raw_df['extras'] = raw_df['extras'].fillna(0).astype(int)
        raw_df['runs_total'] = raw_df['runs_batter'] + raw_df['extras']
        raw_df['is_out'] = raw_df['player_dismissed'].notna().astype(int)
        raw_df['is_legal'] = ((raw_df['wide'] == 0) & (raw_df['noballs'] == 0)).astype(int)

        return self._engineer_dynamic_game_states(raw_df)

    def _engineer_dynamic_game_states(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Reconstructing sequential state vectors...")
        processed_matches = []

        for match_id, match_grp in df.groupby('match_id'):
            inn1 = match_grp[match_grp['innings'] == 1]
            target_score = inn1['runs_total'].sum() + 1 if not inn1.empty else 0
            
            for inn_num, inn_grp in match_grp.groupby('innings'):
                if inn_num > 2:
                    continue  # Protect boundaries by omitting Super Overs
                
                # Fixed: Sort using only structural spatial indicators (omitting 'ball' key matches)
                inn_sorted = inn_grp.sort_values(by=['over']).copy()
                
                # Map dynamic resource tracking values
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
        
        # Categorize choices: 0=Dot, 1=Strike Rotation, 2=Boundary Attacking Choice
        action_conditions = [
            final_df['runs_batter'] == 0,
            final_df['runs_batter'].isin([1, 2, 3]),
            final_df['runs_batter'] >= 4
        ]
        final_df['action'] = np.select(action_conditions, [0, 1, 2], default=0)
        
        # Phase boundaries: 0=Powerplay, 1=Middle overs, 2=Death overs
        phase_conditions = [
            final_df['balls_remaining'] > 84,
            final_df['balls_remaining'] > 30
        ]
        final_df['state_phase'] = np.select(phase_conditions, [0, 1], default=2)

        # Map context variables
        final_df['bowler_rank'] = final_df['bowler'].map(BOWLER_ICC_RANKINGS).fillna(CONFIG['default_bowler_rank'])
        final_df['opp_team_rank'] = final_df['bowling_team'].map(OPPOSITION_TEAM_RANKINGS).fillna(CONFIG['default_team_rank'])

        return final_df


# ==============================================================================
# MODULE 3: MAXIMUM ENTROPY INVERSE REINFORCEMENT LEARNING ENGINE
# ==============================================================================
class CBIEngine:
    def __init__(self):
        self.alpha = CONFIG['alpha']
        self.theta1 = CONFIG['theta1']
        self.theta2 = CONFIG['theta2']
        self.beta = CONFIG['beta']

    def compute_policy_alignment(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
            
        logging.info("Extracting empirical policy transitions via MaxEnt Softmax...")
        
        # Build baseline lookup matrix from global data distributions
        lookup_matrix = df.groupby(['state_phase', 'action']).agg(
            exp_runs=('runs_batter', 'mean'),
            p_out=('is_out', 'mean')
        ).to_dict('index')

        cbi_probabilities = []
        
        for _, row in df.iterrows():
            omega = 1.0 + self.alpha * ((row['bowler_rank'] + row['opp_team_rank']) / 2000.0)
            resource_ratio = (row['wickets_lost'] + 1) / (row['balls_remaining'] + 1)
            
            if row['innings'] == 1:
                lambda_s = self.theta1 * resource_ratio
            else:
                pressure_delta = np.clip((row['rrr'] - row['crr']) / (row['crr'] + 1.0), -2.0, 2.0)
                lambda_s = self.theta2 * resource_ratio * np.exp(pressure_delta)
                
            utilities = {}
            for act in [0, 1, 2]:
                state_key = (row['state_phase'], act)
                defaults = {'exp_runs': 0.70, 'p_out': 0.04}
                e_runs = lookup_matrix.get(state_key, defaults)['exp_runs']
                p_out = lookup_matrix.get(state_key, defaults)['p_out']
                
                utilities[act] = (omega * e_runs) - (lambda_s * p_out)
                
            exp_utilities = {a: np.exp(self.beta * u) for a, u in utilities.items()}
            sum_exp = sum(exp_utilities.values())
            
            policy_prob = exp_utilities[row['action']] / sum_exp if sum_exp > 0 else 0.333
            cbi_probabilities.append(policy_prob)
            
        df['cbi_probability'] = cbi_probabilities
        return df


# ==============================================================================
# MODULE 4: RANKING COMPARISON & REPORT GENERATOR
# ==============================================================================
class ExcelReportGenerator:
    @staticmethod
    def construct_workbook(df: pd.DataFrame, output_path: str):
        logging.info("Compiling master sheets and computing rank differentials...")
        
        # Compile global summary statistics per player
        summary = df.groupby('batsman').agg(
            Runs=('runs_batter', 'sum'),
            Outs=('is_out', 'sum'),
            Balls=('runs_batter', 'count'),
            CBI_Index=('cbi_probability', 'mean')
        ).reset_index()
        
        # Enforce statistical noise filtering volume
        summary = summary[summary['Balls'] >= CONFIG['min_balls']].copy()
        
        # Form classical metric baselines
        summary['Batting_Avg'] = summary['Runs'] / summary['Outs'].replace(0, 1)
        summary['Strike_Rate'] = (summary['Runs'] / summary['Balls']) * 100
        
        # Symmetrize and build comparative metrics
        summary['Traditional_Rank'] = summary['Batting_Avg'].rank(ascending=False, method='min').astype(int)
        summary['CBI_Rank'] = summary['CBI_Index'].rank(ascending=False, method='min').astype(int)
        
        # Compute Rank Differential (Traditional - CBI)
        summary['Rank_Shift'] = summary['Traditional_Rank'] - summary['CBI_Rank']
        
        summary = summary.sort_values(by='CBI_Rank').reset_index(drop=True)
        
        # Reorder variables cleanly for target presentation
        master_cols = [
            'batsman', 'CBI_Rank', 'Traditional_Rank', 'Rank_Shift', 
            'CBI_Index', 'Batting_Avg', 'Strike_Rate', 'Runs', 'Balls'
        ]
        summary = summary[master_cols]

        try:
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                summary.to_excel(writer, sheet_name="Master Leaderboard", index=False)
                
                # Append granular timeline delivery traces for each qualified player
                for player in summary['batsman'].unique():
                    p_df = df[df['batsman'] == player].sort_values(by=['match_id', 'innings', 'over'])
                    export_cols = [
                        'match_id', 'bowling_team', 'bowler', 'bowler_rank', 'innings',
                        'balls_remaining', 'wickets_lost', 'crr', 'rrr', 
                        'action', 'runs_batter', 'is_out', 'cbi_probability'
                    ]
                    # Format names within Excel sheet text boundaries safely
                    safe_name = "".join([c for c in player[:25] if c.isalnum() or c in (' ', '_')]).strip()
                    p_df[export_cols].to_excel(writer, sheet_name=f"Trace_{safe_name}", index=False)
                    
            logging.info(f"Workbook compiled perfectly. Output successfully targeted to -> '{output_path}'")
        except ImportError:
            logging.error("Missing file writer engine. Execute command in console: 'pip install openpyxl'")


# ==============================================================================
# MODULE 5: RUNTIME INITIALIZATION ENGINE
# ==============================================================================
if __name__ == "__main__":
    DATA_PATH = "t20_json_data/mens_t20_world_cup_ball_by_ball.csv"
    OUTPUT_PATH = "T20_WC_CBI_vs_Traditional_Rankings.xlsx"
    
    logging.info("Initializing Contextual Batting Intelligence Analytics Framework...")
    
    pipeline = CricketDataPipeline(DATA_PATH)
    structured_data = pipeline.process_dataset()
    
    if not structured_data.empty:
        engine = CBIEngine()
        computed_dataset = engine.compute_policy_alignment(structured_data)
        
        ExcelReportGenerator.construct_workbook(computed_dataset, OUTPUT_PATH)
    else:
        logging.critical("Data ingest loop severed. Run terminal checklist checks.")
