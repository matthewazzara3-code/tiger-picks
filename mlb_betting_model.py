# =============================================================================
# MLB BETTING MODEL - v5 (Daily Auto-Run + Record Tracking)
# =============================================================================
# HOW TO RUN MANUALLY:
#   cd "C:\Users\Owner\OneDrive - University of Cincinnati\Attachments\files\MLB Model"
#   python mlb_betting_model.py
#
# HOW TO SET UP DAILY AUTO-RUN (Windows Task Scheduler):
#   1. Search "Task Scheduler" in your Start menu and open it
#   2. Click "Create Basic Task" on the right side
#   3. Name it "MLB Betting Model"
#   4. Set trigger: Daily, at whatever time you want (e.g. 11:00 AM)
#   5. Action: "Start a program"
#   6. Program: C:\Users\Owner\AppData\Local\Python\pythoncore-3.14-64\python.exe
#   7. Arguments: mlb_betting_model.py
#   8. Start in: C:\Users\Owner\OneDrive - University of Cincinnati\Attachments\files\MLB Model
#   9. Click Finish
#
# RECORD TRACKING:
#   - All predictions saved to: mlb_predictions_log.csv
#   - Daily summary saved to:   mlb_record_summary.csv
#   - Run mlb_betting_model.py --results to grade yesterday's picks
# =============================================================================

import pandas as pd
import numpy as np
import warnings
import os
import sys
import json
from datetime import datetime, timedelta, timezone
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import requests


# =============================================================================
# SETTINGS — edit these as needed
# =============================================================================
ODDS_API_KEY = "448d2c129dd35fcf3a4d1b05466a46ac"   # ← paste your API key here
SEASON       = 2025                   # current season for predictions
LOG_FILE     = "mlb_predictions_log.csv"     # where all picks are saved
SUMMARY_FILE = "mlb_record_summary.csv"      # win/loss record summary


# =============================================================================
# TEAM NAME MAPPING
# =============================================================================
ABBREV_TO_FULL = {
    'NYY': 'New York Yankees',     'BOS': 'Boston Red Sox',
    'TBR': 'Tampa Bay Rays',       'TOR': 'Toronto Blue Jays',
    'BAL': 'Baltimore Orioles',    'CLE': 'Cleveland Guardians',
    'MIN': 'Minnesota Twins',      'CHW': 'Chicago White Sox',
    'DET': 'Detroit Tigers',       'KCR': 'Kansas City Royals',
    'HOU': 'Houston Astros',       'LAA': 'Los Angeles Angels',
    'SEA': 'Seattle Mariners',     'TEX': 'Texas Rangers',
    'OAK': 'Athletics',            'ATL': 'Atlanta Braves',
    'NYM': 'New York Mets',        'PHI': 'Philadelphia Phillies',
    'MIA': 'Miami Marlins',        'WSN': 'Washington Nationals',
    'MIL': 'Milwaukee Brewers',    'CHC': 'Chicago Cubs',
    'STL': 'St. Louis Cardinals',  'PIT': 'Pittsburgh Pirates',
    'CIN': 'Cincinnati Reds',      'LAD': 'Los Angeles Dodgers',
    'SFG': 'San Francisco Giants', 'SDP': 'San Diego Padres',
    'COL': 'Colorado Rockies',     'ARI': 'Arizona Diamondbacks',
}


# =============================================================================
# STEP 1: LOAD CSV FILES
# =============================================================================
print("Loading CSV files...")

def load_bbref(filename):
    return pd.read_csv(filename, skiprows=1)

batting_2022  = load_bbref("batting_2022.csv")
batting_2023  = load_bbref("batting_2023.csv")
batting_2024  = load_bbref("batting_2024.csv")
batting_2025  = load_bbref("batting_2025.csv")
pitching_2022 = load_bbref("pitching_2022.csv")
pitching_2023 = load_bbref("pitching_2023.csv")
pitching_2024 = load_bbref("pitching_2024.csv")
pitching_2025 = load_bbref("pitching_2025.csv")

print("All 8 files loaded!\n")


# =============================================================================
# STEP 2: CLEAN DATA
# =============================================================================
def clean_batting(df, year):
    df = df.copy()
    df['Season'] = year
    df = df[df['Tm'].notna()]
    df = df[~df['Tm'].str.contains('Lg|Average|Total|League', na=True)]
    df = df.rename(columns={'Tm': 'Team'})
    for col in ['OBP', 'SLG', 'BA', 'R']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df['ISO'] = df['SLG'] - df['BA']
    return df[['Team','Season','R','OBP','SLG','BA','ISO']].dropna(subset=['Team'])

def clean_pitching(df, year):
    df = df.copy()
    df['Season'] = year
    df = df[df['Tm'].notna()]
    df = df[~df['Tm'].str.contains('Lg|Average|Total|League', na=True)]
    df = df.rename(columns={
        'Tm':'Team','ERA':'ERA_allowed','FIP':'FIP_allowed',
        'WHIP':'WHIP_allowed','SO9':'K9_allowed',
        'BB9':'BB9_allowed','HR9':'HR9_allowed','RA/G':'RA_per_game',
    })
    for col in ['ERA_allowed','FIP_allowed','WHIP_allowed',
                'K9_allowed','BB9_allowed','HR9_allowed','RA_per_game']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    keep = ['Team','Season'] + [c for c in
            ['ERA_allowed','FIP_allowed','WHIP_allowed',
             'K9_allowed','BB9_allowed','HR9_allowed','RA_per_game']
            if c in df.columns]
    return df[keep].dropna(subset=['Team'])

print("Cleaning data...")
all_batting = pd.concat([clean_batting(b,y) for b,y in [
    (batting_2022,2022),(batting_2023,2023),(batting_2024,2024),(batting_2025,2025)
]], ignore_index=True)
all_pitching = pd.concat([clean_pitching(p,y) for p,y in [
    (pitching_2022,2022),(pitching_2023,2023),(pitching_2024,2024),(pitching_2025,2025)
]], ignore_index=True)
all_teams = pd.merge(all_batting, all_pitching, on=['Team','Season'], how='inner')
print(f"Team-season records: {len(all_teams)}\n")


# =============================================================================
# STEP 3: LOAD HISTORICAL SCHEDULES & TRAIN MODELS
# =============================================================================
print("Loading historical game results (this takes 3-8 minutes)...")

def get_season_schedule(year):
    from pybaseball import schedule_and_record
    all_games = []
    for team in list(ABBREV_TO_FULL.keys()):
        try:
            sched = schedule_and_record(year, team)
            sched['Season'] = year
            for col in ['Home','Away','Tm','Opp']:
                if col in sched.columns:
                    sched[col] = sched[col].map(ABBREV_TO_FULL).fillna(sched[col])
            all_games.append(sched)
        except Exception:
            pass
    if not all_games:
        return pd.DataFrame()
    df = pd.concat(all_games, ignore_index=True)
    if 'Home' not in df.columns and 'Tm' in df.columns:
        df = df.rename(columns={'Tm':'Home','Opp':'Away'})
    for col in ['R','RA']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    if all(c in df.columns for c in ['Date','Home','Away']):
        df = df.drop_duplicates(subset=['Date','Home','Away'])
    return df

try:
    schedules = []
    for year in [2022, 2023, 2024, 2025]:
        print(f"  Fetching {year} schedule...")
        s = get_season_schedule(year)
        if not s.empty:
            schedules.append(s)
            print(f"  {year}: {len(s)} games loaded")
    all_schedules  = pd.concat(schedules, ignore_index=True) if schedules else pd.DataFrame()
    schedule_loaded = not all_schedules.empty
except Exception as e:
    print(f"Schedule loading failed: {e}")
    all_schedules  = pd.DataFrame()
    schedule_loaded = False

models_trained = False
ML_FEATS  = ['OBP_diff','ERA_diff','OBP_home','OBP_away','ERA_home','ERA_away','WHIP_home','WHIP_away']
TOT_FEATS = ['OBP_home','OBP_away','SLG_home','SLG_away','ERA_home','ERA_away','WHIP_home','WHIP_away']
SP_FEATS  = ['OBP_diff','ERA_diff','OBP_home','OBP_away','ERA_home','ERA_away']

if schedule_loaded and len(all_schedules) > 100:
    if all(c in all_schedules.columns for c in ['R','RA','Home']):
        print("\nBuilding matchup features...")
        rows = []
        for _, game in all_schedules.iterrows():
            home = game.get('Home'); away = game.get('Away'); season = game.get('Season',2024)
            hs  = all_teams[(all_teams['Team']==home)&(all_teams['Season']==season)]
            as_ = all_teams[(all_teams['Team']==away)&(all_teams['Season']==season)]
            if hs.empty or as_.empty: continue
            h = hs.iloc[0]; a = as_.iloc[0]
            hr = pd.to_numeric(game.get('R', np.nan), errors='coerce')
            ar = pd.to_numeric(game.get('RA',np.nan), errors='coerce')
            if np.isnan(hr) or np.isnan(ar): continue
            rows.append({
                'OBP_home':h.get('OBP',.320),'OBP_away':a.get('OBP',.320),
                'SLG_home':h.get('SLG',.400),'SLG_away':a.get('SLG',.400),
                'ERA_home':h.get('ERA_allowed',4.0),'ERA_away':a.get('ERA_allowed',4.0),
                'WHIP_home':h.get('WHIP_allowed',1.3),'WHIP_away':a.get('WHIP_allowed',1.3),
                'OBP_diff':h.get('OBP',.320)-a.get('OBP',.320),
                'ERA_diff':h.get('ERA_allowed',4.0)-a.get('ERA_allowed',4.0),
                'home_win':1 if hr>ar else 0,'run_diff':hr-ar,'total_runs':hr+ar,
            })
        games_df = pd.DataFrame(rows).dropna()
        print(f"Game rows for training: {len(games_df)}")

        if len(games_df) > 100:
            def train(X, y, model):
                Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.2,random_state=42)
                sc = StandardScaler()
                model.fit(sc.fit_transform(Xtr), ytr)
                return model, sc, Xte, yte

            print("\n" + "="*60)
            print("MODEL 1: MONEYLINE")
            print("="*60)
            ml_model,scaler_ml,Xte,yte = train(games_df[ML_FEATS],games_df['home_win'],LogisticRegression(random_state=42))
            print(f"Accuracy: {accuracy_score(yte, ml_model.predict(scaler_ml.transform(Xte))):.1%}")

            print("\n" + "="*60)
            print("MODEL 2: TOTAL RUNS")
            print("="*60)
            tot_model,scaler_tot,Xte_t,yte_t = train(games_df[TOT_FEATS],games_df['total_runs'],LinearRegression())
            print(f"MAE: {np.mean(np.abs(tot_model.predict(scaler_tot.transform(Xte_t))-yte_t)):.2f} runs")

            print("\n" + "="*60)
            print("MODEL 3: RUN SPREAD")
            print("="*60)
            sp_model,scaler_sp,Xte_s,yte_s = train(games_df[SP_FEATS],games_df['run_diff'],LinearRegression())
            print(f"MAE: {np.mean(np.abs(sp_model.predict(scaler_sp.transform(Xte_s))-yte_s)):.2f} runs\n")
            models_trained = True


# =============================================================================
# STEP 4: RECORD TRACKING FUNCTIONS
# =============================================================================

def load_log():
    """Loads the predictions log CSV, or creates an empty one."""
    cols = [
        'Date','Home','Away',
        'Model_Home_WinPct','Model_Fair_ML','DK_Home_ML','DK_Away_ML','ML_Pick','ML_Pick_Odds',
        'Model_Total','DK_Total','Total_Pick',
        'Model_Spread','DK_Spread','Spread_Pick',
        'Actual_Home_Runs','Actual_Away_Runs',
        'ML_Result','Total_Result','Spread_Result'
    ]
    if os.path.exists(LOG_FILE):
        df = pd.read_csv(LOG_FILE)
        for c in cols:
            if c not in df.columns:
                df[c] = np.nan
        return df
    return pd.DataFrame(columns=cols)


def save_prediction(log_df, date, home, away,
                    win_prob, fair_ml, dk_home_ml, dk_away_ml,
                    pred_total, dk_total, pred_spread, dk_spread):
    """Saves today's prediction to the log file."""

    # Determine picks (only log when there's an edge)
    ml_pick      = None
    ml_pick_odds = None
    if dk_home_ml is not None and dk_away_ml is not None:
        if win_prob >= 0.5 and dk_home_ml > fair_ml:
            ml_pick = home; ml_pick_odds = dk_home_ml
        elif win_prob < 0.5 and dk_away_ml is not None:
            away_win_prob = 1 - win_prob
            fair_away_ml  = round(-(away_win_prob/(1-away_win_prob))*100) if away_win_prob >= 0.5 else round(((1-away_win_prob)/away_win_prob)*100)
            if dk_away_ml > fair_away_ml:
                ml_pick = away; ml_pick_odds = dk_away_ml

    total_pick = None
    if dk_total is not None:
        diff = pred_total - dk_total
        if diff >= 0.5:   total_pick = f"OVER {dk_total}"
        elif diff <= -0.5: total_pick = f"UNDER {dk_total}"

    spread_pick = None
    if dk_spread is not None:
        if pred_spread > 0 and dk_spread <= -1.5:
            spread_pick = f"{home} {dk_spread}"
        elif pred_spread < 0 and dk_spread >= 1.5:
            spread_pick = f"{away} +{abs(dk_spread)}"

    new_row = {
        'Date': date, 'Home': home, 'Away': away,
        'Model_Home_WinPct': round(win_prob, 4),
        'Model_Fair_ML':     fair_ml,
        'DK_Home_ML':        dk_home_ml,
        'DK_Away_ML':        dk_away_ml,
        'ML_Pick':           ml_pick,
        'ML_Pick_Odds':      ml_pick_odds,
        'Model_Total':       round(pred_total, 2),
        'DK_Total':          dk_total,
        'Total_Pick':        total_pick,
        'Model_Spread':      round(pred_spread, 2),
        'DK_Spread':         dk_spread,
        'Spread_Pick':       spread_pick,
        'Actual_Home_Runs':  np.nan,
        'Actual_Away_Runs':  np.nan,
        'ML_Result':         'PENDING',
        'Total_Result':      'PENDING',
        'Spread_Result':     'PENDING',
    }

    # Don't duplicate — skip if this game is already logged for today
    existing = log_df[
        (log_df['Date'] == date) &
        (log_df['Home'] == home) &
        (log_df['Away'] == away)
    ]
    if existing.empty:
        log_df = pd.concat([log_df, pd.DataFrame([new_row])], ignore_index=True)

    return log_df


def grade_previous_games(log_df):
    """
    Looks up yesterday's game scores and grades all PENDING picks.
    Grades: WIN, LOSS, PUSH, or NO_PICK (if no edge was found).
    """
    print("\n" + "="*60)
    print("GRADING YESTERDAY'S PICKS")
    print("="*60)

    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    pending   = log_df[
        (log_df['Date'] == yesterday) &
        (log_df['ML_Result'] == 'PENDING')
    ]

    if pending.empty:
        print(f"No pending picks found for {yesterday}.\n")
        return log_df

    print(f"Fetching scores for {yesterday}...")

    try:
        # Fetch yesterday's scores from the Odds API results endpoint
        r = requests.get(
            "https://api.the-odds-api.com/v4/sports/baseball_mlb/scores/",
            params={
                'apiKey':        ODDS_API_KEY,
                'daysFrom':      1,
                'dateFormat':    'iso',
            },
            timeout=10
        )

        if r.status_code != 200:
            print(f"Could not fetch scores: {r.status_code}")
            return log_df

        scores = {}
        for event in r.json():
            if event.get('completed'):
                home = event.get('home_team','')
                away = event.get('away_team','')
                home_score = None
                away_score = None
                for team_score in event.get('scores') or []:
                    if team_score['name'] == home:
                        home_score = int(team_score['score'])
                    elif team_score['name'] == away:
                        away_score = int(team_score['score'])
                if home_score is not None and away_score is not None:
                    scores[(home, away)] = (home_score, away_score)

        graded = 0
        for idx, row in pending.iterrows():
            home = row['Home']
            away = row['Away']
            result = scores.get((home, away))

            if result is None:
                continue

            home_runs, away_runs = result
            log_df.at[idx, 'Actual_Home_Runs'] = home_runs
            log_df.at[idx, 'Actual_Away_Runs'] = away_runs

            # Grade moneyline
            ml_pick = row.get('ML_Pick')
            if pd.isna(ml_pick) or ml_pick is None:
                log_df.at[idx, 'ML_Result'] = 'NO_PICK'
            else:
                actual_winner = home if home_runs > away_runs else away
                log_df.at[idx, 'ML_Result'] = 'WIN' if ml_pick == actual_winner else 'LOSS'

            # Grade total
            total_pick = row.get('Total_Pick')
            dk_total   = row.get('DK_Total')
            if pd.isna(total_pick) or total_pick is None or pd.isna(dk_total):
                log_df.at[idx, 'Total_Result'] = 'NO_PICK'
            else:
                actual_total = home_runs + away_runs
                if actual_total == dk_total:
                    log_df.at[idx, 'Total_Result'] = 'PUSH'
                elif 'OVER' in str(total_pick) and actual_total > dk_total:
                    log_df.at[idx, 'Total_Result'] = 'WIN'
                elif 'UNDER' in str(total_pick) and actual_total < dk_total:
                    log_df.at[idx, 'Total_Result'] = 'WIN'
                else:
                    log_df.at[idx, 'Total_Result'] = 'LOSS'

            # Grade spread
            spread_pick = row.get('Spread_Pick')
            dk_spread   = row.get('DK_Spread')
            if pd.isna(spread_pick) or spread_pick is None or pd.isna(dk_spread):
                log_df.at[idx, 'Spread_Result'] = 'NO_PICK'
            else:
                actual_margin = home_runs - away_runs
                if home in str(spread_pick):
                    covered = actual_margin > abs(dk_spread) if dk_spread < 0 else actual_margin > dk_spread
                else:
                    covered = (away_runs - home_runs) > abs(dk_spread)
                if actual_margin == abs(dk_spread):
                    log_df.at[idx, 'Spread_Result'] = 'PUSH'
                else:
                    log_df.at[idx, 'Spread_Result'] = 'WIN' if covered else 'LOSS'

            graded += 1
            print(f"  {away} @ {home}: {away_runs}-{home_runs}  "
                  f"ML:{log_df.at[idx,'ML_Result']}  "
                  f"Total:{log_df.at[idx,'Total_Result']}  "
                  f"Spread:{log_df.at[idx,'Spread_Result']}")

        print(f"\nGraded {graded} games.")

    except Exception as e:
        print(f"Error fetching scores: {e}")

    return log_df


def print_record(log_df):
    """Prints the overall win/loss record for all three markets."""
    print("\n" + "="*60)
    print("OVERALL MODEL RECORD")
    print("="*60)

    for market, col in [('MONEYLINE','ML_Result'),
                        ('TOTAL RUNS','Total_Result'),
                        ('RUN SPREAD','Spread_Result')]:
        results = log_df[log_df[col].isin(['WIN','LOSS','PUSH'])]
        wins    = len(results[results[col]=='WIN'])
        losses  = len(results[results[col]=='LOSS'])
        pushes  = len(results[results[col]=='PUSH'])
        no_pick = len(log_df[log_df[col]=='NO_PICK'])
        total   = wins + losses + pushes

        if total > 0:
            pct = wins / (wins + losses) * 100 if (wins+losses) > 0 else 0
            print(f"\n  {market}")
            print(f"  Record   : {wins}W - {losses}L - {pushes}P  ({pct:.1f}%)")
            print(f"  No pick  : {no_pick} games (no edge found)")
        else:
            print(f"\n  {market}: No graded picks yet")

    # Save updated summary
    summary = {
        'Last_Updated': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'ML_Wins':   len(log_df[log_df['ML_Result']=='WIN']),
        'ML_Losses': len(log_df[log_df['ML_Result']=='LOSS']),
        'ML_Pushes': len(log_df[log_df['ML_Result']=='PUSH']),
        'Tot_Wins':  len(log_df[log_df['Total_Result']=='WIN']),
        'Tot_Losses':len(log_df[log_df['Total_Result']=='LOSS']),
        'Tot_Pushes':len(log_df[log_df['Total_Result']=='PUSH']),
        'Sp_Wins':   len(log_df[log_df['Spread_Result']=='WIN']),
        'Sp_Losses': len(log_df[log_df['Spread_Result']=='LOSS']),
        'Sp_Pushes': len(log_df[log_df['Spread_Result']=='PUSH']),
    }
    pd.DataFrame([summary]).to_csv(SUMMARY_FILE, index=False)
    print(f"\n  Full log saved to: {LOG_FILE}")
    print(f"  Summary saved to:  {SUMMARY_FILE}")


# =============================================================================
# STEP 5: FETCH TODAY'S DRAFTKINGS ODDS
# =============================================================================

def fetch_todays_games(api_key):
    if api_key == "448d2c129dd35fcf3a4d1b05466a46ac":
        print("No API key set. Get a free key at https://the-odds-api.com\n")
        return []
    try:
        r = requests.get(
            "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds/",
            params={
                'apiKey':api_key,'regions':'us',
                'markets':'h2h,spreads,totals',
                'bookmakers':'draftkings','oddsFormat':'american'
            },
            timeout=10
        )
        if r.status_code != 200:
            print(f"Odds API error: {r.status_code}\n"); return []

        games = []
        for event in r.json():
            game = {
                'home':event.get('home_team',''),'away':event.get('away_team',''),
                'dk_home_ml':None,'dk_away_ml':None,'dk_spread':None,'dk_total':None,
            }
            for bm in event.get('bookmakers',[]):
                for mkt in bm.get('markets',[]):
                    k = mkt.get('key','')
                    if k=='h2h':
                        for o in mkt.get('outcomes',[]):
                            if o['name']==game['home']:   game['dk_home_ml']=o['price']
                            elif o['name']==game['away']: game['dk_away_ml']=o['price']
                    elif k=='spreads':
                        for o in mkt.get('outcomes',[]):
                            if o['name']==game['home']:   game['dk_spread']=o.get('point')
                    elif k=='totals':
                        for o in mkt.get('outcomes',[]):
                            if o.get('name')=='Over':     game['dk_total']=o.get('point')
            games.append(game)
        return games
    except Exception as e:
        print(f"Could not fetch odds: {e}\n"); return []


# =============================================================================
# STEP 6: PREDICT A SINGLE GAME
# =============================================================================

def predict_game(home_input, away_input, dk_home_ml=None,
                 dk_away_ml=None, dk_total=None, dk_spread=None,
                 season=2025, log_df=None, date=None):

    home = ABBREV_TO_FULL.get(home_input, home_input)
    away = ABBREV_TO_FULL.get(away_input, away_input)

    hs  = all_teams[(all_teams['Team']==home)&(all_teams['Season']==season)]
    as_ = all_teams[(all_teams['Team']==away)&(all_teams['Season']==season)]
    if hs.empty or as_.empty: return log_df

    h = hs.iloc[0]; a = as_.iloc[0]

    if not models_trained:
        print(f"\n  {away} @ {home}")
        print(f"  OBP edge: {h['OBP']-a['OBP']:+.3f} | ERA edge: {a['ERA_allowed']-h['ERA_allowed']:+.2f}")
        return log_df

    feats = {
        'OBP_diff':h['OBP']-a['OBP'],'ERA_diff':h['ERA_allowed']-a['ERA_allowed'],
        'OBP_home':h['OBP'],'OBP_away':a['OBP'],
        'SLG_home':h['SLG'],'SLG_away':a['SLG'],
        'ERA_home':h['ERA_allowed'],'ERA_away':a['ERA_allowed'],
        'WHIP_home':h['WHIP_allowed'],'WHIP_away':a['WHIP_allowed'],
    }

    win_prob    = ml_model.predict_proba(scaler_ml.transform([[feats[f] for f in ML_FEATS]]))[0][1]
    pred_total  = tot_model.predict(scaler_tot.transform([[feats[f] for f in TOT_FEATS]]))[0]
    pred_spread = sp_model.predict(scaler_sp.transform([[feats[f] for f in SP_FEATS]]))[0]
    fair_ml     = round(-(win_prob/(1-win_prob))*100) if win_prob>=0.5 else round(((1-win_prob)/win_prob)*100)

    print(f"\n{'='*60}")
    print(f"  {away[:26]} @ {home[:26]}")
    print(f"{'='*60}")

    print(f"\n  MONEYLINE")
    print(f"  Model fair line  : {fair_ml:+d}  ({win_prob:.1%} home win)")
    if dk_home_ml is not None:
        print(f"  DraftKings       : Home {dk_home_ml:+d} | Away {dk_away_ml:+d}")
        if win_prob >= 0.5 and dk_home_ml > fair_ml:
            print(f"  ✅ VALUE — {home[:22]} is underpriced")
        elif win_prob < 0.5:
            away_wp = 1 - win_prob
            fair_away = round(-(away_wp/(1-away_wp))*100) if away_wp>=0.5 else round(((1-away_wp)/away_wp)*100)
            if dk_away_ml and dk_away_ml > fair_away:
                print(f"  ✅ VALUE — {away[:22]} (away) is underpriced")
            else:
                print(f"  ❌ No moneyline value")
        else:
            print(f"  ❌ No moneyline value")

    print(f"\n  TOTAL RUNS")
    print(f"  Model predicts   : {pred_total:.1f} runs")
    if dk_total is not None:
        print(f"  DraftKings line  : {dk_total}")
        diff = pred_total - dk_total
        if diff >= 0.5:    print(f"  ✅ Lean OVER {dk_total}")
        elif diff <= -0.5: print(f"  ✅ Lean UNDER {dk_total}")
        else:              print(f"  ➖ Too close to call")

    print(f"\n  RUN SPREAD")
    print(f"  Model margin     : {pred_spread:+.1f} runs")
    if dk_spread is not None:
        print(f"  DraftKings spread: {dk_spread:+.1f}")
        if pred_spread > 0 and dk_spread <= -1.5:   print(f"  ✅ Home covers {dk_spread}")
        elif pred_spread < 0 and dk_spread >= 1.5:  print(f"  ✅ Away covers")
        else:                                        print(f"  ➖ Too close to call")

    print(f"\n  {home[:22]:<24} OBP:{h['OBP']:.3f} SLG:{h['SLG']:.3f} ERA:{h['ERA_allowed']:.2f} WHIP:{h['WHIP_allowed']:.2f}")
    print(f"  {away[:22]:<24} OBP:{a['OBP']:.3f} SLG:{a['SLG']:.3f} ERA:{a['ERA_allowed']:.2f} WHIP:{a['WHIP_allowed']:.2f}")

    # Save to log
    if log_df is not None and date is not None:
        log_df = save_prediction(
            log_df, date, home, away,
            win_prob, fair_ml, dk_home_ml, dk_away_ml,
            pred_total, dk_total, pred_spread, dk_spread
        )

    return log_df


# =============================================================================
# STEP 7: MAIN — RUN TODAY'S GAMES + GRADE YESTERDAY
# =============================================================================
today     = datetime.now().strftime('%Y-%m-%d')
log_df    = load_log()

# Grade yesterday's picks first
log_df = grade_previous_games(log_df)

# Print running record
print_record(log_df)

# Fetch and predict today's games
print("\n" + "="*60)
print(f"TODAY'S MLB GAMES ({today}) — MODEL vs DRAFTKINGS")
print("="*60)

todays_games = fetch_todays_games(ODDS_API_KEY)

if todays_games and models_trained:
    print(f"Found {len(todays_games)} games on DraftKings today.\n")
    for game in todays_games:
        log_df = predict_game(
            home_input = game['home'],  away_input = game['away'],
            dk_home_ml = game['dk_home_ml'], dk_away_ml = game['dk_away_ml'],
            dk_total   = game['dk_total'],   dk_spread  = game['dk_spread'],
            log_df=log_df, date=today
        )
elif not todays_games:
    print("No games found or API key not set. Running manual predictions.\n")
    for matchup in [('NYY','BOS'),('LAD','ATL'),('HOU','CHC')]:
        log_df = predict_game(matchup[0], matchup[1], log_df=log_df, date=today)

# Save updated log
log_df.to_csv(LOG_FILE, index=False)
print(f"\n✅ Predictions saved to {LOG_FILE}")
print(f"✅ Run again tomorrow to auto-grade today's picks!\n")

print("ALL TEAM ABBREVIATIONS:")
for abbrev, full in sorted(ABBREV_TO_FULL.items(), key=lambda x: x[1]):
    print(f"  {abbrev} = {full}")
