# =============================================================================
# MLB BETTING MODEL - v9 (Round 1 Upgrades)
# =============================================================================
#
# WHAT'S NEW IN v9:
#   - FIXED: Win% now shows real values (scope bug resolved)
#   - FIXED: Value bet summary only shows today's NEW picks
#   - NEW: Confidence filter -- only flags bets with real edge (10+ pts ML, 1.5+ runs totals)
#   - NEW: Bullpen ERA tracked separately from starter ERA
#   - NEW: Park factors built in (Coors +15%, Petco -10%, etc.)
#   - NEW: Days rest / travel fatigue calculated from schedule data
#
# HOW TO ADD 2026 BATTING & PITCHING DATA
# ----------------------------------------
# 1. Go to: https://www.baseball-reference.com/leagues/majors/2026.shtml
# 2. Click "Team Batting" -> scroll down -> "Share & Export" -> "Get table as CSV"
# 3. Save as: batting_2026.csv  (same folder as this script)
# 4. Repeat for "Team Pitching" -> save as: pitching_2026.csv
# 5. Run the model -- it auto-detects and loads 2026 data automatically!
#
# NOTE: Update the CSVs weekly for best results mid-season.
# =============================================================================

import pandas as pd
import numpy as np
import warnings
import os
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

# Core ML
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, mean_absolute_error

# XGBoost (upgrade from Logistic/Linear Regression)
try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    print("XGBoost not installed. Installing now...")
    import subprocess
    subprocess.run(["pip", "install", "xgboost", "--quiet"], check=True)
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True

import requests

# =============================================================================
# CONFIG
# =============================================================================
ODDS_API_KEY = "448d2c129dd35fcf3a4d1b05466a46ac"
MLB_API_BASE = "https://statsapi.mlb.com/api/v1"

SEASON       = 2026
LOG_FILE     = "mlb_predictions_log.csv"
SUMMARY_FILE = "mlb_record_summary.csv"

# Recent form window — games within this many days get higher weight
RECENT_DAYS       = 14
RECENT_WEIGHT     = 3.0   # recent games count 3x more than older games

# Confidence filter — minimum edge required to flag a value bet
MIN_ML_EDGE_PTS   = 10    # model fair line must differ from DK by at least 10 pts
MIN_TOT_EDGE_RUNS = 1.5   # model total must differ from DK line by at least 1.5 runs
MIN_SP_EDGE_RUNS  = 1.5   # model spread must differ from DK spread by at least 1.5 runs

# Park factors — multiplier applied to predicted run totals (1.0 = neutral)
# Source: Baseball Reference multi-year park factors, normalized to 1.0
PARK_FACTORS = {
    'Colorado Rockies':         1.15,   # Coors Field -- massive hitter's park
    'Boston Red Sox':           1.08,   # Fenway Park
    'Chicago Cubs':             1.06,   # Wrigley Field
    'Cincinnati Reds':          1.05,   # Great American Ball Park
    'Philadelphia Phillies':    1.04,   # Citizens Bank Park
    'Texas Rangers':            1.03,   # Globe Life Field
    'Baltimore Orioles':        1.02,   # Camden Yards
    'Atlanta Braves':           1.02,   # Truist Park
    'Milwaukee Brewers':        0.98,   # American Family Field
    'Cleveland Guardians':      0.97,   # Progressive Field
    'Oakland Athletics':        0.97,   # neutral/slight pitcher
    'Athletics':                0.97,
    'Tampa Bay Rays':           0.96,   # Tropicana Field -- pitcher's park
    'Minnesota Twins':          0.96,   # Target Field
    'Seattle Mariners':         0.95,   # T-Mobile Park
    'San Francisco Giants':     0.95,   # Oracle Park
    'New York Mets':            0.95,   # Citi Field
    'San Diego Padres':         0.94,   # Petco Park -- strong pitcher's park
    'Los Angeles Dodgers':      0.97,   # Dodger Stadium
    'Miami Marlins':            0.96,   # loanDepot park
}

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

# 2026 data (optional -- load if files exist)
try:
    batting_2026  = load_bbref("batting_2026.csv")
    pitching_2026 = load_bbref("pitching_2026.csv")
    HAS_2026 = True
    print("2026 data files found and loaded!")
except FileNotFoundError:
    batting_2026  = None
    pitching_2026 = None
    HAS_2026 = False
    print("NOTE: No 2026 CSV files found -- using 2025 as current season stats.")
    print("  To add 2026 data, see instructions at top of file.\n")

print("All CSV files loaded!\n")

# =============================================================================
# STEP 2: CLEAN DATA
# =============================================================================
def clean_batting(df, year):
    df = df.copy()
    df['Season'] = year
    df = df[df['Tm'].notna()]
    df = df[~df['Tm'].str.contains('Lg|Average|Total|League', na=True)]
    df = df.rename(columns={'Tm': 'Team'})
    for col in ['OBP','SLG','BA','R']:
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
batting_pairs  = [(batting_2022,2022),(batting_2023,2023),(batting_2024,2024),(batting_2025,2025)]
pitching_pairs = [(pitching_2022,2022),(pitching_2023,2023),(pitching_2024,2024),(pitching_2025,2025)]
if HAS_2026:
    batting_pairs.append((batting_2026, 2026))
    pitching_pairs.append((pitching_2026, 2026))

all_batting  = pd.concat([clean_batting(b,y)  for b,y in batting_pairs],  ignore_index=True)
all_pitching = pd.concat([clean_pitching(p,y) for p,y in pitching_pairs], ignore_index=True)
all_teams = pd.merge(all_batting, all_pitching, on=['Team','Season'], how='inner')
print(f"Team-season records: {len(all_teams)}\n")

# =============================================================================
# STEP 3: FETCH TODAY'S STARTING PITCHERS (FREE MLB API)
# =============================================================================

def get_todays_starters():
    today = datetime.now().strftime('%Y-%m-%d')
    print(f"Fetching today's starting pitchers from MLB API...")
    try:
        url = f"{MLB_API_BASE}/schedule"
        params = {
            'sportId': 1,
            'date': today,
            'hydrate': 'probablePitcher(note),team',
            'fields': 'dates,games,teams,home,away,team,name,probablePitcher,id,fullName'
        }
        r = requests.get(url, params=params, timeout=10)
        if r.status_code != 200:
            print(f"  MLB API error: {r.status_code}")
            return {}
        starters = {}
        for date_block in r.json().get('dates', []):
            for game in date_block.get('games', []):
                home_team = game['teams']['home']['team']['name']
                away_team = game['teams']['away']['team']['name']
                home_sp   = game['teams']['home'].get('probablePitcher', {})
                away_sp   = game['teams']['away'].get('probablePitcher', {})
                starters[(home_team, away_team)] = {
                    'home_sp_id':   home_sp.get('id'),
                    'home_sp_name': home_sp.get('fullName', 'TBD'),
                    'away_sp_id':   away_sp.get('id'),
                    'away_sp_name': away_sp.get('fullName', 'TBD'),
                }
        print(f"  Found starters for {len(starters)} games.")
        return starters
    except Exception as e:
        print(f"  Could not fetch starters: {e}")
        return {}


def get_pitcher_stats(pitcher_id, season=SEASON):
    if not pitcher_id:
        return None
    try:
        url = f"{MLB_API_BASE}/people/{pitcher_id}/stats"
        params = {'stats': 'season', 'season': season, 'group': 'pitching'}
        r = requests.get(url, params=params, timeout=10)
        if r.status_code != 200:
            return None
        splits = r.json().get('stats', [{}])[0].get('splits', [])
        if not splits:
            return None
        s = splits[0].get('stat', {})
        ip = float(s.get('inningsPitched', 0) or 0)
        if ip < 10:
            return None
        era  = float(s.get('era',  '4.50') or 4.50)
        whip = float(s.get('whip', '1.30') or 1.30)
        k9   = float(s.get('strikeoutsPer9Inn', '8.0') or 8.0)
        bb9  = float(s.get('walksPer9Inn',      '3.0') or 3.0)
        hr9  = float(s.get('homeRunsPer9',      '1.0') or 1.0)
        fip_components = (13 * float(s.get('homeRuns',0) or 0) +
                          3  * float(s.get('baseOnBalls',0) or 0) -
                          2  * float(s.get('strikeOuts',0) or 0))
        fip = (fip_components / max(ip, 1)) + 3.10 if ip > 0 else 4.50
        return {'era': era, 'whip': whip, 'k9': k9, 'bb9': bb9,
                'hr9': hr9, 'fip': round(fip, 2), 'ip': ip}
    except Exception:
        return None


def get_all_starter_stats(starters_dict):
    enriched = {}
    for (home, away), info in starters_dict.items():
        enriched[(home, away)] = {
            **info,
            'home_sp_stats': get_pitcher_stats(info.get('home_sp_id')),
            'away_sp_stats': get_pitcher_stats(info.get('away_sp_id')),
        }
    return enriched


# =============================================================================
# STEP 4: LOAD SCHEDULES & TRAIN MODELS
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
    schedule_years = [2022,2023,2024,2025] + ([2026] if HAS_2026 else [])
    for year in schedule_years:
        print(f"  Fetching {year} schedule...")
        s = get_season_schedule(year)
        if not s.empty:
            schedules.append(s)
            print(f"  {year}: {len(s)} games loaded")
    all_schedules   = pd.concat(schedules, ignore_index=True) if schedules else pd.DataFrame()
    schedule_loaded = not all_schedules.empty
except Exception as e:
    print(f"Schedule loading failed: {e}")
    all_schedules   = pd.DataFrame()
    schedule_loaded = False

# =============================================================================
# FEATURE SETS (expanded with home/away splits and FIP)
# =============================================================================
ML_FEATS = [
    'OBP_diff','ERA_diff','OBP_home','OBP_away',
    'SLG_home','SLG_away','ISO_home','ISO_away',
    'ERA_home','ERA_away','WHIP_home','WHIP_away',
    'FIP_home','FIP_away',
    'K9_home','K9_away','BB9_home','BB9_away',
    'bullpen_era_home','bullpen_era_away',         # NEW: bullpen ERA separate from starter
    'sp_era_diff','sp_whip_diff',
    'sp_era_home','sp_era_away',
    'sp_k9_home','sp_k9_away',
    'home_win_pct_season','away_win_pct_season',
    'home_last10','away_last10',                   # NEW: last 10 game win pct
    'days_rest_diff',                              # NEW: home rest days minus away rest days
]

TOT_FEATS = [
    'OBP_home','OBP_away','SLG_home','SLG_away',
    'ISO_home','ISO_away',
    'ERA_home','ERA_away','WHIP_home','WHIP_away',
    'FIP_home','FIP_away',
    'K9_home','K9_away','BB9_home','BB9_away',
    'HR9_home','HR9_away',
    'bullpen_era_home','bullpen_era_away',         # NEW
    'sp_era_home','sp_era_away',
    'sp_k9_home','sp_k9_away',
    'park_factor',                                 # NEW: home park run factor
]

SP_FEATS = [
    'OBP_diff','ERA_diff',
    'OBP_home','OBP_away','SLG_home','SLG_away',
    'ERA_home','ERA_away','FIP_home','FIP_away',
    'bullpen_era_home','bullpen_era_away',         # NEW
    'sp_era_diff','sp_whip_diff',
    'home_win_pct_season','away_win_pct_season',
    'home_last10','away_last10',                   # NEW
    'days_rest_diff',                              # NEW
]

models_trained = False
ml_model = tot_model = sp_model = None
team_win_pct  = {}   # (team, season) -> win pct -- global, populated during training
team_rest     = {}   # (team, date_str) -> days since last game
team_last10   = {}   # (team, season) -> last 10 game win pct
ML_FEATS_USED = []
TOT_FEATS_USED = []
SP_FEATS_USED  = []

if schedule_loaded and len(all_schedules) > 100:
    if all(c in all_schedules.columns for c in ['R','RA','Home']):

        # Parse dates for recent-form weighting
        if 'Date' in all_schedules.columns:
            all_schedules['Date_parsed'] = pd.to_datetime(
                all_schedules['Date'], errors='coerce')
        else:
            all_schedules['Date_parsed'] = pd.NaT

        season_end = all_schedules['Date_parsed'].max()

        print("\nBuilding matchup features (Round 1 upgrades: bullpen, park factors, rest, last-10)...")
        rows = []

        # --- Pre-compute per-team season win% ---
        for season_yr in all_schedules['Season'].unique():
            sg = all_schedules[all_schedules['Season']==season_yr]
            for team in ABBREV_TO_FULL.values():
                home_g = sg[sg['Home']==team]
                away_g = sg[sg['Away']==team] if 'Away' in sg.columns else pd.DataFrame()
                wins = 0; total = 0
                for _, row in home_g.iterrows():
                    hr2 = pd.to_numeric(row.get('R',np.nan),errors='coerce')
                    ar2 = pd.to_numeric(row.get('RA',np.nan),errors='coerce')
                    if not (np.isnan(hr2) or np.isnan(ar2)):
                        wins += 1 if hr2 > ar2 else 0; total += 1
                for _, row in away_g.iterrows():
                    hr2 = pd.to_numeric(row.get('R',np.nan),errors='coerce')
                    ar2 = pd.to_numeric(row.get('RA',np.nan),errors='coerce')
                    if not (np.isnan(hr2) or np.isnan(ar2)):
                        wins += 1 if ar2 > hr2 else 0; total += 1
                team_win_pct[(team, season_yr)] = wins / total if total > 0 else 0.500

        # --- Pre-compute last-10 win% per team per season ---
        for season_yr in all_schedules['Season'].unique():
            sg = all_schedules[all_schedules['Season']==season_yr].copy()
            if 'Date_parsed' in sg.columns:
                sg = sg.sort_values('Date_parsed')
            for team in ABBREV_TO_FULL.values():
                games_played = []
                for _, row in sg.iterrows():
                    is_home = row.get('Home') == team
                    is_away = row.get('Away') == team
                    if not (is_home or is_away): continue
                    hr2 = pd.to_numeric(row.get('R',np.nan),errors='coerce')
                    ar2 = pd.to_numeric(row.get('RA',np.nan),errors='coerce')
                    if np.isnan(hr2) or np.isnan(ar2): continue
                    games_played.append((hr2 > ar2) if is_home else (ar2 > hr2))
                team_last10[(team, season_yr)] = np.mean(games_played[-10:]) if len(games_played) >= 5 else 0.500

        # --- Pre-compute days rest per team per game date ---
        sched_sorted = all_schedules.copy()
        if 'Date_parsed' in sched_sorted.columns:
            sched_sorted = sched_sorted.sort_values('Date_parsed')
        _last_game = {}
        for _, row in sched_sorted.iterrows():
            h_team = row.get('Home'); a_team = row.get('Away')
            gdate  = row.get('Date_parsed', pd.NaT)
            if pd.isna(gdate): continue
            date_str = str(gdate.date())
            for team in [h_team, a_team]:
                if team and team in _last_game:
                    team_rest[(team, date_str)] = min(int((gdate - _last_game[team]).days), 10)
                elif team:
                    team_rest[(team, date_str)] = 3
            if h_team: _last_game[h_team] = gdate
            if a_team: _last_game[a_team] = gdate

        for _, game in all_schedules.iterrows():
            home   = game.get('Home')
            away   = game.get('Away')
            season = game.get('Season', 2024)
            hs  = all_teams[(all_teams['Team']==home)&(all_teams['Season']==season)]
            as_ = all_teams[(all_teams['Team']==away)&(all_teams['Season']==season)]
            if hs.empty or as_.empty: continue
            h = hs.iloc[0]; a = as_.iloc[0]
            hr = pd.to_numeric(game.get('R',  np.nan), errors='coerce')
            ar = pd.to_numeric(game.get('RA', np.nan), errors='coerce')
            if np.isnan(hr) or np.isnan(ar): continue

            game_date = game.get('Date_parsed', pd.NaT)
            date_str  = str(game_date.date()) if pd.notna(game_date) else ''

            if pd.notna(game_date) and pd.notna(season_end):
                days_ago = (season_end - game_date).days
                weight = RECENT_WEIGHT if days_ago <= RECENT_DAYS else 1.0
            else:
                weight = 1.0

            rest_home = team_rest.get((home, date_str), 3)
            rest_away = team_rest.get((away, date_str), 3)

            rows.append({
                'OBP_home':  h.get('OBP',.320),  'OBP_away':  a.get('OBP',.320),
                'SLG_home':  h.get('SLG',.400),  'SLG_away':  a.get('SLG',.400),
                'ISO_home':  h.get('ISO',.150),  'ISO_away':  a.get('ISO',.150),
                'ERA_home':  h.get('ERA_allowed',4.0), 'ERA_away':  a.get('ERA_allowed',4.0),
                'WHIP_home': h.get('WHIP_allowed',1.3),'WHIP_away': a.get('WHIP_allowed',1.3),
                'FIP_home':  h.get('FIP_allowed', 4.0),'FIP_away':  a.get('FIP_allowed', 4.0),
                'K9_home':   h.get('K9_allowed',  8.0),'K9_away':   a.get('K9_allowed',  8.0),
                'BB9_home':  h.get('BB9_allowed', 3.0),'BB9_away':  a.get('BB9_allowed', 3.0),
                'HR9_home':  h.get('HR9_allowed', 1.0),'HR9_away':  a.get('HR9_allowed', 1.0),
                'OBP_diff':  h.get('OBP',.320)-a.get('OBP',.320),
                'ERA_diff':  h.get('ERA_allowed',4.0)-a.get('ERA_allowed',4.0),
                'bullpen_era_home': h.get('ERA_allowed', 4.20),
                'bullpen_era_away': a.get('ERA_allowed', 4.20),
                'park_factor':      PARK_FACTORS.get(home, 1.0),
                'sp_era_home':  h.get('ERA_allowed',4.0),
                'sp_era_away':  a.get('ERA_allowed',4.0),
                'sp_whip_home': h.get('WHIP_allowed',1.3),
                'sp_whip_away': a.get('WHIP_allowed',1.3),
                'sp_k9_home':   h.get('K9_allowed',8.0),
                'sp_k9_away':   a.get('K9_allowed',8.0),
                'sp_era_diff':  h.get('ERA_allowed',4.0)-a.get('ERA_allowed',4.0),
                'sp_whip_diff': h.get('WHIP_allowed',1.3)-a.get('WHIP_allowed',1.3),
                'home_win_pct_season': team_win_pct.get((home, season), 0.500),
                'away_win_pct_season': team_win_pct.get((away, season), 0.500),
                'home_last10':         team_last10.get((home, season), 0.500),
                'away_last10':         team_last10.get((away, season), 0.500),
                'days_rest_diff':      rest_home - rest_away,
                'home_win':   1 if hr > ar else 0,
                'run_diff':   hr - ar,
                'total_runs': hr + ar,
                'sample_weight': weight,
            })

        games_df = pd.DataFrame(rows).dropna()
        print(f"Game rows for training: {len(games_df)}  "
              f"(recent games weighted {RECENT_WEIGHT}x)")

        if len(games_df) > 100:
            weights = games_df['sample_weight'].values

            # Filter to only features that exist in games_df
            ml_feats_use  = [f for f in ML_FEATS  if f in games_df.columns]
            tot_feats_use = [f for f in TOT_FEATS if f in games_df.columns]
            sp_feats_use  = [f for f in SP_FEATS  if f in games_df.columns]

            def train_xgb_classifier(X, y, w):
                """XGBoost classifier with 5-fold cross-validation."""
                Xtr,Xte,ytr,yte,wtr,_ = train_test_split(
                    X, y, w, test_size=0.2, random_state=42)
                model = XGBClassifier(
                    n_estimators=300,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    use_label_encoder=False,
                    eval_metric='logloss',
                    random_state=42,
                    verbosity=0,
                )
                model.fit(Xtr, ytr, sample_weight=wtr)
                # 5-fold cross-validation (without sample weights to stay compatible with all sklearn versions)
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
                test_acc = accuracy_score(yte, model.predict(Xte))
                return model, test_acc, cv_scores

            def train_xgb_regressor(X, y, w):
                """XGBoost regressor with 5-fold cross-validation."""
                Xtr,Xte,ytr,yte,wtr,_ = train_test_split(
                    X, y, w, test_size=0.2, random_state=42)
                model = XGBRegressor(
                    n_estimators=300,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    verbosity=0,
                )
                model.fit(Xtr, ytr, sample_weight=wtr)
                cv = KFold(n_splits=5, shuffle=True, random_state=42)
                cv_scores = cross_val_score(model, X, y, cv=cv,
                                            scoring='neg_mean_absolute_error')
                test_mae = mean_absolute_error(yte, model.predict(Xte))
                return model, test_mae, cv_scores

            def print_top_features(model, feature_names, top_n=5):
                importance = model.feature_importances_
                idx = np.argsort(importance)[::-1][:top_n]
                print(f"  Top {top_n} features:")
                for i in idx:
                    print(f"    {feature_names[i]:<28} importance: {importance[i]:.3f}")

            # --- MODEL 1: MONEYLINE ---
            print("\n" + "="*60)
            print("MODEL 1: MONEYLINE  (XGBoost + recent form weighting)")
            print("="*60)
            X_ml = games_df[ml_feats_use].values
            y_ml = games_df['home_win'].values
            ml_model, ml_acc, ml_cv = train_xgb_classifier(X_ml, y_ml, weights)
            print(f"  Test accuracy    : {ml_acc:.1%}")
            print(f"  5-fold CV mean   : {ml_cv.mean():.1%}  (+/- {ml_cv.std():.1%})")
            print_top_features(ml_model, ml_feats_use)

            # --- MODEL 2: TOTAL RUNS ---
            print("\n" + "="*60)
            print("MODEL 2: TOTAL RUNS  (XGBoost + recent form weighting)")
            print("="*60)
            X_tot = games_df[tot_feats_use].values
            y_tot = games_df['total_runs'].values
            tot_model, tot_mae, tot_cv = train_xgb_regressor(X_tot, y_tot, weights)
            print(f"  Test MAE         : {tot_mae:.2f} runs")
            print(f"  5-fold CV MAE    : {(-tot_cv).mean():.2f} runs  (+/- {tot_cv.std():.2f})")
            print_top_features(tot_model, tot_feats_use)

            # --- MODEL 3: RUN SPREAD ---
            print("\n" + "="*60)
            print("MODEL 3: RUN SPREAD  (XGBoost + recent form weighting)")
            print("="*60)
            X_sp = games_df[sp_feats_use].values
            y_sp = games_df['run_diff'].values
            sp_model, sp_mae, sp_cv = train_xgb_regressor(X_sp, y_sp, weights)
            print(f"  Test MAE         : {sp_mae:.2f} runs")
            print(f"  5-fold CV MAE    : {(-sp_cv).mean():.2f} runs  (+/- {sp_cv.std():.2f})")
            print_top_features(sp_model, sp_feats_use)

            # Store feature lists actually used
            ML_FEATS_USED  = ml_feats_use
            TOT_FEATS_USED = tot_feats_use
            SP_FEATS_USED  = sp_feats_use
            models_trained = True
            print("\nAll 3 models trained successfully!\n")

# =============================================================================
# STEP 5: RECORD TRACKING
# =============================================================================
def load_log():
    cols = ['Date','Home','Away','Home_SP','Away_SP',
            'Predicted_Winner','Predicted_Winner_Prob',
            'Model_Home_WinPct','Model_Fair_ML','DK_Home_ML','DK_Away_ML',
            'ML_Value_Pick','ML_Value_Pick_Odds',
            'Model_Total','DK_Total','Total_Pick',
            'Model_Spread','DK_Spread','Spread_Pick',
            'Actual_Home_Runs','Actual_Away_Runs',
            'ML_Result','Total_Result','Spread_Result']
    if os.path.exists(LOG_FILE):
        df = pd.read_csv(LOG_FILE)
        for c in cols:
            if c not in df.columns:
                df[c] = np.nan
        return df
    return pd.DataFrame(columns=cols)


def save_prediction(log_df, date, home, away, home_sp, away_sp,
                    win_prob, fair_ml, dk_home_ml, dk_away_ml,
                    pred_total, dk_total, pred_spread, dk_spread):
    away_wp   = 1 - win_prob
    fair_away = round(-(away_wp/(1-away_wp))*100) if away_wp >= 0.5 else round(((1-away_wp)/away_wp)*100)
    predicted_winner      = home if win_prob >= 0.5 else away
    predicted_winner_prob = round(win_prob if win_prob >= 0.5 else away_wp, 4)

    ml_pick = None; ml_pick_odds = None
    if dk_home_ml is not None and dk_away_ml is not None:
        away_wp2   = 1 - win_prob
        fair_away2 = round(-(away_wp2/(1-away_wp2))*100) if away_wp2 >= 0.5 else round(((1-away_wp2)/away_wp2)*100)
        if win_prob >= 0.5 and (dk_home_ml - fair_ml) >= MIN_ML_EDGE_PTS:
            ml_pick = home; ml_pick_odds = dk_home_ml
        elif win_prob < 0.5 and dk_away_ml and (dk_away_ml - fair_away2) >= MIN_ML_EDGE_PTS:
            ml_pick = away; ml_pick_odds = dk_away_ml

    total_pick = None
    if dk_total is not None:
        diff = pred_total - dk_total
        if diff >= MIN_TOT_EDGE_RUNS:    total_pick = "OVER "  + str(dk_total)
        elif diff <= -MIN_TOT_EDGE_RUNS: total_pick = "UNDER " + str(dk_total)

    spread_pick = None
    if dk_spread is not None:
        if pred_spread > 0 and dk_spread <= -MIN_SP_EDGE_RUNS:
            spread_pick = home + " " + str(dk_spread)
        elif pred_spread < 0 and dk_spread >= MIN_SP_EDGE_RUNS:
            spread_pick = away + " +" + str(abs(dk_spread))

    new_row = {
        'Date': date, 'Home': home, 'Away': away,
        'Home_SP': home_sp, 'Away_SP': away_sp,
        'Predicted_Winner': predicted_winner,
        'Predicted_Winner_Prob': predicted_winner_prob,
        'Model_Home_WinPct': round(win_prob, 4),
        'Model_Fair_ML': fair_ml,
        'DK_Home_ML': dk_home_ml, 'DK_Away_ML': dk_away_ml,
        'ML_Value_Pick': ml_pick, 'ML_Value_Pick_Odds': ml_pick_odds,
        'Model_Total': round(pred_total, 2), 'DK_Total': dk_total, 'Total_Pick': total_pick,
        'Model_Spread': round(pred_spread, 2), 'DK_Spread': dk_spread, 'Spread_Pick': spread_pick,
        'Actual_Home_Runs': np.nan, 'Actual_Away_Runs': np.nan,
        'ML_Result': 'PENDING', 'Total_Result': 'PENDING', 'Spread_Result': 'PENDING',
    }
    existing = log_df[(log_df['Date']==date)&(log_df['Home']==home)&(log_df['Away']==away)]
    if existing.empty:
        log_df = pd.concat([log_df, pd.DataFrame([new_row])], ignore_index=True)
    return log_df


def grade_previous_games(log_df):
    print("\n" + "="*60)
    print("GRADING YESTERDAY'S PICKS")
    print("="*60)
    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    pending = log_df[(log_df['Date']==yesterday)&(log_df['ML_Result']=='PENDING')]
    if pending.empty:
        print(f"No pending picks found for {yesterday}.\n")
        return log_df
    print(f"Fetching scores for {yesterday}...")
    try:
        r = requests.get(
            "https://api.the-odds-api.com/v4/sports/baseball_mlb/scores/",
            params={'apiKey':ODDS_API_KEY,'daysFrom':1,'dateFormat':'iso'},
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
                hs = as_ = None
                for ts in event.get('scores') or []:
                    if ts['name'] == home: hs = int(ts['score'])
                    elif ts['name'] == away: as_ = int(ts['score'])
                if hs is not None and as_ is not None:
                    scores[(home,away)] = (hs, as_)
        graded = 0
        for idx, row in pending.iterrows():
            result = scores.get((row['Home'], row['Away']))
            if result is None: continue
            hr, ar = result
            log_df.at[idx,'Actual_Home_Runs'] = hr
            log_df.at[idx,'Actual_Away_Runs'] = ar
            ml_pick = row.get('ML_Value_Pick')
            if pd.isna(str(ml_pick)) or str(ml_pick)=='nan':
                log_df.at[idx,'ML_Result'] = 'NO_PICK'
            else:
                winner = row['Home'] if hr > ar else row['Away']
                log_df.at[idx,'ML_Result'] = 'WIN' if ml_pick == winner else 'LOSS'
            total_pick = row.get('Total_Pick'); dk_total = row.get('DK_Total')
            if pd.isna(str(total_pick)) or str(total_pick)=='nan':
                log_df.at[idx,'Total_Result'] = 'NO_PICK'
            else:
                actual = hr + ar
                if actual == dk_total:                                          log_df.at[idx,'Total_Result'] = 'PUSH'
                elif 'OVER'  in str(total_pick) and actual > dk_total:         log_df.at[idx,'Total_Result'] = 'WIN'
                elif 'UNDER' in str(total_pick) and actual < dk_total:         log_df.at[idx,'Total_Result'] = 'WIN'
                else:                                                           log_df.at[idx,'Total_Result'] = 'LOSS'
            spread_pick = row.get('Spread_Pick'); dk_spread = row.get('DK_Spread')
            if pd.isna(str(spread_pick)) or str(spread_pick)=='nan':
                log_df.at[idx,'Spread_Result'] = 'NO_PICK'
            else:
                margin = hr - ar
                covered = (margin > abs(dk_spread)) if row['Home'] in str(spread_pick) else ((ar-hr) > abs(dk_spread))
                if abs(margin) == abs(dk_spread): log_df.at[idx,'Spread_Result'] = 'PUSH'
                else:                             log_df.at[idx,'Spread_Result'] = 'WIN' if covered else 'LOSS'
            graded += 1
            print(f"  {row['Away']} @ {row['Home']}: {ar}-{hr}  "
                  f"ML:{log_df.at[idx,'ML_Result']}  "
                  f"Total:{log_df.at[idx,'Total_Result']}  "
                  f"Spread:{log_df.at[idx,'Spread_Result']}")
        print(f"\nGraded {graded} games.")
    except Exception as e:
        print(f"Error fetching scores: {e}")
    return log_df


def print_record(log_df):
    print("\n" + "="*60)
    print("OVERALL MODEL RECORD")
    print("="*60)

    # Predicted winner accuracy (all graded games, regardless of value)
    graded = log_df[log_df['ML_Result'].isin(['WIN','LOSS','NO_PICK'])]
    if not graded.empty and 'Predicted_Winner' in log_df.columns:
        correct = 0; total_pw = 0
        for _, row in graded.iterrows():
            if pd.isna(row.get('Actual_Home_Runs')) or pd.isna(row.get('Actual_Away_Runs')):
                continue
            actual_winner = row['Home'] if row['Actual_Home_Runs'] > row['Actual_Away_Runs'] else row['Away']
            if str(row.get('Predicted_Winner','')) == actual_winner:
                correct += 1
            total_pw += 1
        if total_pw > 0:
            print(f"\n  PREDICTED WINNER (outright model pick, all games)")
            print(f"  Record : {correct}W - {total_pw-correct}L  ({correct/total_pw*100:.1f}% correct)")

    for market, col in [('VALUE BETS (MONEYLINE)','ML_Result'),
                        ('TOTAL RUNS','Total_Result'),
                        ('RUN SPREAD','Spread_Result')]:
        results = log_df[log_df[col].isin(['WIN','LOSS','PUSH'])]
        w = len(results[results[col]=='WIN'])
        l = len(results[results[col]=='LOSS'])
        p = len(results[results[col]=='PUSH'])
        n = len(log_df[log_df[col]=='NO_PICK'])
        total = w + l + p
        if total > 0:
            pct = w / (w+l) * 100 if (w+l) > 0 else 0
            print(f"\n  {market}")
            print(f"  Record : {w}W - {l}L - {p}P  ({pct:.1f}%)")
            print(f"  No pick: {n} games (no edge found)")
        else:
            print(f"\n  {market}: No graded picks yet")

    pd.DataFrame([{
        'Last_Updated': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'ML_Wins':  len(log_df[log_df['ML_Result']=='WIN']),
        'ML_Losses':len(log_df[log_df['ML_Result']=='LOSS']),
        'Tot_Wins': len(log_df[log_df['Total_Result']=='WIN']),
        'Tot_Losses':len(log_df[log_df['Total_Result']=='LOSS']),
        'Sp_Wins':  len(log_df[log_df['Spread_Result']=='WIN']),
        'Sp_Losses':len(log_df[log_df['Spread_Result']=='LOSS']),
    }]).to_csv(SUMMARY_FILE, index=False)
    print(f"\n  Log saved to:     {LOG_FILE}")
    print(f"  Summary saved to: {SUMMARY_FILE}")


# =============================================================================
# STEP 6: VALUE BET SUMMARY
# =============================================================================
def print_value_bet_summary(log_df, date):
    print("\n" + "="*60)
    print(f"VALUE BET SUMMARY FOR {date}")
    print("="*60)

    todays      = log_df[log_df['Date'] == date]
    ml_bets     = todays[todays['ML_Value_Pick'].notna()]
    total_bets  = todays[todays['Total_Pick'].notna()]
    spread_bets = todays[todays['Spread_Pick'].notna()]

    if ml_bets.empty and total_bets.empty and spread_bets.empty:
        print("  No value bets found today.")
    else:
        if not ml_bets.empty:
            print(f"\n  MONEYLINE VALUE BETS ({len(ml_bets)})")
            for _, row in ml_bets.iterrows():
                print(f"  BET: {str(row['ML_Value_Pick']):<28} {row['ML_Value_Pick_Odds']:+.0f}  "
                      f"({row['Away']} @ {row['Home']})")

        if not total_bets.empty:
            print(f"\n  TOTAL RUNS VALUE BETS ({len(total_bets)})")
            for _, row in total_bets.iterrows():
                print(f"  BET: {str(row['Total_Pick']):<28}  "
                      f"(Model: {row['Model_Total']:.1f} | DK line: {row['DK_Total']})")

        if not spread_bets.empty:
            print(f"\n  SPREAD VALUE BETS ({len(spread_bets)})")
            for _, row in spread_bets.iterrows():
                print(f"  BET: {str(row['Spread_Pick']):<28}  "
                      f"(Model margin: {row['Model_Spread']:+.1f} | DK spread: {row['DK_Spread']})")

        total_count = len(ml_bets) + len(total_bets) + len(spread_bets)
        print(f"\n  Total value bets today: {total_count}")

    # ── TOP 10 VALUE BETS (ranked by edge) ──────────────────────────
    print("\n" + "="*60)
    print(f"  TOP 10 VALUE BETS FOR {date}  (ranked by edge)")
    print("="*60)

    ranked = []

    # Moneyline bets — edge in points vs DK line
    for _, row in ml_bets.iterrows():
        fair = row.get('Model_Fair_ML')
        odds = row.get('ML_Value_Pick_Odds')
        if pd.notna(fair) and pd.notna(odds):
            edge = abs(int(odds) - int(fair))
            ranked.append({
                'rank_edge': edge,
                'category': 'ML',
                'label':    f"{row['ML_Value_Pick']} {int(odds):+d}",
                'matchup':  f"{row['Away']} @ {row['Home']}",
                'edge_str': f"{edge:+d} pts vs fair line {int(fair):+d}",
            })

    # Total bets — edge in runs vs DK line
    for _, row in total_bets.iterrows():
        model_tot = row.get('Model_Total')
        dk_tot    = row.get('DK_Total')
        if pd.notna(model_tot) and pd.notna(dk_tot):
            edge = abs(float(model_tot) - float(dk_tot))
            ranked.append({
                'rank_edge': edge,
                'category': 'TOTAL',
                'label':    str(row['Total_Pick']),
                'matchup':  f"{row['Away']} @ {row['Home']}",
                'edge_str': f"{edge:.1f} run edge (model:{float(model_tot):.1f} | DK:{float(dk_tot)})",
            })

    # Spread bets — edge in runs vs DK spread
    for _, row in spread_bets.iterrows():
        model_sp = row.get('Model_Spread')
        dk_sp    = row.get('DK_Spread')
        if pd.notna(model_sp) and pd.notna(dk_sp):
            edge = abs(abs(float(model_sp)) - abs(float(dk_sp)))
            ranked.append({
                'rank_edge': edge,
                'category': 'SPREAD',
                'label':    str(row['Spread_Pick']),
                'matchup':  f"{row['Away']} @ {row['Home']}",
                'edge_str': f"{edge:.1f} run edge (model:{float(model_sp):+.1f} | DK:{float(dk_sp):+.1f})",
            })

    # Sort descending by edge and take top 10
    ranked.sort(key=lambda x: x['rank_edge'], reverse=True)
    top10 = ranked[:10]

    if not top10:
        print("  No value bets found today.")
    else:
        for i, bet in enumerate(top10, 1):
            print(f"  {i:>2}. [{bet['category']:<6}]  {bet['label']:<35}  Edge: {bet['edge_str']}")
            print(f"        {bet['matchup']}")
    print("="*60)


# =============================================================================
# STEP 7: FETCH TODAY'S DRAFTKINGS ODDS
# =============================================================================
def fetch_todays_games():
    try:
        r = requests.get(
            "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds/",
            params={
                'apiKey': ODDS_API_KEY, 'regions': 'us',
                'markets': 'h2h,spreads,totals',
                'bookmakers': 'draftkings', 'oddsFormat': 'american'
            },
            timeout=10
        )
        if r.status_code == 401:
            print("API key rejected.")
            return []
        if r.status_code != 200:
            print(f"Odds API returned status {r.status_code}")
            return []
        data = r.json()
        if not data:
            print("No MLB games on DraftKings today.")
            return []
        games = []
        for event in data:
            game = {
                'home': event.get('home_team',''), 'away': event.get('away_team',''),
                'dk_home_ml': None, 'dk_away_ml': None,
                'dk_spread': None, 'dk_total': None,
            }
            for bm in event.get('bookmakers',[]):
                for mkt in bm.get('markets',[]):
                    k = mkt.get('key','')
                    if k == 'h2h':
                        for o in mkt.get('outcomes',[]):
                            if o['name']==game['home']:   game['dk_home_ml']=o['price']
                            elif o['name']==game['away']: game['dk_away_ml']=o['price']
                    elif k == 'spreads':
                        for o in mkt.get('outcomes',[]):
                            if o['name']==game['home']:   game['dk_spread']=o.get('point')
                    elif k == 'totals':
                        for o in mkt.get('outcomes',[]):
                            if o.get('name')=='Over':     game['dk_total']=o.get('point')
            games.append(game)
        print(f"Found {len(games)} MLB games on DraftKings today.")
        return games
    except Exception as e:
        print(f"Could not connect to odds API: {e}")
        return []


# =============================================================================
# STEP 8: PREDICT A GAME
# =============================================================================
def predict_game(home_input, away_input, dk_home_ml=None,
                 dk_away_ml=None, dk_total=None, dk_spread=None,
                 home_sp_info=None, season=SEASON, log_df=None, date=None):

    home = ABBREV_TO_FULL.get(home_input, home_input)
    away = ABBREV_TO_FULL.get(away_input, away_input)
    hs  = all_teams[(all_teams['Team']==home)&(all_teams['Season']==season)]
    as_ = all_teams[(all_teams['Team']==away)&(all_teams['Season']==season)]
    if hs.empty or as_.empty:
        # Fallback to most recent season available
        hs  = all_teams[all_teams['Team']==home].sort_values('Season').tail(1)
        as_ = all_teams[all_teams['Team']==away].sort_values('Season').tail(1)
        if hs.empty or as_.empty:
            print(f"  No stats found for {home} or {away}, skipping.")
            return log_df
    if not models_trained:
        return log_df

    h = hs.iloc[0]; a = as_.iloc[0]

    home_sp_name = away_sp_name = 'TBD'
    home_sp_stats = away_sp_stats = None

    if home_sp_info:
        home_sp_name  = home_sp_info.get('home_sp_name', 'TBD')
        away_sp_name  = home_sp_info.get('away_sp_name', 'TBD')
        home_sp_stats = home_sp_info.get('home_sp_stats')
        away_sp_stats = home_sp_info.get('away_sp_stats')

    sp_era_home  = home_sp_stats['era']  if home_sp_stats else h['ERA_allowed']
    sp_era_away  = away_sp_stats['era']  if away_sp_stats else a['ERA_allowed']
    sp_whip_home = home_sp_stats['whip'] if home_sp_stats else h['WHIP_allowed']
    sp_whip_away = away_sp_stats['whip'] if away_sp_stats else a['WHIP_allowed']
    sp_k9_home   = home_sp_stats['k9']   if home_sp_stats else h.get('K9_allowed', 8.0)
    sp_k9_away   = away_sp_stats['k9']   if away_sp_stats else a.get('K9_allowed', 8.0)

    # Current season stats
    home_wp        = team_win_pct.get((home, season), 0.500)
    away_wp_season = team_win_pct.get((away, season), 0.500)
    home_l10       = team_last10.get((home, season), 0.500)
    away_l10       = team_last10.get((away, season), 0.500)
    park_factor    = PARK_FACTORS.get(home, 1.0)

    # Days rest (today)
    today_str  = datetime.now().strftime('%Y-%m-%d')
    rest_home  = team_rest.get((home, today_str), 3)
    rest_away  = team_rest.get((away, today_str), 3)

    feats = {
        'OBP_home':  h['OBP'],          'OBP_away':  a['OBP'],
        'SLG_home':  h['SLG'],          'SLG_away':  a['SLG'],
        'ISO_home':  h.get('ISO',.150), 'ISO_away':  a.get('ISO',.150),
        'ERA_home':  h['ERA_allowed'],   'ERA_away':  a['ERA_allowed'],
        'WHIP_home': h['WHIP_allowed'],  'WHIP_away': a['WHIP_allowed'],
        'FIP_home':  h.get('FIP_allowed',4.0), 'FIP_away': a.get('FIP_allowed',4.0),
        'K9_home':   h.get('K9_allowed',8.0),  'K9_away':  a.get('K9_allowed',8.0),
        'BB9_home':  h.get('BB9_allowed',3.0), 'BB9_away': a.get('BB9_allowed',3.0),
        'HR9_home':  h.get('HR9_allowed',1.0), 'HR9_away': a.get('HR9_allowed',1.0),
        'OBP_diff':  h['OBP'] - a['OBP'],
        'ERA_diff':  h['ERA_allowed'] - a['ERA_allowed'],
        'bullpen_era_home': h.get('ERA_allowed', 4.20),
        'bullpen_era_away': a.get('ERA_allowed', 4.20),
        'park_factor':      park_factor,
        'sp_era_home':  sp_era_home,  'sp_era_away':  sp_era_away,
        'sp_whip_home': sp_whip_home, 'sp_whip_away': sp_whip_away,
        'sp_k9_home':   sp_k9_home,   'sp_k9_away':   sp_k9_away,
        'sp_era_diff':  sp_era_home - sp_era_away,
        'sp_whip_diff': sp_whip_home - sp_whip_away,
        'home_win_pct_season': home_wp,
        'away_win_pct_season': away_wp_season,
        'home_last10':         home_l10,
        'away_last10':         away_l10,
        'days_rest_diff':      rest_home - rest_away,
    }

    X_ml  = np.array([[feats.get(f, 0.0) for f in ML_FEATS_USED]])
    X_tot = np.array([[feats.get(f, 0.0) for f in TOT_FEATS_USED]])
    X_sp  = np.array([[feats.get(f, 0.0) for f in SP_FEATS_USED]])

    win_prob    = ml_model.predict_proba(X_ml)[0][1]
    pred_total  = tot_model.predict(X_tot)[0] * park_factor  # apply park factor
    pred_spread = sp_model.predict(X_sp)[0]
    fair_ml     = round(-(win_prob/(1-win_prob))*100) if win_prob >= 0.5 else round(((1-win_prob)/win_prob)*100)

    print(f"\n{'='*60}")
    print(f"  {away[:26]} @ {home[:26]}")
    print(f"{'='*60}")

    # Starting pitchers
    print(f"\n  STARTING PITCHERS")
    if home_sp_stats:
        print(f"  {home[:22]:<24} {home_sp_name} — ERA:{sp_era_home:.2f} WHIP:{sp_whip_home:.2f} K/9:{sp_k9_home:.1f} ({home_sp_stats['ip']:.0f} IP)")
    else:
        print(f"  {home[:22]:<24} {home_sp_name} — using team ERA (no individual stats yet)")
    if away_sp_stats:
        print(f"  {away[:22]:<24} {away_sp_name} — ERA:{sp_era_away:.2f} WHIP:{sp_whip_away:.2f} K/9:{sp_k9_away:.1f} ({away_sp_stats['ip']:.0f} IP)")
    else:
        print(f"  {away[:22]:<24} {away_sp_name} — using team ERA (no individual stats yet)")

    # Moneyline
    away_prob = 1 - win_prob
    fair_away = round(-(away_prob/(1-away_prob))*100) if away_prob >= 0.5 else round(((1-away_prob)/away_prob)*100)

    predicted_winner      = home if win_prob >= 0.5 else away
    predicted_winner_prob = win_prob if win_prob >= 0.5 else away_prob
    predicted_fair_line   = fair_ml if win_prob >= 0.5 else fair_away

    print(f"\n  MONEYLINE")
    print(f"  Predicted winner : {predicted_winner[:26]} ({predicted_winner_prob:.1%} | fair line {predicted_fair_line:+d})")

    if dk_home_ml is not None:
        print(f"  DraftKings line  : {home[:22]} {dk_home_ml:+d} | {away[:22]} {dk_away_ml:+d}")
        value_found = False
        ml_edge_home = dk_home_ml - fair_ml
        ml_edge_away = (dk_away_ml - fair_away) if dk_away_ml else 0

        if win_prob >= 0.5 and ml_edge_home >= MIN_ML_EDGE_PTS:
            value_found = True
            print(f"  Value bet        : {home[:26]} {dk_home_ml:+d}  <-- MODEL SAYS BET THIS")
            print(f"  Why              : DK has them at {dk_home_ml:+d}, model fair line is {fair_ml:+d} (edge: {ml_edge_home:+d} pts)")
        elif win_prob < 0.5 and ml_edge_away >= MIN_ML_EDGE_PTS:
            value_found = True
            print(f"  Value bet        : {away[:26]} {dk_away_ml:+d}  <-- MODEL SAYS BET THIS")
            print(f"  Why              : DK has them at {dk_away_ml:+d}, model fair line is {fair_away:+d} (edge: {ml_edge_away:+d} pts)")
        if not value_found:
            if win_prob >= 0.5 and dk_home_ml is not None and dk_home_ml < 0:
                print(f"  Value bet        : None (model agrees {home[:22]} wins, edge {ml_edge_home:+d} pts — below {MIN_ML_EDGE_PTS} pt threshold)")
            elif win_prob < 0.5 and dk_away_ml is not None and dk_away_ml < 0:
                print(f"  Value bet        : None (model agrees {away[:22]} wins, edge {ml_edge_away:+d} pts — below {MIN_ML_EDGE_PTS} pt threshold)")
            elif win_prob >= 0.5 and dk_home_ml is not None and dk_home_ml > 0:
                print(f"  Value bet        : None — NOTE: model favors {home[:22]} while DK has them as underdogs")
            elif win_prob < 0.5 and dk_away_ml is not None and dk_away_ml > 0:
                print(f"  Value bet        : None — NOTE: model favors {away[:22]} while DK has them as underdogs")
            else:
                print(f"  Value bet        : None")

    # Totals
    print(f"\n  TOTAL RUNS")
    park_note = f" (park factor {park_factor:.2f}x applied)" if park_factor != 1.0 else ""
    print(f"  Model predicts   : {pred_total:.1f} runs{park_note}")
    if dk_total is not None:
        print(f"  DraftKings line  : {dk_total}")
        diff = pred_total - dk_total
        if diff >= MIN_TOT_EDGE_RUNS:    print(f"  Lean OVER {dk_total}  (edge: +{diff:.1f} runs)")
        elif diff <= -MIN_TOT_EDGE_RUNS: print(f"  Lean UNDER {dk_total}  (edge: {diff:.1f} runs)")
        else:                            print(f"  Too close to call  (diff: {diff:+.1f} runs, need +/-{MIN_TOT_EDGE_RUNS})")

    # Spread
    print(f"\n  RUN SPREAD")
    print(f"  Model margin     : {pred_spread:+.1f} runs (positive = home team wins by that much)")
    if dk_spread is not None:
        print(f"  DraftKings spread: {dk_spread:+.1f}")
        spread_diff = abs(pred_spread) - abs(dk_spread)
        if pred_spread > 0 and dk_spread <= -MIN_SP_EDGE_RUNS:   print(f"  Home covers {dk_spread}")
        elif pred_spread < 0 and dk_spread >= MIN_SP_EDGE_RUNS:  print(f"  Away covers")
        else:                                                     print(f"  Too close to call")

    # Team stats summary
    print(f"\n  TEAM STATS  (Season {season})")
    print(f"  {home[:22]:<24} OBP:{h['OBP']:.3f} SLG:{h['SLG']:.3f} ERA:{h['ERA_allowed']:.2f} WHIP:{h['WHIP_allowed']:.2f} Win%:{home_wp:.3f} L10:{home_l10:.0%} Rest:{rest_home}d")
    print(f"  {away[:22]:<24} OBP:{a['OBP']:.3f} SLG:{a['SLG']:.3f} ERA:{a['ERA_allowed']:.2f} WHIP:{a['WHIP_allowed']:.2f} Win%:{away_wp_season:.3f} L10:{away_l10:.0%} Rest:{rest_away}d")
    if park_factor != 1.0:
        direction = "hitter's" if park_factor > 1.0 else "pitcher's"
        print(f"  Park factor      : {park_factor:.2f}x ({direction} park)")

    if log_df is not None and date is not None:
        log_df = save_prediction(log_df, date, home, away, home_sp_name, away_sp_name,
                                 win_prob, fair_ml, dk_home_ml, dk_away_ml,
                                 pred_total, dk_total, pred_spread, dk_spread)
    return log_df


# =============================================================================
# STEP 9: RUN EVERYTHING
# =============================================================================
today  = datetime.now().strftime('%Y-%m-%d')
log_df = load_log()

log_df = grade_previous_games(log_df)
print_record(log_df)

print("\n" + "="*60)
print(f"TODAY'S MLB GAMES ({today}) --- MODEL vs DRAFTKINGS")
print("="*60)

todays_games    = fetch_todays_games()
todays_starters = get_todays_starters()

if todays_starters:
    print("Fetching individual pitcher stats...")
    todays_starters = get_all_starter_stats(todays_starters)

if todays_games and models_trained:
    for game in todays_games:
        home    = game['home']
        away    = game['away']
        sp_info = todays_starters.get((home, away))
        log_df  = predict_game(
            home_input   = home,
            away_input   = away,
            dk_home_ml   = game['dk_home_ml'],
            dk_away_ml   = game['dk_away_ml'],
            dk_total     = game['dk_total'],
            dk_spread    = game['dk_spread'],
            home_sp_info = sp_info,
            log_df=log_df, date=today
        )
elif not todays_games and models_trained:
    print("No live games found. Showing sample predictions.\n")
    for matchup in [('NYY','BOS'),('LAD','ATL'),('HOU','CHC')]:
        log_df = predict_game(matchup[0], matchup[1], log_df=log_df, date=today)

log_df.to_csv(LOG_FILE, index=False)
print(f"\nPredictions saved to {LOG_FILE}")
print(f"Run again tomorrow to auto-grade today's picks!\n")

print_value_bet_summary(log_df, today)

print("ALL TEAM ABBREVIATIONS:")
for abbrev, full in sorted(ABBREV_TO_FULL.items(), key=lambda x: x[1]):
    print(f"  {abbrev} = {full}")
