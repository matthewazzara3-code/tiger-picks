import json, os
import pandas as pd
import numpy as np
from datetime import datetime

LOG_FILE    = "mlb_predictions_log.csv"
OUTPUT_JSON = "picks_today.json"

def safe_float(val, default=None):
    try:
        f = float(val)
        return None if np.isnan(f) else round(f, 4)
    except: return default

def safe_int(val, default=None):
    try:
        f = float(val)
        return None if np.isnan(f) else int(f)
    except: return default

def edge_to_confidence(edge_pts=None, edge_runs=None, bet_type='ml'):
    if bet_type=='ml' and edge_pts is not None:
        return min(99, int(50 + edge_pts * 1.5))
    elif bet_type in ('total','spread') and edge_runs is not None:
        return min(99, int(50 + edge_runs * 10))
    return 60

def export_picks():
    today = datetime.now().strftime('%Y-%m-%d')
    print(f"\nExporting picks for {today}...")

    if not os.path.exists(LOG_FILE):
        print(f"  {LOG_FILE} not found.")
        out = {"date":today,"picks":[],"record":{},"generated_at":datetime.now().isoformat()}
        with open(OUTPUT_JSON,'w') as f: json.dump(out,f,indent=2)
        return

    log = pd.read_csv(LOG_FILE)
    todays = log[log['Date']==today].copy()
    picks = []

    for _, row in todays.iterrows():
        home = str(row.get('Home',''))
        away = str(row.get('Away',''))
        matchup = f"{away} @ {home}"
        home_sp = str(row.get('Home_SP','TBD'))
        away_sp = str(row.get('Away_SP','TBD'))
        win_prob    = safe_float(row.get('Predicted_Winner_Prob'))
        model_total = safe_float(row.get('Model_Total'))
        dk_total    = safe_float(row.get('DK_Total'))
        model_spread= safe_float(row.get('Model_Spread'))
        dk_spread   = safe_float(row.get('DK_Spread'))
        fair_ml     = safe_int(row.get('Model_Fair_ML'))

        ml_pick = row.get('ML_Value_Pick')
        ml_odds = row.get('ML_Value_Pick_Odds')
        if pd.notna(str(ml_pick)) and str(ml_pick) not in ('nan','None',''):
            dk_ml = safe_int(row.get('DK_Home_ML') if str(ml_pick)==home else row.get('DK_Away_ML'))
            edge_pts = abs(int(dk_ml)-int(fair_ml)) if dk_ml and fair_ml else 10
            conf = edge_to_confidence(edge_pts=edge_pts, bet_type='ml')
            picks.append({
                "id": f"{today}_{home}_{away}_ml",
                "date": today, "matchup": matchup, "home": home, "away": away,
                "home_sp": home_sp, "away_sp": away_sp,
                "type": "moneyline",
                "pick": f"{ml_pick} ML",
                "odds": f"{int(float(str(ml_odds))):+d}" if pd.notna(str(ml_odds)) and str(ml_odds) not in ('nan','None','') else "N/A",
                "confidence": conf,
                "edge": f"{edge_pts:+d} pts vs fair line {fair_ml:+d}" if fair_ml else "",
                "model_win_prob": f"{float(win_prob)*100:.1f}%" if win_prob else None,
                "notes": f"SP: {home_sp} vs {away_sp}. Model fair line: {fair_ml:+d}." if fair_ml else "",
                "result": str(row.get('ML_Result','PENDING')),
            })

        total_pick = row.get('Total_Pick')
        if pd.notna(str(total_pick)) and str(total_pick) not in ('nan','None',''):
            edge_runs = abs(float(model_total)-float(dk_total)) if model_total and dk_total else 1.5
            conf = edge_to_confidence(edge_runs=edge_runs, bet_type='total')
            picks.append({
                "id": f"{today}_{home}_{away}_total",
                "date": today, "matchup": matchup, "home": home, "away": away,
                "home_sp": home_sp, "away_sp": away_sp,
                "type": "total",
                "pick": str(total_pick), "odds": "-110", "confidence": conf,
                "edge": f"{edge_runs:.1f} run edge (model:{model_total:.1f} | DK:{dk_total})" if model_total and dk_total else "",
                "dk_total": dk_total, "model_total": model_total,
                "notes": f"SP: {home_sp} vs {away_sp}. Model projects {model_total:.1f} total runs." if model_total else "",
                "result": str(row.get('Total_Result','PENDING')),
            })

        spread_pick = row.get('Spread_Pick')
        if pd.notna(str(spread_pick)) and str(spread_pick) not in ('nan','None',''):
            edge_runs = abs(abs(float(model_spread))-abs(float(dk_spread))) if model_spread and dk_spread else 1.5
            conf = edge_to_confidence(edge_runs=edge_runs, bet_type='spread')
            picks.append({
                "id": f"{today}_{home}_{away}_spread",
                "date": today, "matchup": matchup, "home": home, "away": away,
                "home_sp": home_sp, "away_sp": away_sp,
                "type": "runline",
                "pick": str(spread_pick), "odds": "-110", "confidence": conf,
                "edge": f"{edge_runs:.1f} run edge (model:{model_spread:+.1f} | DK:{dk_spread:+.1f})" if model_spread and dk_spread else "",
                "dk_spread": dk_spread, "model_spread": model_spread,
                "notes": f"SP: {home_sp} vs {away_sp}. Model margin: {model_spread:+.1f} runs." if model_spread else "",
                "result": str(row.get('Spread_Result','PENDING')),
            })

    # ── FULL HISTORICAL RECORD from entire log ──────────────────────────────
    def count_results(col):
        settled = log[log[col].isin(['WIN','LOSS','PUSH'])]
        return {
            "wins":   int(len(settled[settled[col]=='WIN'])),
            "losses": int(len(settled[settled[col]=='LOSS'])),
            "pushes": int(len(settled[settled[col]=='PUSH'])),
        }

    record = {
        "moneyline": count_results('ML_Result'),
        "total":     count_results('Total_Result'),
        "runline":   count_results('Spread_Result'),
        "last_updated": datetime.now().isoformat(),
    }

    output = {
        "date": today, "picks": picks, "record": record,
        "generated_at": datetime.now().isoformat(),
    }
    with open(OUTPUT_JSON,'w') as f: json.dump(output,f,indent=2)
    ml=record['moneyline']; tot=record['total']; rl=record['runline']
    print(f"  Exported {len(picks)} picks")
    print(f"  ALL-TIME Record — ML:{ml['wins']}-{ml['losses']} | Total:{tot['wins']}-{tot['losses']} | RL:{rl['wins']}-{rl['losses']}")

export_picks()
