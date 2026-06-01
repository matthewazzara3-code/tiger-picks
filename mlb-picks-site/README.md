# MLB Edge — Daily Value Picks

Automated MLB betting model that runs daily and publishes value picks to this site.

## Setup

### 1. Add your files to this repo
```
mlb_betting_model_v9.py    ← your main model
export_picks.py            ← the export script (provided)
batting_2022.csv           ← historical data CSVs
batting_2023.csv
batting_2024.csv
batting_2025.csv
pitching_2022.csv
pitching_2023.csv
pitching_2024.csv
pitching_2025.csv
index.html                 ← the website
picks_today.json           ← auto-generated daily (commit an empty one first)
```

### 2. Add your Odds API key as a GitHub Secret
- Go to repo Settings → Secrets and variables → Actions
- Add secret: `ODDS_API_KEY` = your key from the-odds-api.com

### 3. Enable GitHub Pages
- Go to repo Settings → Pages
- Set source to: Deploy from branch → main → / (root)
- Your site will be live at: `https://YOURUSERNAME.github.io/REPONAME`

### 4. Enable GitHub Actions
- Go to Actions tab → enable workflows
- The model runs daily at 11 AM ET automatically
- Or click "Run workflow" to trigger manually

### 5. Update PICKS_URL in index.html
Change this line in `index.html`:
```js
const PICKS_URL = './picks_today.json';
```
To your raw GitHub URL if needed (usually not necessary with GitHub Pages).

## How it works
```
GitHub Actions (11 AM ET daily)
  → runs mlb_betting_model_v9.py
  → runs export_picks.py  (writes picks_today.json)
  → git commits + pushes
  → GitHub Pages serves updated index.html
```

## Local testing
```bash
# Run the model
python mlb_betting_model_v9.py

# Export picks
python export_picks.py

# Serve locally (Python built-in server)
python -m http.server 8000
# Open http://localhost:8000
```
