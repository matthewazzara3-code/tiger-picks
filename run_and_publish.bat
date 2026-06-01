@echo off
title Tiger Picks - Run Model & Publish to Website
color 0A

echo ============================================
echo   TIGER PICKS - Daily Model Runner
echo ============================================
echo.

:: Navigate to MLB Model folder
cd /d "C:\Users\Owner\OneDrive - University of Cincinnati\Attachments\files\MLB Model"

echo [1/3] Running MLB model... (this takes 3-8 minutes)
echo.
python mlb_betting_model_v9.py

:: Check if model ran successfully
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Model failed to run. Check output above for errors.
    pause
    exit /b 1
)

echo.
echo [2/3] Model complete! Pushing picks to website...
echo.

:: Push picks_today.json to GitHub
git add picks_today.json
git add mlb_predictions_log.csv
git add mlb_record_summary.csv

git commit -m "Auto-update picks for %date%"

git push

:: Check if push was successful
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: GitHub push failed. Make sure you're connected to the internet
    echo and your GitHub credentials are set up.
    pause
    exit /b 1
)

echo.
echo ============================================
echo   SUCCESS! Website updated.
echo   Visit: https://matthewazzara3-code.github.io/tiger-picks/
echo   (may take 30 seconds to refresh)
echo ============================================
echo.
pause
