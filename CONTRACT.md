# Script Contract - Task 3 Pipeline

This defines the input/output paths each script must follow.
DO NOT change paths - dvc.yaml depends on them.

## Sandaru (Data Engineer) - DONE

- Set up DVC + DagsHub remote storage
- Added Tetouan dataset to DVC tracking
- Wrote 4-stage dvc.yaml pipeline
- Wrote placeholder Python scripts

## Binara (ML Engineer) - YOUR WORK

Rewrite the following scripts with the real ML logic. Keep paths exactly as listed.

### src/preprocess_new_data.py
Reads:
- data/power_consumption.csv (Tetouan dataset; target column: "Zone 1 Power Consumption")

Writes:
- artifacts/data/X_train.npy
- artifacts/data/y_train.npy
- artifacts/data/X_test.npy
- artifacts/data/y_test.npy
- artifacts/preprocessing/scaler.pkl
- artifacts/preprocessing/feature_columns.json

### src/model.py
Reads:
- artifacts/data/X_train.npy
- artifacts/data/y_train.npy

Writes:
- artifacts/models/model.pkl (current placeholder is Random Forest; if you switch to Keras, save as model.keras and update dvc.yaml accordingly)
- artifacts/metrics/training_history.json

### src/evaluate.py
Reads:
- artifacts/models/model.pkl
- artifacts/data/X_test.npy
- artifacts/data/y_test.npy

Writes:
- artifacts/metrics/evaluation_metrics.json (must include MAE, RMSE, R2)

## Matthew (GitHub Actions + Monitoring) - YOUR WORK (BIGGEST CHUNK - 20 RUBRIC PTS)

### src/monitor.py
Reads:
- artifacts/models/model.pkl
- artifacts/data/X_test.npy

Writes:
- reports/drift_report.json
- artifacts/metrics/monitoring_metrics.json

Add proper drift detection (consider Evidently AI or expand the KS-test placeholder).

### .github/workflows/retrain-on-push.yml
MUST include ALL THREE triggers for full 12 pts:
1. on: push (paths: "data/**") - auto-retrain on data push
2. on: schedule (cron, e.g. weekly) - scheduled retraining
3. on: workflow_dispatch - manual trigger button

The workflow should:
- Check out the repo
- Install dependencies
- Run dvc pull to fetch data
- Run dvc repro to retrain
- Run dvc push to upload new artifacts
- Commit + push the updated dvc.lock back

## Ashen (Project Management + Report)

- Set up Trello board with backlog, in-progress, done columns
- Break Task 3 work into tickets, assign across the team
- Maintain board activity from now until submission (rubric explicitly penalises last-minute setup)
- Draft the report skeleton (cover page, AoC, rubric headings)
- Compile screenshots from each member into the final report

## Commit Style (EVERYONE)

Use semantic commits for the 4-pt Commit Structure rubric:
- feat: new feature
- fix: bug fix
- chore: housekeeping
- docs: documentation
- refactor: code restructure

Example: feat(pipeline): replace placeholder model with tuned Random Forest

## Local Setup

1. Clone the repo into a NON-OneDrive folder (e.g. C:\Users\YOUR_NAME\projects\)
2. python -m venv venv
3. .\venv\Scripts\Activate.ps1  (Windows)  OR  source venv/bin/activate  (Mac/Linux)
4. pip install dvc pandas numpy scikit-learn joblib scipy
5. Get DagsHub token from dagshub.com/user/settings/tokens
6. dvc remote modify origin --local auth basic
7. dvc remote modify origin --local user YOUR_DAGSHUB_USERNAME
8. dvc remote modify origin --local password YOUR_TOKEN
9. dvc pull
10. dvc repro

## Critical Warning - OneDrive

Do NOT clone this repo into a OneDrive-synced folder. OneDrive locks files
and breaks Git operations. Clone outside OneDrive.
