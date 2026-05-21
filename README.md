# Task 3 - Automated Retraining & Monitoring Pipeline

**CL04 G05** | AI for Engineering | Swinburne University

End-to-end MLOps pipeline for the Tetouan power consumption prediction system.
Combines DVC for data versioning, GitHub Actions for automated retraining,
and Python scripts for monitoring and drift detection.

## Pipeline Stages

1. **preprocess** - cleans and splits the Tetouan dataset
2. **train** - trains regression model on power consumption data
3. **evaluate** - computes MAE, RMSE, R-squared on the test set
4. **monitor** - Kolmogorov-Smirnov drift detection + performance tracking

## Quick Start

git clone https://github.com/Sandaru757/Task-3-Automated-retraining-and-monitoring-pipeline.-CL04-G05.git
cd Task-3-Automated-retraining-and-monitoring-pipeline.-CL04-G05
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install dvc pandas numpy scikit-learn joblib scipy
dvc pull
dvc repro

See CONTRACT.md for full I/O specification and per-person responsibilities.

## Team (Task 3 roles)

- **Sandaru** - DVC pipeline, dataset tracking, DagsHub remote storage
- **Binara** - ML scripts: preprocessing, training, evaluation
- **Matthew** - GitHub Actions workflow, automated retraining triggers, monitoring
- **Ashen** - Trello project board, report skeleton, screenshot compilation

## Remote Storage

Dataset and model artifacts are stored on DagsHub:
https://dagshub.com/Sandaru757/Task-3-Automated-retraining-and-monitoring-pipeline.-CL04-G05
