# F1 Prediction Pipeline with Prefect

## What's Prefect?

Prefect is a workflow tool that runs your scripts in the right order and handles errors automatically. Think of it like a smart manager for your code!

## What's Set Up

### The 6 Main Scripts
- `01_data_collection.py` - Downloads F1 data from FastF1 API
- `02_data_validation.py` - Checks if the data looks good
- `03_data_cleaning.py` - Fixes messy team names and bad data
- `04_feature_engineering.py` - Creates 24 features for ML
- `05_model_training_new.py` - Trains the models (97.9% accuracy!)
- `06_prediction.py` - Makes predictions for upcoming races
- `f1_prediction_flow.py` - Runs everything with Prefect

---

## 🚀 Quick Start

### Run Individual Scripts

```bash
# 1. Collect data
python src/01_data_collection.py

# 2. Validate data
python src/02_data_validation.py

# 3. Clean data
python src/03_data_cleaning.py

# 4. Engineer features
python src/04_feature_engineering.py

# 5. Train models
python src/05_model_training_new.py

# 6. Generate predictions
python src/06_prediction.py
```

### Run with Prefect Orchestration

```bash
# Full pipeline (data collection → prediction)
python src/f1_prediction_flow.py --mode full

# Training only (skip data collection)
python src/f1_prediction_flow.py --mode training

# Prediction only (use existing models)
python src/f1_prediction_flow.py --mode prediction --year 2025 --round 22

# Force data collection
python src/f1_prediction_flow.py --mode full --collect-data

# Retry failed sessions
python src/f1_prediction_flow.py --mode full --retry-failed
```

---

## 📊 Pipeline Architecture

```
Data Collection (FastF1 API)
    ↓ checkpoint.json
Data Validation (Quality Checks)
    ↓
Data Cleaning (Team Standardization)
    ↓ clean CSVs
Feature Engineering (24 features)
    ↓ f1_model_features.csv
Model Training (RF + GB)
    ↓ models/*.pkl
Prediction (Las Vegas GP 2025)
    ↓ las_vegas_gp_predictions.csv
```

---

## Why Prefect Is Cool

### It Handles Failures
- If a task fails, it tries again (2-3 times)
- Saves your progress so you don't lose work
- Shows clear error messages when something breaks

### You Can See What's Happening
- Full logs of everything that runs
- History of all your runs
- Can track how long each step takes

### Runs Things Efficiently
- Can run some tasks at the same time (parallel)
- Skips unnecessary steps
- Handles dependencies automatically (runs things in order)

### Different Ways to Run

1. **Full Pipeline** (`--mode full`)
   - Does everything: collect data → validate → clean → features → train → predict
   - Use when: Starting fresh or want to update everything

2. **Training Only** (`--mode training`)
   - Skips data collection, just trains models
   - Use when: You already have the data and just want to retrain

3. **Prediction Only** (`--mode prediction`)
   - Uses existing models to make quick predictions
   - Use when: Models are trained, just want new predictions

---

## 📈 Model Performance

Current models (trained on 2023-2024, tested on 2025):
- **Random Forest**: 97.4% accuracy
- **Gradient Boosting**: 97.9% accuracy
- **Ensemble**: 3/3 model agreement on top 3

### Las Vegas GP 2025 Predictions
1. Lando Norris (McLaren) - 93.8%
2. Max Verstappen (Red Bull Racing) - 79.1%
3. Andrea Kimi Antonelli (Mercedes) - 10.1%

---

## 🔧 Configuration

### Data Files
- `data/checkpoint.json` - Collection progress
- `data/f1_all_session_race_results.csv` - Raw race data
- `data/f1_all_session_race_results_clean.csv` - Cleaned race data
- `data/f1_model_features.csv` - ML-ready features

### Models
- `models/rf_model.pkl` - Random Forest
- `models/gb_model.pkl` - Gradient Boosting
- `models/le_team.pkl` - Team name encoder
- `models/feature_columns.pkl` - Feature list

---

## 🔧 What You Need

```txt
fastf1        # F1 data API
pandas        # Data manipulation
numpy         # Math stuff
scikit-learn  # ML models
joblib        # Saving models
pyyaml        # Config files
prefect       # Workflow management
```

Install everything:
```bash
pip install -r requirements.txt
```

---

## 💡 Quick Tips

**Speed Things Up**
```bash
# Skip the slow data collection when testing
python src/f1_prediction_flow.py --mode training
```

**Debug One Step**
```bash
# Run just one script to test it
python src/04_feature_engineering.py
```

**See What's Happening**
```bash
# Start Prefect UI to watch your flows run
prefect server start
# Then open http://127.0.0.1:4200 in browser
```

---

## 🎉 That's It!

Prefect makes the pipeline easier to run and debug. It's not required (you can run scripts individually), but it makes life easier!

**Main benefit**: Run everything with one command, and it handles the rest! 🚀
