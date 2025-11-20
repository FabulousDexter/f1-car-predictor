# 🏎️ F1 Race Predictor

A machine learning project that predicts Formula 1 race winners! Built as a learning project to practice ML pipelines, Docker, and workflow orchestration.

## What Does It Do?

Predicts which drivers will finish in the top 3 (podium) for upcoming F1 races. Currently trained on 2023-2025 data with **97.9% accuracy**!

### Latest Prediction: Las Vegas GP 2025
1. **Lando Norris (McLaren)** - 93.8% chance
2. **Max Verstappen (Red Bull)** - 79.1% chance
3. **Andrea Kimi Antonelli (Mercedes)** - 10.1% chance

---

## Quick Start

### Option 1: Run Directly
```bash
# Install dependencies
pip install -r requirements.txt

# Train models
python src/f1_prediction_flow.py --mode training

# Make predictions
python src/f1_prediction_flow.py --mode prediction
```

### Option 2: Use Docker
```bash
# Build container
docker compose build

# Run training
docker compose up f1-training

# Make predictions
docker compose up f1-prediction
```

---

## How It Works

```
1. Collect F1 Data (from FastF1 API)
   ↓
2. Clean the Data (fix team names, remove bad data)
   ↓
3. Create Features (recent form, grid position, points)
   ↓
4. Train Models (Random Forest + Gradient Boosting)
   ↓
5. Make Predictions! 🏁
```

---

## Project Structure

```
f1-car-predictor/
├── src/
│   ├── 01_data_collection.py       # Get F1 data from API
│   ├── 02_data_validation.py       # Check data quality
│   ├── 03_data_cleaning.py         # Clean up messy data
│   ├── 04_feature_engineering.py   # Create ML features
│   ├── 05_model_training_new.py    # Train the models
│   ├── 06_prediction.py            # Make predictions
│   └── f1_prediction_flow.py       # Run everything together
├── data/                           # Where data lives
├── models/                         # Saved ML models
├── f1_cache/                       # FastF1 cache (~16GB)
└── docs/                           # Help guides
```

---

## What I Learned

### Machine Learning
- Building end-to-end ML pipelines
- Feature engineering for real-world data
- Ensemble models (combining Random Forest + Gradient Boosting)
- Handling imbalanced datasets (15% podium vs 85% non-podium)

### Data Engineering
- Working with APIs (FastF1)
- Data cleaning and validation
- Checkpoint systems for resuming failed jobs
- Handling missing data and outliers

### DevOps
- Docker containerization
- Workflow orchestration with Prefect
- Error handling and retries
- Logging and monitoring

---

## Tech Stack

- **Python 3.13**: Main language
- **FastF1**: Gets F1 race data
- **Scikit-learn**: Machine learning models
- **Pandas**: Data manipulation
- **Prefect**: Workflow management
- **Docker**: Containerization

---

## Model Performance

**Random Forest**: 97.4% accuracy
**Gradient Boosting**: 97.9% accuracy

Tested on 2025 season (419 races)

### What the model looks at:
1. Recent podium finishes (36%)
2. Recent average position (27%)
3. Grid advantage (13%)
4. Starting position (11%)
5. Team performance (6%)
6. Other stats (7%)

---

## Running Different Modes

```bash
# Full pipeline (collect data + train + predict)
python src/f1_prediction_flow.py --mode full

# Training only (skip data collection)
python src/f1_prediction_flow.py --mode training

# Prediction only (use existing models)
python src/f1_prediction_flow.py --mode prediction
```

---

## Files You'll Need

### For Power BI / Visualization
- `data/f1_model_features.csv` - Complete dataset with all features
- `data/predictions_round_22_2025.csv` - Latest predictions

### Models
- `models/rf_model.pkl` - Random Forest model
- `models/gb_model.pkl` - Gradient Boosting model
- `models/le_team.pkl` - Team name encoder

---

## Troubleshooting

**Docker issues?** See [docs/DOCKER_INSTALL.md](docs/DOCKER_INSTALL.md)

**Want to understand Prefect?** Check [PREFECT_README.md](PREFECT_README.md)

**Docker commands?** Look at [docs/DOCKER_GUIDE.md](docs/DOCKER_GUIDE.md)

---

## Next Steps (If You Want)

- [ ] Add more features (weather, tire strategy)
- [ ] Try different ML models (XGBoost, Neural Networks)
- [ ] Build a web dashboard
- [ ] Add GitHub Actions for automation
- [ ] Deploy to cloud (AWS, GCP, Azure)

---

## License

Feel free to use this for learning! MIT License.

---

**Built for learning ML pipelines and having fun with F1 data!** 🏎️💨
