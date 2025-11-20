# F1 Car Predictor - Project Summary

## 🏎️ What Is This?

A fun ML project that predicts F1 race winners! Uses real race data from 2023-2025 to predict who'll finish in the top 3.

**Current Status:** Working great! ✅  
**Model Accuracy:** 97.9% on 2025 races  
**Next Prediction:** Las Vegas GP 2025 (Round 22)

---

## 🎯 Cool Features

### The ML Stuff
- Two models working together (Random Forest + Gradient Boosting)
- 24 different features to predict race outcomes
- Trained on 5,854 races from the last 3 years
- Predicts top 3 finishers (podium positions)

### The Engineering Stuff
- Uses Prefect to run everything in the right order
- Docker container so it runs the same everywhere
- Split into 6 separate scripts (easier to understand)
- Automatically checks data quality

### Smart Features
- Auto-retries if the API fails
- Saves progress so you can resume if something breaks
- Different modes: full pipeline, just training, or just predictions
- Logs everything so you can see what's happening

---

## 📂 Project Structure

```
f1-car-predictor/
├── src/
│   ├── 01_data_collection.py       # FastF1 API integration
│   ├── 02_data_validation.py       # Data quality checks
│   ├── 03_data_cleaning.py         # Team name standardization
│   ├── 04_feature_engineering.py   # 24 ML features
│   ├── 05_model_training_new.py    # RF + GB training
│   ├── 06_prediction.py            # Race predictions
│   └── f1_prediction_flow.py       # Prefect orchestration
├── data/
│   ├── checkpoint.json             # Collection progress
│   ├── f1_model_features.csv       # Engineered features
│   └── las_vegas_gp_predictions.csv # Predictions
├── models/
│   ├── rf_model.pkl                # Random Forest
│   ├── gb_model.pkl                # Gradient Boosting
│   └── le_team.pkl                 # Label encoders
├── f1_cache/                       # FastF1 cache (~16GB)
├── Dockerfile                      # Container definition
├── docker-compose.yml              # Multi-service orchestration
├── requirements.txt                # Python dependencies
└── docs/
    ├── DOCKER_GUIDE.md
    ├── DOCKER_INSTALL.md
    └── TUTOR.md
```

---

## 🚀 Quick Start

### Option 1: Direct Python Execution

```bash
# Install dependencies
pip install -r requirements.txt

# Run training pipeline
python src/f1_prediction_flow.py --mode training

# Run prediction
python src/f1_prediction_flow.py --mode prediction

# Full pipeline (with data collection)
python src/f1_prediction_flow.py --mode full
```

### Option 2: Docker (Recommended for Production)

```bash
# Build image
docker compose build

# Run training
docker compose up f1-training

# Run with Prefect UI
docker compose --profile server up prefect-server
# Access: http://localhost:4200
```

---

## 📊 Latest Predictions - Las Vegas GP 2025

| Position | Driver | Team | Probability |
|----------|--------|------|-------------|
| 1 | Lando Norris | McLaren | 93.8% |
| 2 | Max Verstappen | Red Bull Racing | 79.1% |
| 3 | Andrea Kimi Antonelli | Mercedes | 10.1% |

**Model Agreement:** 3/3 drivers (both RF and GB agree)  
**Ensemble Confidence:** 61.0% average for top 3

---

## 🛠️ Technical Stack

### Core Technologies
- **Python 3.13**: Primary language
- **FastF1**: F1 data API
- **Scikit-learn**: ML models (RandomForest, GradientBoosting)
- **Pandas/NumPy**: Data manipulation
- **Prefect**: Workflow orchestration

### DevOps & Deployment
- **Docker**: Containerization
- **Docker Compose**: Multi-container orchestration
- **Git**: Version control (GitHub)

### ML Pipeline
```
Data Collection (FastF1 API)
    ↓
Data Validation (Quality Checks)
    ↓
Data Cleaning (Standardization)
    ↓
Feature Engineering (24 Features)
    ↓
Model Training (RF + GB)
    ↓
Prediction (Ensemble)
```

---

## 🎓 What I Learned Building This

### Writing Clean Code
- Breaking big problems into small pieces
- Making code that handles errors gracefully
- Command-line interfaces that are easy to use
- Writing helpful comments and documentation

### ML Pipelines
- Using Prefect to automate workflows
- Checking data quality before training
- Saving and loading models properly
- Making results reproducible

### Docker & Deployment
- Packaging apps in containers
- Managing multiple services with Docker Compose
- Keeping data between container runs
- Configuration for different environments

### Working with Data
- Getting data from APIs (FastF1)
- Building data pipelines (collect → clean → transform)
- Creating useful features from raw data
- Making sure data quality is good

---

## 📈 Model Performance Details

### Training Configuration
- **Train Set**: 2023-2024 seasons (878 samples)
- **Test Set**: 2025 season (419 samples)
- **Class Balance**: 15% podium, 85% non-podium
- **Handling**: Balanced class weights

### Feature Importance (Random Forest)
1. **recent_podiums** (36.2%) - Recent top 3 finishes
2. **recent_avg_position** (27.4%) - Rolling average position
3. **grid_advantage** (12.9%) - Grid position vs field average
4. **GridPosition** (10.7%) - Starting position
5. **team_avg_position** (5.9%) - Team performance

### Evaluation Metrics
```
Random Forest (97.4% accuracy):
              precision  recall  f1-score
Not Top 3       0.99      0.98      0.98
Top 3           0.89      0.94      0.91

Gradient Boosting (97.9% accuracy):
              precision  recall  f1-score
Not Top 3       0.99      0.99      0.99
Top 3           0.94      0.92      0.93
```

---

## 🔄 Development Workflow

### Daily Development
1. Make code changes
2. Run: `python src/f1_prediction_flow.py --mode training`
3. Review predictions
4. Iterate

### Data Updates
1. New race weekend completes
2. Run: `python src/f1_prediction_flow.py --mode full`
3. Models retrain automatically
4. New predictions generated

### Production Deployment
1. Build: `docker compose build`
2. Test: `docker compose up f1-training`
3. Deploy: Push to container registry
4. Schedule: Cron/cloud scheduler

---

## 📝 Next Steps (Weeks 3-4)

### Week 3: CI/CD + Testing
- [ ] GitHub Actions workflows
- [ ] Automated testing (pytest)
- [ ] Code quality checks (black, flake8)
- [ ] Automated Docker builds

### Week 4: Monitoring + Documentation
- [ ] Model performance monitoring
- [ ] Prediction accuracy tracking
- [ ] Grafana dashboards
- [ ] Complete documentation

---

## 💡 Cool Things I Figured Out

### The Big Challenge
The FastF1 API sometimes fails or is slow, so I added:
- Automatic retries (tries 3 times before giving up)
- Checkpoint system (remembers what's already downloaded)
- Smart error handling (doesn't crash on bad data)

### Why Split Into 6 Scripts?
Easier to:
- Debug (run just one part at a time)
- Understand (each script does one thing)
- Modify (change features without touching data collection)
- Test (test each piece separately)

### Docker Was Tricky But Worth It
- Works the same on any computer
- No "works on my machine" problems
- Easy to share with others
- Can deploy to cloud later if wanted

### Prefect Makes Life Easier
- Runs tasks in the right order automatically
- Retries failed tasks
- Shows what's happening in real-time
- Can schedule to run daily

---

## 📚 Documentation

- **[PREFECT_README.md](PREFECT_README.md)** - Prefect orchestration guide
- **[docs/DOCKER_GUIDE.md](docs/DOCKER_GUIDE.md)** - Docker usage and deployment
- **[docs/DOCKER_INSTALL.md](docs/DOCKER_INSTALL.md)** - Docker installation for Windows
- **[docs/TUTOR.md](docs/TUTOR.md)** - Learning resources and tutorials

---

## 🔗 Repository

**GitHub**: https://github.com/FabulousDexter/f1-car-predictor  
**Branch**: main  
**License**: MIT

---

## 📊 Current Status

- ✅ Data collection with checkpoint system
- ✅ Data cleaning and validation
- ✅ Feature engineering (24 features)
- ✅ Model training (RF + GB, 97.9% accuracy)
- ✅ Prediction system with auto-detection
- ✅ Prefect orchestration
- ✅ Docker containerization

**What's Working**: Everything! Models trained, predictions accurate, Docker running smoothly.

**Next Ideas** (if I feel like it):
- Add GitHub Actions for automation
- Build a simple web dashboard
- Try other ML models (XGBoost, Neural Nets)
- Add weather data as features

---

**Built for fun and learning!** 🏎️

**Tech Used**: Python, Prefect, Docker, Scikit-learn, FastF1, Pandas
