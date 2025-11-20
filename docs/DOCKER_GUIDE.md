# F1 Car Predictor - Docker Guide

## 🐳 Docker Setup

### Prerequisites
- Docker Desktop installed
- Docker Compose installed

### Quick Start

```bash
# Build the Docker image
docker-compose build

# Run training pipeline
docker-compose up f1-training

# Run full pipeline (with data collection)
docker-compose --profile full up f1-full-pipeline

# Run prediction only
docker-compose --profile prediction up f1-prediction

# Run with Prefect UI
docker-compose --profile server up prefect-server
```

---

## 📦 Docker Services

### 1. f1-training (Default)
Runs training pipeline without data collection.

**Use when:**
- Data is already collected
- Quick model retraining
- Development/testing

**Command:**
```bash
docker-compose up f1-training
```

**What it does:**
- Feature engineering
- Model training
- Saves models to `./models/`

---

### 2. f1-full-pipeline
Runs complete pipeline from data collection to prediction.

**Use when:**
- First time setup
- New race data available
- Full pipeline testing

**Command:**
```bash
docker-compose --profile full up f1-full-pipeline
```

**What it does:**
- Checks data status
- Collects new data (if needed)
- Validates → Cleans → Features → Trains → Predicts

---

### 3. f1-prediction
Runs prediction only using existing models.

**Use when:**
- Models already trained
- Quick predictions needed
- Different race scenarios

**Command:**
```bash
docker-compose --profile prediction up f1-prediction
```

**Customize race:**
```bash
# Edit docker-compose.yml environment section:
environment:
  - YEAR=2025
  - ROUND=23  # Change round number
```

---

### 4. prefect-server (Optional)
Runs Prefect UI for monitoring.

**Command:**
```bash
docker-compose --profile server up prefect-server
```

**Access UI:**
- Open http://localhost:4200
- View flow runs, logs, and task details

---

## 🔧 Development Workflow

### Build Image
```bash
# Build fresh image
docker-compose build

# Build with no cache
docker-compose build --no-cache
```

### View Logs
```bash
# Follow logs
docker-compose logs -f f1-training

# View last 100 lines
docker-compose logs --tail=100 f1-training
```

### Stop Services
```bash
# Stop all services
docker-compose down

# Stop and remove volumes
docker-compose down -v
```

### Shell Access
```bash
# Access running container
docker exec -it f1-training bash

# Run one-off command
docker-compose run f1-training python src/04_feature_engineering.py
```

---

## 📁 Volume Mounts

Data persists between container runs via volumes:

```yaml
volumes:
  - ./data:/app/data          # Data files
  - ./models:/app/models      # Trained models
  - ./f1_cache:/app/f1_cache  # FastF1 cache (16GB+)
```

**Important:**
- Models saved in container → appear in `./models/` on host
- Data collected → saved to `./data/` on host
- Cache persists → faster subsequent runs

---

## 🚀 Production Deployment

### Build for Production
```bash
# Build optimized image
docker build -t f1-predictor:latest .

# Tag for registry
docker tag f1-predictor:latest myregistry/f1-predictor:latest

# Push to registry
docker push myregistry/f1-predictor:latest
```

### Run in Production
```bash
# Run with resource limits
docker run -d \
  --name f1-training \
  --memory="4g" \
  --cpus="2" \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  f1-predictor:latest
```

---

## 🎯 Common Use Cases

### Daily Model Update
```bash
# Automated daily training
docker-compose up f1-training
```

### Race Day Prediction
```bash
# Get latest predictions
docker-compose --profile prediction up f1-prediction
```

### New Season Setup
```bash
# Collect all data from scratch
docker-compose --profile full up f1-full-pipeline
```

### Monitoring & Debugging
```bash
# Start Prefect UI
docker-compose --profile server up -d prefect-server

# Run training and monitor
docker-compose up f1-training

# View in browser: http://localhost:4200
```

---

## 🐛 Troubleshooting

### Image Won't Build
```bash
# Clear Docker cache
docker system prune -a

# Rebuild from scratch
docker-compose build --no-cache
```

### Container Exits Immediately
```bash
# Check logs
docker-compose logs f1-training

# Check if data/models exist
ls -la data/ models/
```

### Slow Performance
```bash
# Increase Docker resources in Docker Desktop:
# Settings → Resources → Memory (4GB+)
# Settings → Resources → CPUs (2+)
```

### Permission Issues
```bash
# Fix ownership (Linux/Mac)
sudo chown -R $USER:$USER data/ models/ f1_cache/

# Windows: Run Docker Desktop as Administrator
```

---

## 📊 Resource Usage

**Typical Requirements:**
- **Memory**: 2-4GB
- **CPU**: 2 cores
- **Disk**: 20GB+ (F1 cache is large)
- **Network**: FastF1 API access

**Optimization Tips:**
- Use `f1-training` instead of `f1-full-pipeline` when possible
- Mount existing F1 cache to avoid re-downloads
- Use Docker layer caching for faster builds

---

## 🔐 Security Notes

**For production:**
1. Use secrets for API keys (if added later)
2. Run as non-root user
3. Scan images: `docker scan f1-predictor:latest`
4. Keep base images updated

```dockerfile
# Add to Dockerfile for security:
RUN useradd -m -u 1000 f1user
USER f1user
```

---

## 📚 Next Steps

1. **Test locally**: `docker-compose up f1-training`
2. **Deploy to cloud**: AWS ECS, GCP Cloud Run, Azure Container Instances
3. **Add CI/CD**: GitHub Actions (Week 3)
4. **Monitor**: Prefect Cloud + Container logs


