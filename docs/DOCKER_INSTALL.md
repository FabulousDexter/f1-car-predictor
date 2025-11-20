# Docker Installation Guide for Windows

## 📥 Install Docker Desktop

### Step 1: Download Docker Desktop
1. Visit: https://www.docker.com/products/docker-desktop/
2. Click "Download for Windows"
3. Choose: **Docker Desktop for Windows** (requires Windows 10/11)

### Step 2: Install Docker Desktop
1. Run the installer (Docker Desktop Installer.exe)
2. Follow installation wizard
3. **Enable WSL 2** when prompted (recommended)
4. Restart computer when installation completes

### Step 3: Verify Installation
Open PowerShell or Command Prompt:

```bash
# Check Docker version
docker --version

# Check Docker Compose version
docker compose version

# Test Docker is working
docker run hello-world
```

Expected output:
```
Docker version 24.0.x, build xxxxx
Docker Compose version v2.x.x
Hello from Docker!
```

---

## ⚙️ Docker Desktop Settings

### Recommended Configuration

**Resources:**
- Memory: 4 GB minimum (8 GB recommended)
- CPUs: 2 minimum (4 recommended)
- Disk: 30 GB for F1 cache

**Settings Path:**
Docker Desktop → Settings → Resources

---

## 🐛 Troubleshooting

### WSL 2 Installation Required

If you see: "WSL 2 installation is incomplete"

**Fix:**
1. Open PowerShell as Administrator
2. Run: `wsl --install`
3. Restart computer
4. Run: `wsl --set-default-version 2`

### Virtualization Not Enabled

If you see: "Hardware assisted virtualization and data execution protection must be enabled"

**Fix:**
1. Restart computer
2. Enter BIOS/UEFI (press F2, F10, or DEL during startup)
3. Enable: "Intel VT-x" or "AMD-V"
4. Enable: "Virtualization Technology"
5. Save and restart

### Docker Desktop Won't Start

**Fix:**
1. Right-click Docker Desktop icon
2. Select "Quit Docker Desktop"
3. Open as Administrator
4. Wait for Docker to start (whale icon in system tray)

---

## ✅ After Installation

Once Docker is installed, return to project directory and run:

```bash
# Build the F1 predictor image
docker compose build

# Run training pipeline
docker compose up f1-training

# View logs
docker compose logs -f f1-training
```

---

## 🔄 Alternative: Run Without Docker

If Docker installation is problematic, you can run the pipeline directly:

```bash
# Install dependencies
pip install -r requirements.txt

# Run training
python src/f1_prediction_flow.py --mode training

# Run prediction
python src/f1_prediction_flow.py --mode prediction
```

Docker provides:
- ✅ Consistent environment
- ✅ Easy deployment
- ✅ Isolation from system Python

But it's **not required** for development!

---

## 📚 Resources

- **Docker Desktop**: https://docs.docker.com/desktop/install/windows-install/
- **WSL 2**: https://docs.microsoft.com/en-us/windows/wsl/install
- **Docker Docs**: https://docs.docker.com/
- **Troubleshooting**: https://docs.docker.com/desktop/troubleshoot/overview/

---

## 💡 For UK Placement Applications

**If Docker is installed:**
- ✅ Mention containerization in CV/interviews
- ✅ Show Dockerfile and docker-compose.yml
- ✅ Discuss deployment advantages

**If Docker isn't installed:**
- ✅ Still valuable project without Docker
- ✅ Focus on Prefect orchestration
- ✅ Highlight modular Python architecture
- ✅ Mention "Docker-ready" codebase

Both approaches demonstrate strong engineering skills!
