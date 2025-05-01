# Machine Learning Course (CO3117-HCMUT)

## Project Setup Guide

### Environment Requirements
- **Python Version**: 3.9.x - 3.12.x (Recommended: 3.11)
- **Supported Operating Systems**: 
  - Windows 10/11
  - macOS (Catalina and newer)
  - Linux (Ubuntu 20.04+)

### Development Environment Setup

#### Prerequisites
- Python (3.9-3.12)

#### Setup Steps

##### Windows
1. Install Python from official website
2. Open Command Prompt
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

##### macOS/Linux
```bash
# Install Python via Homebrew (macOS) or package manager (Linux)
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```
## Get the Data

1. **Run the setup script**  
   ```bash
   python setup_dvc_gdrive.py
   ```  
   - Paste the base64 string when prompted.  
   - Follow the printed instruction to `export`/`set` `GOOGLE_APPLICATION_CREDENTIALS`.

2. **Pull all DVC-tracked files**  
   ```bash
   dvc pull -r myremote
   ```  