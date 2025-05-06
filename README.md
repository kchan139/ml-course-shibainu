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

## DVC Workflow

This repository uses DVC (Data Version Control) to track large files and datasets. Follow these steps to work with the repository:

### Getting Started

When starting work on the project, always fetch the latest changes:

```bash
git pull           # Get latest code changes
dvc pull           # Get latest data tracked by DVC
```

### Making Changes

After making changes to DVC-tracked folders:

```bash
python dvc_add.py  # Add/update files to DVC tracking
dvc push           # Push data changes to remote storage
git add .          # Stage your changes including DVC metadata
git commit -m "<message>"
git push           # Push code changes to GitHub
```

### Important Notes

- Always run `dvc pull` after `git pull` to ensure you have the most up-to-date data
- The `dvc_add.py` script handles tracking files with DVC - don't use `dvc add` directly
- Remember to `dvc push` before `git push` to ensure your team can access the data
```