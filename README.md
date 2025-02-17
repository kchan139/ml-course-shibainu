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

### Troubleshooting

#### Common Dependency Issues
- Ensure matching Python versions across team
- Use `pip freeze > requirements.txt` to capture exact versions
- Check architecture compatibility (x86/ARM)

#### Potential Resolution Steps
1. Update pip: `python -m pip install --upgrade pip`
2. Install build tools:
   - Windows: Visual C++ Build Tools
   - macOS: Xcode Command Line Tools
   - Linux: `build-essential` package

### Workflow
- Always activate virtual environment before development
- Update `requirements.txt` when adding new dependencies
- Use consistent Python version across team

### Project Structure
```
kchan139-ml-course-shibainu/
├── dataset/       # Raw and processed data
├── homework/      # Weekly homework solutions
├── models/        # Trained models
├── notebooks/     # Jupyter notebooks
├── reports/       # Project documentation
├── src/           # Source code
└── test/          # Unit tests
```

### Contributing

##### I. Ensure Your Fork and Branch are Up-to-Date

1. Add the original repository as a remote (if not done earlier):
```bash
   git remote add upstream https://github.com/kchan139/ml-course-shibainu
```
2. Fetch the latest changes:
```bash
   git fetch upstream
```
3. Switch to your local `develop` branch and merge the changes:
```bash
   git checkout develop
   git merge upstream/develop
```
4. Push the updated `develop` to your forked repository:
```bash
   git push origin develop
```

##### II. Create a Model Branch
1. Create and switch to a new branch (if not done earlier):
```bash
   git checkout -b model/your-model
```
2. Make changes
3. Stage and commit your changes:
```bash
   git add .
   git commit -m "desciption of your changes"
```
4. Push the feature branch to your fork:
```bash
   git push origin model/your-model
```

##### III. Create a Pull Request on GitHub


### License
MIT License
---