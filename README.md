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

## MLflow Experiment Tracking

To monitor training experiments and models:
```bash
# Start MLflow tracking server
mlflow ui --backend-store-uri models/experiments
```
Access the dashboard at: http://localhost:5000

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

git add .
git commit -m "<message>"
git push           # Push code changes to GitHub
```

### Important Notes

- Always run `dvc pull` after `git pull` to ensure you have the most up-to-date data
- The `dvc_add.py` script handles tracking files with DVC - don't use `dvc add` directly
- Remember to `dvc push` before `git push` to ensure your team can access the data

## Model Comparison

The project implemented six different models for sentiment analysis: Logistic Regression, Gradient Boosting, Random Forest, Support Vector Machine (SVM), Kernel Support Vector Machine, and Conditional Random Fields (CRF). Below is a summary of the best-performing configurations and their key metrics for the models where detailed information was available:

### Logistic Regression
- **Best Hyperparameters**: C: 5.0, Penalty: l2, Solver: liblinear
- **Test Accuracy**: 0.8790
- **Test F1 Score**: 0.8734
- **Test Precision**: 0.8743
- **Test Recall**: 0.8790
- **Train Accuracy**: 0.9468
- **Analysis**: A higher 'C' value (5.0) minimized regularization, outperforming lower 'C' values despite a noticeable generalization gap. The L2 penalty was effective in preserving predictive signals, and the `liblinear` solver proved robust for this binary classification problem with a medium-sized dataset.

### Gradient Boosting
- **Best Hyperparameters**: n_estimators: 200, learning_rate: 0.2, max_depth: 7
- **Test Accuracy**: 0.8538
- **Test F1 Score**: 0.8398
- **Test Precision**: 0.8483
- **Test Recall**: 0.8538
- **Analysis**: The optimal configuration leveraged a sufficient number of estimators, an aggressive learning rate, and deep trees to capture complex patterns and rich feature interactions essential for sentiment analysis.

### Random Forest
- **Best Hyperparameters**: n_estimators: 200, max_depth: None, min_samples_split: 2
- **Test Accuracy**: 0.8111
- **Test F1 Score**: 0.7577
- **Test Precision**: 0.8391
- **Test Recall**: 0.8111
- **Analysis**: Feature importance analysis highlighted words like "waste", "excellent", "perfect", and "disappointed" as highly influential in sentiment prediction.

*(Note: Detailed performance metrics for Support Vector Machine (SVM), Kernel Support Vector Machine, and Conditional Random Fields (CRF) were not available in the provided PDF content.)*

## Team Members and Workload Distribution

| Name                  | Student ID | Key Contributions                                               |
| :-------------------- | :--------- | :-------------------------------------------------------------- |
| Tran Dinh Dang Khoa   | 2211649    | Repository setup, DVC/MLflow integration, Logistic Regression model implementation |
| Phan Chi Vy           | 2252938    | Feature engineering, Gradient Boosting, Random Forest models implementation |
| Nguyen Duc Tam        | 2252734    | SVM, Kernel SVM, CRF models implementation |