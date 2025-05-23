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

We trained six different models on the processed review data, tuning their corresponding hyperparameters via grid search and tracking everything with MLflow. Each model was evaluated based on the same metrics: training accuracy, test accuracy, precision, recall, F1 score, and training vs. test gap (to check for overfitting). The MLflow run ID for the best model from each architecture was logged for later retrieval.

Below is a comprehensive comparison of all models, sorted by test accuracy:

| Model                 | Test Acc. | F1     | Precision | Recall | Train Acc. | Gap    |
| :-------------------- | :-------- | :----- | :-------- | :----- | :--------- | :----- |
| Kernel SVM            | **0.8847**| **0.8786** | 0.8811    | **0.8847** | 1.0000     | 0.1153 |
| SVM                   | 0.8807    | 0.8759 | 0.8762    | 0.8807 | 0.9505     | 0.0698 |
| Logistic Regression   | 0.8790    | 0.8734 | **0.8743** | 0.8790 | 0.9468     | 0.0678 |
| Gradient Boosting     | 0.8538    | 0.8398 | 0.8483    | 0.8538 | --         | --     |
| Random Forest         | 0.8111    | 0.7577 | 0.8391    | 0.8111 | --         | --     |
| CRF                   | 0.7680    | 0.7160 | 0.7186    | 0.7680 | 1.0000     | 0.2320 |

### Key Findings

- **Top Performers:** Kernel SVM achieved the highest test accuracy (0.8847) and F1 score (0.8786), closely followed by linear SVM (0.8807) and Logistic Regression (0.8790). The minimal margin (<2%) between these top three models suggests comparable effectiveness for this sentiment classification task.

- **Overfitting Patterns:** Models with perfect training accuracy (Kernel SVM and CRF) exhibited the largest generalization gaps (0.1153 and 0.2320, respectively). Linear SVM and Logistic Regression showed more balanced train-test performance with smaller gaps (0.0698 and 0.0678).

- **Feature Learning Mechanisms:**
    - **Linear models** (SVM, Logistic Regression) excelled through effective regularization and linear separability of TF-IDF features.
    - **Kernel SVM** successfully captured nonlinear sentiment patterns via RBF kernel mapping.
    - **Tree-based methods** (Random Forest, Gradient Boosting) identified semantically meaningful words but generally underperformed linear approaches.
    - **CRF** struggled due to the absence of sequential dependencies in the bag-of-words representation, which is typical for sentiment analysis tasks.

- **Computational Efficiency:** Linear SVM demonstrated superior efficiency, converging in just 100 iterations while maintaining high performance, especially compared to kernel methods or ensemble approaches.

## Team Members and Workload Distribution

| Member              | ID        | Responsibilities                                               |
| :------------------ | :-------- | :-------------------------------------------------------------- |
| Tran Dinh Dang Khoa | 2211649   | Repository setup, DVC/MLflow integration, Logistic Regression model implementation |
| Phan Chi Vy         | 2252938   | Feature engineering, Gradient Boosting, Random Forest models implementation |
| Nguyen Duc Tam      | 2252734   | SVM, Kernel SVM, CRF models implementation |