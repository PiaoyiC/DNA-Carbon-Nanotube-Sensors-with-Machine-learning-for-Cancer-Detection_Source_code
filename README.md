# Integrating Structurally Defined DNA-Carbon Nanotube Sensors with Machine-learning for Cancer Detection

This repository contains code and data files for reproducing the machine learning model training and testing results presented in Piaoyi Chen, Xin Zheng, Yinong Li, Jian Li, Xuan Zhou, Yawei Wen, Jialong Liu, Pengbo Wang, Xiaohui Li, Runzhe Chen, Zhiwei Lin (2025).

## Overview

This project implements machine learning-based cancer detection using structurally defined DNA-single chirality Carbon Nanotube (DNA-scCNT) fluorescence features:

1. **Multicancer Detection**: Multi-class classification for liver cancer (LC), lung cancer (LuC), ovarian cancer (OC), and non-cancer samples (N)
2. **Early Cancer Detection**: Detection for early-stage lung cancer (Early LuC vs N)

## Datasets

The repository includes three data folders containing Excel files:

- **Multicancer detection-Data/**: Training and testing data for multi-class cancer detection (LC, LuC, OC, N)
- **Early cancer detection-Data/**: Training and testing data for early-stage lung cancer detection (Early LuC vs N)
- **Mechanism analysis-Data/**: Data for mechanism analysis between DNA-scCNT features and tumor markers

## Project Architecture

```
Source code for submission/
│
├── Multicancer detection/
│   ├── 01_Algorithm assessment/
│   │   └── Overall Assessment.py                  # Comparison of 7 algorithms (Bayesian Optimization)
│   │
│   ├── 02_Algorithm optimization/
│   │   ├── RF.py                                  # Random Forest optimization (Bayesian Optimization)
│   │   ├── XGB.py                                 # XGBoost optimization (Bayesian Optimization)
│   │   └── ANN.py                                 # Artificial Neural Network optimization
│   │
│   ├── 03_Ensemble/
│   │   ├── AdvancedFeatureInteractionsRF.py       # [Component] Feature engineering for RF (Boruta selection)
│   │   ├── AdvancedFeatureInteractionsXGB.py      # [Component] Feature engineering for XGB (Mutual Info selection)
│   │   ├── Ensemble_RF+XGB_CV.py                  # Main training with Genetic Algorithm (GA)
│   │   ├── Ensemble_RF+XGB_CV_Original_Feature.py # Main training using original features (no feature engineering)
│   │   ├── Ensemble_RF+XGB_External validation.py # External validation
│   │   ├── Feature Permutation Importance.py      # Permutation-based feature importance
│   │   └── SHAP Feature importance.py             # SHAP-based feature importance
│   │
│   └── 04_Mechanism analysis/
│       ├── 01_Spearman Correlation analysis.py    # DNA-scCNT features self-correlation
│       ├── 02-1_Mantel_CNTs self correlation.py   # Mantel test for DNA-scCNT features
│       └── 02-2_Mantel Correlation analysis.py    # DNA-scCNT features vs tumor markers
│
└── Early cancer detection/
    ├── Overall Assessment_Early.py                # Comparison of multiple algorithms
    ├── RF+LDA_Early cancer detection_CV.py        # RF+LDA ensemble training (Optuna TPE)
    └── RF+LDA_Early cancer detection_External validation.py  # External validation
```

## Installation

**Requirements**: Python 3.10, Anaconda/Miniconda (recommended)

```bash
# Create environment
conda create -n cancer_detection python=3.10
conda activate cancer_detection

# Install dependencies
pip install -r requirements.txt
```

## Usage

All file paths (input data and output directories) are automatically resolved relative to each script's location. No manual path configuration is required.

If you use *Code Ocean*, simply set the corresponding script in the code directory as 'Set as file to run' and click the 'Reproducible Run' button for one-click reproducibility.

### Multicancer Detection

```bash
# 1. Algorithm Assessment
cd "Multicancer detection/01_Algorithm assessment"
python "Overall Assessment.py"

# 2. Individual Model Optimization
cd "../02_Algorithm optimization"
python RF.py
python XGB.py
python ANN.py

# 3. Ensemble Training
cd "../03_Ensemble"
python "Ensemble_RF+XGB_CV.py"

# 4. External Validation
python "Ensemble_RF+XGB_External validation.py"

# 5. Feature Importance Analysis
python "Feature Permutation Importance.py"
python "SHAP Feature importance.py"

# 6. Mechanism Analysis
cd "../04_Mechanism analysis"
python "01_Spearman Correlation analysis.py"
python "02-2_Mantel Correlation analysis.py"
```

### Early Cancer Detection

```bash
cd "Early cancer detection"
python "Overall Assessment_Early.py"
python "RF+LDA_Early cancer detection_CV.py"
python "RF+LDA_Early cancer detection_External validation.py"
```

## Data Format

### Input (Excel files)

**Multicancer Detection:**
- Column 1: Sample labels ('LC', 'LuC', 'OC', 'N')
- Columns 2-14: DNA-scCNT fluorescence features
  - `dint(9,1)`, `dint(8,3)`, `dint(7,5)`, `dint(7,3)`, `dint(6,5)`, `dint(6,4)`
  - `dwl(9,1)`, `dwl(8,3)`, `dwl(7,5)`, `dwl(7,3)`, `dwl(6,5)`, `dwl(6,4)`

**Early Cancer Detection:**
- Column 1: Sample labels ('LuC', 'N')
- Columns 2-14: Same DNA-scCNT fluorescence features

**Mechanism Analysis:**
- DNA-scCNT features + tumor markers (AFP, CEA, CA199, CA153, NSE, CyFra21-1, SCC, HCG-β, proGRP, HE4, CA-125, CA-724)

### Output

Each script generates the following output files:

- **Excel files (.xlsx)**:
  - Cross-validation results per fold
  - Summary statistics (mean ± std)
  - Model hyperparameters (RF, XGB, LDA parameters)
  - External validation results
  - Feature importance scores (Permutation, SHAP)
  - Mechanism analysis results (Spearman, Mantel test with p-values and FDR correction)

- **Visualizations (.svg, .png)**:
  - ROC curves with confidence intervals (training and external validation)
  - Confusion matrices (training and external validation)
  - Feature importance bar plots
  - SHAP summary plots and force plots
  - Mechanism analysis heatmaps

*Note: Output paths can be customized in the `main()` function of each script by modifying the `output_path` parameter.*

## Key Algorithms

### Machine Learning Algorithms
- **Multicancer Detection**: RF + XGBoost ensemble
- **Early Cancer Detection**: RF + LDA ensemble

**Objective Function:**

*Algorithm Assessment (01):*
```
Fitness = F1 Score
```

*Optimization (02) and Ensemble (03):*
```
Fitness = a × F1 + b × AUC + c × Recall + d × Specificity
```
*Note: Weights (a, b, c, d) can be adjusted based on specific requirements*

### Hyperparameter Optimization Methods

**Bayesian Optimization:**
- For algorithm assessment and individual model optimization (RF, XGB, ANN)
- Uses Gaussian Process to model the objective function

**Genetic Algorithm (GA):**
- For RF+XGB ensemble optimization in multicancer detection
- Components: Simulated Binary Crossover (SBX), adaptive mutation, elite preservation, tournament selection
- Optimizes F1 score as the primary fitness metric

**Optuna TPE (Tree-structured Parzen Estimator):**
- For RF+LDA ensemble optimization in early cancer detection
- Adaptive sampling strategy for efficient hyperparameter search

### Feature Engineering
- **Gaussian Kernel Interactions**: Pairwise RBF kernels, weighted combinations, distance features, local density estimation
- **Feature Selection**:
  - Boruta (all-relevant feature selection for RF)
  - Mutual Information (information-theoretic selection for XGB)
  - Permutation Importance
  - SHAP (Shapley Additive Explanations)

### Data Augmentation
- **SMOTE** (Synthetic Minority Over-sampling Technique) for Random Forest
- **ADASYN** (Adaptive Synthetic Sampling) for XGBoost

### Ensemble Strategy
- Weighted average of model prediction probabilities
- Hyperparameter of ensemble model optimization via GA or Optuna
- Dynamic threshold selection per class for optimal F1 score

## Contact

For questions or issues, please contact the corresponding author.

