# Code Modification History

## Overview
This document records all modifications made to the machine learning algorithm files (RF.py, XGB.py, ANN.py) in the `02_Algorithm optimization` folder.

---

## Modification List

### 1. Function Renaming
**Requirement**: Remove "global" from function names to simplify naming convention

**Changes**:
- `apply_global_adasyn` → `apply_adasyn` (in RF.py and XGB.py)
- `apply_global_smote` → `apply_smote` (in ANN.py)
- Updated all function calls to use the new names

---

### 2. Remove Confusion Matrix Plots
**Requirement**: Remove per-fold confusion matrix image generation

**Changes**:
- Removed confusion matrix plotting code from the `run()` method in all three files
- Deleted the loop that generated confusion matrix images for each fold

---

### 3. Standardize Excel Output
**Requirement**: Unify Excel output structure across all algorithms

**Changes**:
- **Removed**: `Feature_Importance` sheet from Excel output (all three files)
- **Removed**: Model saving functionality (`save_models()` calls)
- **Added**: `ROC_data` sheet containing:
  - Individual fold ROC curves (FPR, TPR, AUC for each fold)
  - Data arranged in columns: Fold1 (FPR, TPR, AUC), Fold2 (FPR, TPR, AUC), etc.
- **Added**: `mean_ROC` sheet containing:
  - Mean ROC curve data with standard deviation
  - Columns: FPR, Mean_TPR, Std_TPR, TPR_Upper, TPR_Lower
  - Uses 500 interpolation points from 0 to 1

---

### 4. Simplify Logging Output
**Requirement**: Drastically reduce logging to only essential information

**Kept Logging** (only these items):
1. Algorithm name and feature configuration:
   - Feature interaction method
   - Feature selection method: boruta
2. Data information:
   - Data shape: X samples, Y features (direct output without "Loading data...")
3. Class processing:
   - "Processing class: [class_name]"
4. Performance metrics (single line per class):
   - F1, AUC, Precision, Recall, Sensitivity, Specificity
5. Completion message:
   - "Training completed successfully"

**Removed Logging**:
- Configuration summary
- Data path information
- Detailed ADASYN/SMOTE initialization messages
- Per-fold detailed results
- Feature selection progress messages
- Hyperparameter optimization details
- Model saving messages
- All other verbose logging statements

---

### 5. Excel Formatting
**Requirement**: Format all Excel output files with consistent styling

**Changes**:
- **Added imports**:
  - `from openpyxl import load_workbook`
  - `from openpyxl.styles import Font`
- **Created function**: `format_excel_file(self, file_path)`
  - Sets all cell fonts to Times New Roman (11pt)
  - Sets header row (first row) to bold
  - Applies formatting to all sheets in the workbook
- **Integration**: Called `self.format_excel_file(output_path)` at the end of `save_results()` method

---

## Technical Notes

### Function Placement (Python)
- Functions must be defined before they are called
- Functions can be placed anywhere in the class/module as long as they're defined before use
- The `weighted_specificity` function doesn't need to be at the beginning of the file

### Specificity Metric
- Unlike F1, sensitivity, and recall, scikit-learn does not have a built-in specificity function
- Requires custom implementation: `weighted_specificity(y_true, y_pred, labels)`
- Cannot use `average='weighted'` parameter like other metrics

### ROC Curve Calculation
- Mean ROC curve uses interpolation across 500 points
- Ensures all curves start at (0,0) and end at (1, TPR_final)
- Calculates mean ± std for confidence intervals
- Clips TPR_Upper at 1.0 and TPR_Lower at 0.0

---

## Files Modified
- `RF.py` (Random Forest with ADASYN)
- `XGB.py` (XGBoost with ADASYN)
- `ANN.py` (MLPClassifier with SMOTE)

All files located in: `Multicancer detection\02_Algorithm optimization\`

---

## Validation
All files have been syntax-checked and compile successfully using:
```bash
python -m py_compile RF.py
python -m py_compile XGB.py
python -m py_compile ANN.py
```

---

*Document created: 2025-10-27*
*Purpose: Context preservation for future reference*
