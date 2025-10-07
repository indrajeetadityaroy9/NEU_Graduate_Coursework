# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a statistical modeling project focused on **spline regression and classification** using Python. The main work is contained in a Jupyter notebook (`HW3_Lab2Spline.ipynb`) that demonstrates various spline techniques applied to health-related datasets.

## Development Commands

**Running the notebook:**
```bash
jupyter notebook HW3_Lab2Spline.ipynb
# or
jupyter lab HW3_Lab2Spline.ipynb
```

**Installing dependencies:**
```bash
pip install pandas numpy matplotlib scikit-learn statsmodels pygam csaps
```

## Project Structure

The repository contains:
- `HW3_Lab2Spline.ipynb` - Main notebook with two major sections
- `fev.csv` - FEV (Forced Expiratory Volume) dataset (654 observations)
- `Heart.csv` - Coronary heart disease dataset (referenced but not present)

## Notebook Architecture

### Section 1: Regression Splines (cells 1-18)
Analyzes FEV data with `height` as predictor for `fev`:
- Linear regression baseline
- Natural cubic splines (df=5, df=10)
- Smoothing cubic splines (cross-validated lambda)
- Cubic B-splines (df=5, df=10)
- Cross-validated MSE comparison across methods

**Key libraries:** `statsmodels`, `scikit-learn`, `scipy`

### Section 2: Classification with GAMs (cells 19+)
Logistic regression with GAM splines for coronary heart disease (`chd`):
- Predictors: `tobacco`, `sbp` (systolic blood pressure), `age`
- Uses PyGAM library for GAM logistic models
- Includes AUC ROC evaluation and comparison with linear logistic regression

**Key library:** `pygam`

## Data Details

**fev.csv columns:**
- `id`: Patient identifier
- `age`: Age in years
- `fev`: Forced expiratory volume (target for regression)
- `height`: Height measurement (primary predictor)
- `sex`: Patient sex
- `smoke`: Smoking status

**Heart.csv columns** (expected):
- `chd`: Coronary heart disease status (0/1, target for classification)
- `sbp`, `tobacco`, `age`: Risk factor predictors
- Additional features: `ldl`, `adiposity`, `famhist`, `typea`, `obesity`, `alcohol`

## Working with This Codebase

When modifying spline models:
- The notebook has multiple empty cells (cells 7-16, 23-26) likely intended for implementing the different spline techniques
- Natural cubic splines typically require `statsmodels` or manual basis function construction
- B-splines are available through `scipy.interpolate.BSpline` or `statsmodels`
- Smoothing splines can use `csaps` library or `scipy.interpolate.UnivariateSpline`
- PyGAM uses `s()` terms to specify spline features in formula syntax

Cross-validation should use `sklearn.model_selection.cross_val_score` or custom K-fold splits with MSE/AUC metrics.
