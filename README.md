# Classifier Robustness Under Distribution Shift
### Empirical Analysis of Class Imbalance Effects on Breast Cancer Classification


## Overview

This project presents an empirical analysis of how class distribution affects the performance of supervised learning classifiers on breast cancer diagnostic data. Rather than correcting for class imbalance — a common approach in the literature — this study deliberately varies the malignant-to-benign ratio across three controlled scenarios to observe and visualize how classifier behavior shifts under different distributional conditions.

The analysis is built around the Wisconsin Breast Cancer Diagnostic Dataset and implemented as an interactive visualization dashboard using Plotly and Streamlit.


## Motivation

In real clinical settings, approximately 80% of confirmed breast tissue diagnoses are benign and only 20% are malignant. Yet most machine learning studies train and evaluate classifiers on artificially balanced datasets that do not reflect this real-world distribution.

> **How does the balance of malignant versus benign cases in training data affect how well a classifier performs — and which classifier is most robust to that shift?**

This project addresses that question directly through a controlled distribution experiment, evaluating classifier performance across accuracy, sensitivity, and specificity under three distinct class distribution scenarios.

## Dataset

**Wisconsin Breast Cancer Diagnostic Dataset**
- Source: UCI Machine Learning Repository
- Citation: Wolberg, W.H., Street, W.N., & Mangasarian, O.L. (1995)
- Link: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

| Property | Value |
|----------|-------|
| Samples | 569 |
| Features | 30 (numerical) |
| Classes | Malignant (M), Benign (B) |
| Class Distribution | 37% malignant / 63% benign |
| Missing Values | None |


## Project Structure

## Methodology
### Stage 1 — Feature Exploration
- Feature distribution histograms (malignant vs benign)
- Box plots highlighting median values and outliers
- Correlation heatmap across all features
- Scatter plots of key feature pairs colored by diagnosis

### Stage 2 — Classification
- **Support Vector Machine (SVM)**
- **Logistic Regression**

Training/testing split: 80% / 20%

Metrics:
- Accuracy
- Sensitivity
- Specificity
- Confusion Matrix

### Stage 3 — Distribution Experiment

| Case | Malignant | Benign | Description |
|------|-----------|--------|-------------|
| Case 1 | 20% | 80% | Mirrors real-world clinical distribution |
| Case 2 | 50% | 50% | Ideally balanced |
| Case 3 | 80% | 20% | Flipped distribution — stress test |


## Results

*To be updated upon project completion.*


## Limitations

- Results specific to Wisconsin dataset
- Undersampling reduces training sample size
- Accuracy alone insufficient — sensitivity and specificity must be interpreted together
- Experimental analysis only — not a clinical diagnostic tool


## References

- Wolberg et al. (1995). Wisconsin Breast Cancer Diagnostic Dataset. UCI ML Repository.
- BCRF (2024). What percentage of breast biopsies are cancer?
- Pedregosa et al. (2011). Scikit-learn. JMLR 12, 2825-2830.

