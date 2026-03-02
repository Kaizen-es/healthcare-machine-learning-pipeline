# Healthcare Machine Learning Pipeline
End-to-end ML pipeline for breast cancer classification using Wisconsin clinical datasets 

Implementation:
## Overview
This project presents an empirical analysis of how class distribution affects the performance of supervised learning classifiers on breast cancer diagnostic data. Rather than correcting for class imbalance — a common approach in the literature — this study deliberately varies the malignant-to-benign ratio across three controlled scenarios to observe and visualize how classifier behavior shifts under different distributional conditions.
The analysis is built around the Wisconsin Breast Cancer Diagnostic Dataset and implemented as an interactive visualization dashboard using Plotly and Streamlit.

Motivation
In real clinical settings, approximately 80% of confirmed breast tissue diagnoses are benign and only 20% are malignant. Yet most machine learning studies train and evaluate classifiers on artificially balanced datasets that do not reflect this real-world distribution.
This raises an important and underexplored question:

How does the balance of malignant versus benign cases in training data affect how well a classifier performs — and which classifier is most robust to that shift?

This project addresses that question directly through a controlled distribution experiment, evaluating classifier performance across accuracy, sensitivity, and specificity under three distinct class distribution scenarios.

Dataset
Wisconsin Breast Cancer Diagnostic Dataset

Source: UCI Machine Learning Repository
Citation: Wolberg, W.H., Street, W.N., & Mangasarian, O.L. (1995)
Link: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

PropertyValueSamples569Features30 (numerical)ClassesMalignant (M), Benign (B)Class Distribution37% malignant / 63% benignMissing ValuesNone
Features describe cell nucleus characteristics — size, texture, smoothness, compactness, shape irregularity, symmetry, and fractal dimension — each measured as mean, standard error, and worst value across cells in each digitized microscopic image.

Project Structure
classifier-robustness-under-distribution-shift/
│
├── data/
│   └── data.csv                  # Wisconsin Breast Cancer Dataset
│
├── notebooks/
│   └── analysis.ipynb            # Full analysis notebook (Google Colab)
│
├── app/
│   └── app.py                    # Streamlit dashboard
│
├── visualizations/
│   └── (exported chart images)   # Key visualization outputs
│
├── README.md
└── requirements.txt

Methodology
The project is structured in three stages:
Stage 1 — Feature Exploration
Exploratory data analysis of all 30 features across malignant and benign classes. Visualizations include:

Feature distribution histograms (malignant vs benign)
Box plots highlighting median values and outliers
Correlation heatmap across all features
Scatter plots of key feature pairs colored by diagnosis

Stage 2 — Classification
Two classifiers are trained and evaluated on the original dataset distribution:

Support Vector Machine (SVM) — non-parametric, finds optimal decision boundary with maximum margin
Logistic Regression — parametric baseline, outputs class probabilities

Training/testing split: 80% / 20%
Performance is evaluated using:

Accuracy
Sensitivity (recall for malignant class)
Specificity (recall for benign class)
Confusion Matrix

Stage 3 — Distribution Experiment
Both classifiers are retrained and evaluated across three controlled class distribution scenarios:
CaseMalignantBenignDescriptionCase 120%80%Mirrors real-world clinical distributionCase 250%50%Ideally balanced — common in researchCase 380%20%Flipped distribution — stress test
Distribution is controlled via undersampling the majority class. Performance metrics are compared across all three cases to identify which classifier is most robust to distribution shift.

Results
To be updated upon project completion.

Interactive Dashboard
The project includes a Streamlit dashboard that allows users to:

Explore feature distributions interactively
View classifier performance metrics and confusion matrices
Switch between the three distribution scenarios in real time
Compare SVM and Logistic Regression side by side

To run the dashboard locally:
bashpip install -r requirements.txt
streamlit run app/app.py

Limitations

Results are specific to the Wisconsin dataset and may not generalize to other breast cancer datasets or clinical populations
Undersampling to achieve target distributions reduces training sample size, which may affect the reliability of performance estimates
Accuracy alone is an insufficient metric for imbalanced medical classification — sensitivity and specificity must be interpreted together
This is a controlled experimental analysis and should not be interpreted as a clinical diagnostic tool


Requirements
pandas
numpy
scikit-learn
plotly
streamlit
matplotlib
seaborn
Install all dependencies:
bashpip install -r requirements.txt

References

Wolberg, W.H., Street, W.N., & Mangasarian, O.L. (1995). Wisconsin Breast Cancer Diagnostic Dataset. UCI Machine Learning Repository. https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
Breast Cancer Research Foundation. (2024). What percentage of breast biopsies are cancer? https://www.bcrf.org/blog/what-percentage-of-breast-biopsies-are-cancer/
Pedregosa et al. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12, 2825–2830.
Streamlit Documentation. https://docs.streamlit.io
Plotly Python Documentation. https://plotly.com/python


Course Context
Developed as a final project for EECE 5642 Data Visualization, Northeastern University, under Prof. Y. Raymond Fu.
