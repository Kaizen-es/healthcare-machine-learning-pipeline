---

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

---

## Results

*To be updated upon project completion.*

---

## Limitations

- Results specific to Wisconsin dataset
- Undersampling reduces training sample size
- Accuracy alone insufficient — sensitivity and specificity must be interpreted together
- Experimental analysis only — not a clinical diagnostic tool

---

## References

- Wolberg et al. (1995). Wisconsin Breast Cancer Diagnostic Dataset. UCI ML Repository.
- BCRF (2024). What percentage of breast biopsies are cancer?
- Pedregosa et al. (2011). Scikit-learn. JMLR 12, 2825-2830.

---

## Course Context

EECE 5642 Data Visualization, Northeastern University, Prof. Y. Raymond Fu.
