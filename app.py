from sklearn.metrics import confusion_matrix
import numpy as np

cases = ['Baseline (37/63)', 'Case 1 (20/80)', 'Case 2 (50/50)', 'Case 3 (80/20)']
classifiers = ['Logistic Regression', 'SVM', 'Random Forest']

all_preds = {
    'Baseline (37/63)': (y_p_LR, y_p_SVM, y_p_RF),
    'Case 1 (20/80)':   (y_p_LR_case1, y_p_SVM_case1, y_p_RF_case1),
    'Case 2 (50/50)':   (y_p_LR_case2, y_p_SVM_case2, y_p_RF_case2),
    'Case 3 (80/20)':   (y_p_LR_case3, y_p_SVM_case3, y_p_RF_case3)
}

print("=" * 60)
print("STREAMLIT HARDCODED VALUES")
print("=" * 60)

print("\naccuracy = {")
for case, preds in all_preds.items():
    vals = [round(accuracy_score(y_test, p), 4) for p in preds]
    print(f"    '{case}': {{'Logistic Regression': {vals[0]}, 'SVM': {vals[1]}, 'Random Forest': {vals[2]}}},")
print("}")

print("\nsensitivity = {")
for case, preds in all_preds.items():
    vals = []
    for p in preds:
        TN, FP, FN, TP = confusion_matrix(y_test, p).ravel()
        vals.append(round(TP / (TP + FN), 4))
    print(f"    '{case}': {{'Logistic Regression': {vals[0]}, 'SVM': {vals[1]}, 'Random Forest': {vals[2]}}},")
print("}")

print("\nspecificity = {")
for case, preds in all_preds.items():
    vals = []
    for p in preds:
        TN, FP, FN, TP = confusion_matrix(y_test, p).ravel()
        vals.append(round(TN / (TN + FP), 4))
    print(f"    '{case}': {{'Logistic Regression': {vals[0]}, 'SVM': {vals[1]}, 'Random Forest': {vals[2]}}},")
print("}")

print("\nauc = {")
for case, preds in all_preds.items():
    vals = [round(results[clf]['auc'][i], 4) 
            for i, clf in enumerate(classifiers)]
    # fix indexing
    case_idx = list(all_preds.keys()).index(case)
    vals = [round(results[clf]['auc'][case_idx], 4) for clf in classifiers]
    print(f"    '{case}': {{'Logistic Regression': {vals[0]}, 'SVM': {vals[1]}, 'Random Forest': {vals[2]}}},")
print("}")

print("\ncm_data = {")
for case, preds in all_preds.items():
    print(f"    '{case}': {{")
    for clf_name, p in zip(classifiers, preds):
        TN, FP, FN, TP = confusion_matrix(y_test, p).ravel()
        print(f"        '{clf_name}': {{'TN': {TN}, 'FP': {FP}, 'FN': {FN}, 'TP': {TP}}},")
    print("    },")
print("}")

print("\n# Feature stats for Stage 1")
mean_features = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
    'smoothness_mean', 'compactness_mean', 'concavity_mean',
    'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean'
]
benign = df_clean[df_clean['diagnosis'] == 'B']
malignant = df_clean[df_clean['diagnosis'] == 'M']

print("feature_data = {")
for feat in mean_features:
    b_vals = benign[feat].tolist()
    m_vals = malignant[feat].tolist()
    print(f"    '{feat}': {{")
    print(f"        'benign': {b_vals},")
    print(f"        'malignant': {m_vals}")
    print("    },")
print("}")

print("\n# Class distribution")
print(f"class_counts = {{'Benign': {len(benign)}, 'Malignant': {len(malignant)}}}")

print("\n# Training set sizes per case")
print(f"training_sizes = {{")
print(f"    'Baseline (37/63)': {{'Benign': 285, 'Malignant': 170}},")
print(f"    'Case 1 (20/80)':   {{'Benign': 364, 'Malignant': 91}},")
print(f"    'Case 2 (50/50)':   {{'Benign': 227, 'Malignant': 228}},")
print(f"    'Case 3 (80/20)':   {{'Benign': 91, 'Malignant': 364}}")
print(f"}}")

print("\n# Real world impact")
print("us_cases = 310720")
print("global_cases = 2300000")
print("=" * 60)
print("DONE — copy everything above into your Streamlit file")
print("=" * 60)