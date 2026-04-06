import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde
import numpy as np

# ── Page Config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Breast Cancer Classification Dashboard",
    page_icon="🔬",
    layout="wide"
)

# ── Hardcoded Data ────────────────────────────────────────────────────────
accuracy = {
    'Baseline (37/63)': {'Logistic Regression': 0.9825, 'SVM': 0.9737, 'Random Forest': 0.9561},
    'Case 1 (20/80)':   {'Logistic Regression': 0.9649, 'SVM': 0.9561, 'Random Forest': 0.9474},
    'Case 2 (50/50)':   {'Logistic Regression': 0.9737, 'SVM': 0.9649, 'Random Forest': 0.9561},
    'Case 3 (80/20)':   {'Logistic Regression': 0.9123, 'SVM': 0.9035, 'Random Forest': 0.9211},
}
sensitivity = {
    'Baseline (37/63)': {'Logistic Regression': 1.0,    'SVM': 0.9762, 'Random Forest': 0.9286},
    'Case 1 (20/80)':   {'Logistic Regression': 0.9286, 'SVM': 0.9048, 'Random Forest': 0.8571},
    'Case 2 (50/50)':   {'Logistic Regression': 1.0,    'SVM': 0.9762, 'Random Forest': 0.9524},
    'Case 3 (80/20)':   {'Logistic Regression': 1.0,    'SVM': 1.0,    'Random Forest': 1.0},
}
specificity = {
    'Baseline (37/63)': {'Logistic Regression': 0.9722, 'SVM': 0.9722, 'Random Forest': 0.9722},
    'Case 1 (20/80)':   {'Logistic Regression': 0.9861, 'SVM': 0.9861, 'Random Forest': 1.0},
    'Case 2 (50/50)':   {'Logistic Regression': 0.9583, 'SVM': 0.9583, 'Random Forest': 0.9583},
    'Case 3 (80/20)':   {'Logistic Regression': 0.8611, 'SVM': 0.8472, 'Random Forest': 0.875},
}
auc = {
    'Baseline (37/63)': {'Logistic Regression': 0.9977, 'SVM': 0.9974, 'Random Forest': 0.9934},
    'Case 1 (20/80)':   {'Logistic Regression': 0.9964, 'SVM': 0.9947, 'Random Forest': 0.975},
    'Case 2 (50/50)':   {'Logistic Regression': 0.9983, 'SVM': 0.997,  'Random Forest': 0.9931},
    'Case 3 (80/20)':   {'Logistic Regression': 0.9983, 'SVM': 0.9931, 'Random Forest': 0.9919},
}
cm_data = {
    'Baseline (37/63)': {
        'Logistic Regression': {'TN': 70, 'FP': 2, 'FN': 0, 'TP': 42},
        'SVM':                 {'TN': 70, 'FP': 2, 'FN': 1, 'TP': 41},
        'Random Forest':       {'TN': 70, 'FP': 2, 'FN': 3, 'TP': 39},
    },
    'Case 1 (20/80)': {
        'Logistic Regression': {'TN': 71, 'FP': 1, 'FN': 3, 'TP': 39},
        'SVM':                 {'TN': 71, 'FP': 1, 'FN': 4, 'TP': 38},
        'Random Forest':       {'TN': 72, 'FP': 0, 'FN': 6, 'TP': 36},
    },
    'Case 2 (50/50)': {
        'Logistic Regression': {'TN': 69, 'FP': 3, 'FN': 0, 'TP': 42},
        'SVM':                 {'TN': 69, 'FP': 3, 'FN': 1, 'TP': 41},
        'Random Forest':       {'TN': 69, 'FP': 3, 'FN': 2, 'TP': 40},
    },
    'Case 3 (80/20)': {
        'Logistic Regression': {'TN': 62, 'FP': 10, 'FN': 0, 'TP': 42},
        'SVM':                 {'TN': 61, 'FP': 11, 'FN': 0, 'TP': 42},
        'Random Forest':       {'TN': 63, 'FP': 9,  'FN': 0, 'TP': 42},
    },
}
feature_data = {
    'radius_mean':            {'benign': [13.54,13.08,9.504,13.03,8.196,12.05,13.49,11.76,13.64,11.94,11.52,13.05,8.618,10.17,8.598,9.173,9.465,11.31,9.029,12.78,8.888,12.31,13.53,12.86,11.45,13.34,12.0,12.36,14.64,14.62,13.27,13.45,12.18,9.787,11.6,6.981,12.18,9.876,10.49,11.64,12.36,11.34,9.777,12.63,14.26,10.51,8.726,11.93,8.95,11.41,14.5,13.37,13.85,15.1,12.19,15.71,11.71,11.43,11.28,9.738,11.43,12.9,10.75,11.9,14.95,14.44,13.74,13.0,8.219,9.731,11.15,13.15,12.25,16.84,12.06,10.9,11.75,12.34,14.97,10.8,14.97,12.32,11.08,10.66,8.671,9.904,13.01,12.81,11.41,10.08,11.71,11.81,12.3,12.77,9.72,12.91,12.23,12.47,9.876,13.11],'malignant': [17.99,20.57,19.69,11.42,20.29,12.45,18.25,13.71,13.0,12.46,16.02,15.78,19.17,15.85,13.73,14.54,14.68,16.13,19.81,15.34,21.16,16.65,17.14,14.58,18.61,15.3,17.57,18.63,11.84,17.02,19.27,16.13,16.74,14.25,14.99,13.48,13.44,10.95,19.07,13.28,13.17,18.65,13.17,18.22,15.1,19.21,14.71,14.25,12.68,14.78,18.94,17.2,13.8,16.07,18.05,20.18,25.22,19.1,18.46,14.48,19.02,15.37,15.06,20.26,14.42,13.61,13.11,22.27,14.87,15.78,17.95,18.66,24.25,13.61,19.0,19.79,15.46,16.16,18.45,12.77,14.95,16.11,11.8,17.68,19.19,19.59,23.27,16.78,17.47,13.43,15.46,16.46,27.22,21.09,15.7,15.28,18.31,14.22,12.34,14.86]},
    'texture_mean':           {'benign': [14.36,15.71,12.44,18.42,16.84,14.63,22.3,21.6,16.34,18.24,18.75,19.31,11.79,14.88,20.98,13.86,21.01,19.04,17.33,16.49,14.64,16.52,10.94,18.0,20.97,15.86,15.65,21.8,15.24,24.02,14.76,18.3,17.84,19.94,12.84,13.43,20.52,19.4,19.29,18.33,18.54,21.26,16.99,20.76,19.65,20.19,15.83,21.53,15.76,10.82,10.89,16.39,17.21,16.39,13.29,13.93,16.67,15.39,13.39,11.97,17.31,15.92,14.97,14.65,18.77,15.18,17.91,20.78,20.7,15.34,13.08,15.34,17.94,19.46,12.74,12.96,20.18,22.22,19.76,9.71,16.95,12.39,14.71,15.15,14.45,18.06,22.22,13.06,14.92,15.11,17.19,17.39,15.9,21.41,18.22,16.33,19.56,18.6,17.27,22.54],'malignant': [10.38,17.77,21.25,20.38,14.34,15.7,19.98,20.83,21.82,24.04,23.24,17.89,24.8,23.95,22.61,27.54,20.13,20.68,22.15,14.26,23.04,21.38,16.4,21.53,20.25,25.27,15.05,25.11,18.7,23.98,26.47,17.88,21.59,21.72,25.2,20.82,21.58,21.35,24.81,20.28,21.81,17.6,18.66,18.7,22.02,18.57,21.59,22.15,23.84,23.94,21.31,24.52,15.79,19.65,16.15,23.97,24.91,26.29,18.52,21.46,24.59,22.76,19.83,23.03,19.77,24.98,15.56,19.67,16.67,22.91,20.01,17.12,20.2,24.69,18.91,25.12,19.48,21.54,21.91,22.47,17.57,18.05,16.58,20.74,15.94,18.15,22.04,18.8,24.68,19.63,11.89,20.11,21.87,26.57,20.31,22.41,18.58,23.12,26.86,23.21]},
    'perimeter_mean':         {'benign': [87.46,85.63,60.34,82.61,51.71,78.04,86.91,74.72,87.21,75.71,73.34,82.61,54.34,64.55,54.66,59.2,60.11,71.8,58.79,81.37,58.79,79.19,87.91,83.19,73.81,86.49,76.95,79.78,95.77,94.57,84.74,86.6,77.79,62.11,74.34,43.79,77.22,63.95,67.41,75.17,79.01,72.48,62.5,82.15,97.83,68.64,55.84,76.53,58.74,73.34,94.28,86.1,88.44,99.58,79.08,102.0,74.72,73.06,73.0,61.24,73.66,83.74,68.26,78.11,97.84,93.97,88.12,83.51,53.27,63.78,70.87,85.31,78.27,108.4,76.84,68.69,76.1,79.85,95.5,68.77,96.22,78.85,70.21,67.49,54.42,64.6,82.01,81.29,73.53,63.76,74.68,75.27,78.83,82.02,60.73,82.53,78.54,81.09,62.92,87.02],'malignant': [122.8,132.9,130.0,77.58,135.1,82.57,119.6,90.2,87.5,83.97,102.7,103.6,132.4,103.7,93.6,96.73,94.74,108.1,130.0,102.5,137.2,110.0,116.0,97.41,122.1,102.4,115.0,124.8,77.93,112.8,127.9,107.0,110.1,93.63,95.54,88.4,86.18,71.9,128.3,87.32,85.42,123.7,85.98,120.3,97.26,125.5,95.55,96.42,82.69,97.4,123.6,114.2,90.43,104.1,120.2,143.7,171.5,129.1,121.1,94.25,122.0,100.2,100.3,132.4,94.48,88.05,87.21,152.8,98.64,105.7,114.2,121.4,166.2,87.76,123.4,130.4,101.7,106.2,120.2,81.72,96.85,105.1,78.99,117.4,126.3,130.7,152.1,109.3,116.1,85.84,102.5,109.3,182.1,142.7,101.2,98.92,118.6,94.37,81.15,100.4]},
    'area_mean':              {'benign': [566.3,520.0,273.9,523.8,201.9,449.3,561.0,427.9,571.8,437.6,409.0,527.2,224.5,311.9,221.8,260.9,269.4,394.1,250.5,502.5,244.0,470.9,559.2,506.3,401.5,520.0,443.3,466.1,651.9,662.7,551.7,555.1,451.1,294.5,412.6,143.5,458.7,298.3,336.1,412.5,466.7,396.5,290.2,480.4,629.9,334.2,230.9,438.6,245.2,403.3,640.7,553.5,588.7,674.5,455.8,761.7,423.6,399.8,384.8,288.5,398.0,512.2,355.3,432.8,689.5,640.1,585.0,519.4,203.9,300.2,381.9,538.9,460.3,880.2,448.6,366.8,419.8,464.5,690.2,357.6,685.9,464.1,372.7,349.6,227.2,302.4,526.4,508.8,402.0,317.5,420.3,428.9,463.7,507.4,288.1,516.4,461.0,481.9,295.4,529.4],'malignant': [1001.0,1326.0,1203.0,386.1,1297.0,477.1,1040.0,577.9,519.8,475.9,797.8,781.0,1123.0,782.7,578.3,658.8,684.5,798.8,1260.0,704.4,1404.0,904.6,912.7,644.8,1094.0,732.4,955.1,1088.0,440.6,899.3,1162.0,807.2,869.5,633.0,698.8,559.2,563.0,371.1,1104.0,545.2,531.5,1076.0,534.6,1033.0,712.8,1152.0,656.9,645.7,499.0,668.3,1130.0,929.4,584.1,817.7,1006.0,1245.0,1878.0,1132.0,1075.0,648.2,1076.0,728.2,705.6,1264.0,642.5,582.7,530.2,1509.0,682.5,782.6,982.0,1077.0,1761.0,572.6,1138.0,1192.0,748.9,809.8,1075.0,506.3,678.1,813.0,432.0,963.7,1157.0,1214.0,1686.0,886.3,984.6,565.4,736.9,832.9,2250.0,1311.0,766.6,710.6,1041.0,609.9,477.4,671.4]},
    'smoothness_mean':        {'benign': [0.09779,0.1075,0.1024,0.08983,0.086,0.1031,0.08752,0.08637,0.07685,0.08261,0.09524,0.0806,0.09752,0.1134,0.1243,0.07721,0.1044,0.08139,0.1066,0.09831,0.09783,0.09172,0.1291,0.09934,0.1102,0.1078,0.09723,0.08772,0.1132,0.08974,0.07355,0.1022,0.1045,0.1024,0.08983,0.117,0.08013,0.1005,0.09989,0.1142,0.08477,0.08759,0.1037,0.09933,0.07837,0.1122,0.115,0.09768,0.09462,0.09373,0.1101,0.07115,0.08785,0.115,0.1066,0.09462,0.1051,0.09639,0.1164,0.0925,0.1092,0.08677,0.07793,0.1152,0.08138,0.0997,0.07944,0.1135,0.09405,0.1072,0.09754,0.09384,0.08654,0.07445,0.09311,0.07515,0.1089,0.1012,0.08421,0.09594,0.09855,0.1028,0.1006,0.08792,0.09138,0.09699,0.06251,0.08739,0.09059,0.09267,0.09774,0.1007,0.0808,0.08749,0.0695,0.07941,0.09586,0.09965,0.1089,0.1002],'malignant': [0.1184,0.08474,0.1096,0.1425,0.1003,0.1278,0.09463,0.1189,0.1273,0.1186,0.08206,0.0971,0.0974,0.08401,0.1131,0.1139,0.09867,0.117,0.09831,0.1073,0.09428,0.1121,0.1186,0.1054,0.0944,0.1082,0.09847,0.1064,0.1109,0.1197,0.09401,0.104,0.0961,0.09823,0.09387,0.1016,0.08162,0.1227,0.09081,0.1041,0.09714,0.1099,0.1158,0.1148,0.09056,0.1053,0.1137,0.1049,0.1122,0.1172,0.09009,0.1071,0.1007,0.09168,0.1065,0.1286,0.1063,0.1215,0.09874,0.09444,0.09029,0.092,0.1039,0.09078,0.09752,0.09488,0.1398,0.1326,0.1162,0.1155,0.08402,0.1054,0.1447,0.09258,0.08217,0.1015,0.1092,0.1008,0.0943,0.09055,0.1167,0.09721,0.1091,0.1115,0.08694,0.112,0.08439,0.08865,0.1049,0.09048,0.1257,0.09831,0.1094,0.1141,0.09597,0.09057,0.08588,0.1075,0.1034,0.1044]},
    'compactness_mean':       {'benign': [0.08129,0.127,0.06492,0.03766,0.05943,0.09092,0.07698,0.04966,0.06059,0.04751,0.05473,0.03789,0.05272,0.08061,0.08963,0.08751,0.07773,0.04701,0.1413,0.05234,0.1531,0.06829,0.1047,0.09546,0.09362,0.1535,0.07165,0.09445,0.1339,0.08606,0.05055,0.08165,0.07057,0.05301,0.07525,0.07568,0.04038,0.09697,0.08578,0.1017,0.06815,0.06575,0.08404,0.1209,0.2233,0.1303,0.08201,0.07849,0.1243,0.06685,0.1099,0.07325,0.06136,0.1807,0.09509,0.09462,0.06095,0.06889,0.1136,0.04102,0.09486,0.09509,0.05139,0.1296,0.1167,0.1021,0.06376,0.07589,0.1305,0.1599,0.05113,0.08498,0.06679,0.07223,0.05241,0.03718,0.1141,0.1015,0.05352,0.05736,0.07885,0.06981,0.05743,0.04302,0.04276,0.1294,0.01938,0.03774,0.08155,0.04695,0.06141,0.05562,0.07253,0.06601,0.02344,0.05366,0.08087,0.1058,0.07232,0.1483],'malignant': [0.2776,0.07864,0.1599,0.2839,0.1328,0.17,0.109,0.1645,0.1932,0.2396,0.06669,0.1292,0.2458,0.1002,0.2293,0.1595,0.072,0.2022,0.1027,0.2135,0.1022,0.1457,0.2276,0.1868,0.1066,0.1697,0.1157,0.1887,0.1516,0.1496,0.1719,0.1559,0.1336,0.1098,0.05131,0.1255,0.06031,0.1218,0.219,0.1436,0.1047,0.1686,0.1231,0.1485,0.07081,0.1267,0.1365,0.2008,0.1262,0.1479,0.1029,0.183,0.128,0.08424,0.2146,0.3454,0.2665,0.1791,0.1053,0.09947,0.1206,0.1036,0.1553,0.1313,0.1141,0.08511,0.1765,0.2768,0.1649,0.1752,0.06722,0.11,0.2867,0.07862,0.08028,0.1589,0.1223,0.1284,0.09709,0.05761,0.1305,0.1137,0.17,0.1665,0.1185,0.1666,0.1145,0.09182,0.1603,0.06288,0.1555,0.1556,0.1914,0.2832,0.08799,0.1052,0.08468,0.2413,0.1353,0.198]},
    'concavity_mean':         {'benign': [0.06664,0.04568,0.02956,0.02562,0.01588,0.06592,0.04751,0.01657,0.01857,0.01972,0.03036,0.000692,0.02061,0.01084,0.03,0.05988,0.02172,0.03709,0.313,0.03653,0.08606,0.03372,0.06877,0.03889,0.04591,0.1169,0.04151,0.06015,0.09966,0.03102,0.03261,0.03974,0.0249,0.006829,0.04196,0.0,0.02383,0.06154,0.02995,0.0707,0.02643,0.05133,0.04334,0.1065,0.3003,0.06476,0.04132,0.03328,0.09263,0.03512,0.08842,0.08092,0.0142,0.1138,0.02855,0.07135,0.03592,0.03503,0.04635,0.0,0.02031,0.04894,0.02251,0.0371,0.0905,0.08487,0.02881,0.03136,0.1321,0.4108,0.01982,0.09293,0.03885,0.0515,0.01972,0.00309,0.06843,0.0537,0.01947,0.02531,0.02602,0.03987,0.02363,0.0,0.0,0.1307,0.001595,0.009193,0.06181,0.001597,0.03809,0.02353,0.03844,0.03112,0.0,0.03873,0.04187,0.08005,0.01756,0.08705],'malignant': [0.3001,0.0869,0.1974,0.2414,0.198,0.1578,0.1127,0.09366,0.1859,0.2273,0.03299,0.09954,0.2065,0.09938,0.2128,0.1639,0.07395,0.1722,0.1479,0.2077,0.1097,0.1525,0.2229,0.1425,0.149,0.1683,0.09875,0.2319,0.1218,0.2417,0.1657,0.1354,0.1348,0.1319,0.02398,0.1063,0.0311,0.1044,0.2107,0.09847,0.08259,0.1974,0.1226,0.1772,0.05253,0.1323,0.1293,0.2135,0.1128,0.1267,0.108,0.1692,0.07789,0.09769,0.1684,0.3754,0.3339,0.1937,0.1335,0.1204,0.1468,0.1122,0.17,0.1465,0.09388,0.08625,0.2071,0.4264,0.169,0.2133,0.07293,0.1457,0.4268,0.05285,0.09271,0.2545,0.1466,0.1043,0.1153,0.04711,0.1539,0.09447,0.1659,0.1855,0.1193,0.2508,0.1324,0.08422,0.2159,0.05858,0.2032,0.1793,0.2871,0.2487,0.06593,0.05375,0.08169,0.1981,0.1085,0.1697]},
    'concave points_mean':    {'benign': [0.04781,0.0311,0.02076,0.02923,0.005917,0.02749,0.03384,0.01115,0.01723,0.01349,0.02278,0.004167,0.007799,0.0129,0.009259,0.0218,0.01504,0.0223,0.04375,0.02864,0.02872,0.02272,0.06556,0.02315,0.02233,0.06987,0.01863,0.03745,0.07064,0.02957,0.02648,0.0278,0.02941,0.007937,0.0335,0.0,0.0177,0.03029,0.01201,0.03485,0.01921,0.01899,0.01778,0.06021,0.07798,0.03068,0.01924,0.02008,0.02308,0.02623,0.05778,0.028,0.01141,0.08534,0.02882,0.05933,0.026,0.02875,0.04796,0.0,0.01861,0.03088,0.007875,0.03003,0.03562,0.05532,0.01329,0.02645,0.02168,0.07857,0.01786,0.03483,0.02331,0.02771,0.01963,0.006588,0.03738,0.02822,0.01939,0.01698,0.03781,0.037,0.02583,0.0,0.0,0.03716,0.001852,0.0133,0.02361,0.002404,0.03239,0.01553,0.01654,0.02864,0.0,0.02377,0.04107,0.03821,0.01952,0.05102],'malignant': [0.1471,0.07017,0.1279,0.1052,0.1043,0.08089,0.074,0.05985,0.09353,0.08543,0.03323,0.06606,0.1118,0.05364,0.08025,0.07364,0.05259,0.1028,0.09498,0.09756,0.08632,0.0917,0.1401,0.08783,0.07731,0.08751,0.07953,0.1244,0.05182,0.1203,0.07593,0.07752,0.06018,0.05598,0.02899,0.05439,0.02031,0.05669,0.09961,0.06158,0.05252,0.1009,0.0734,0.106,0.03334,0.08994,0.08123,0.08653,0.06873,0.09029,0.07951,0.07944,0.05069,0.06638,0.108,0.1604,0.1845,0.1469,0.08795,0.04938,0.08271,0.07483,0.08815,0.08683,0.05839,0.04489,0.09601,0.1823,0.08923,0.09479,0.05596,0.08665,0.2012,0.03085,0.05627,0.1149,0.08087,0.05613,0.06847,0.02704,0.08624,0.05943,0.07415,0.1054,0.09667,0.1286,0.09702,0.06576,0.1043,0.03438,0.1097,0.08866,0.1878,0.1496,0.05189,0.03263,0.05814,0.06618,0.04562,0.08878]},
    'symmetry_mean':          {'benign': [0.1885,0.1967,0.1815,0.1467,0.1769,0.1675,0.1809,0.1495,0.1353,0.1868,0.192,0.1819,0.1683,0.2743,0.1828,0.2341,0.1717,0.1516,0.2111,0.159,0.1902,0.172,0.2403,0.1718,0.1842,0.1942,0.2079,0.193,0.2116,0.1685,0.1386,0.1638,0.19,0.135,0.162,0.193,0.1739,0.1945,0.2217,0.1801,0.1602,0.1487,0.1584,0.1735,0.1704,0.1922,0.1649,0.1688,0.1305,0.1667,0.1856,0.1422,0.1614,0.2001,0.188,0.1816,0.1339,0.1734,0.1771,0.1903,0.1645,0.1778,0.1399,0.1995,0.1744,0.1724,0.1473,0.254,0.2222,0.2548,0.183,0.1822,0.197,0.1844,0.159,0.1442,0.1993,0.1551,0.1515,0.1381,0.178,0.1959,0.1566,0.1928,0.1722,0.1669,0.1395,0.1466,0.1167,0.1703,0.1516,0.1718,0.1667,0.1694,0.1653,0.1829,0.1979,0.1925,0.1934],'malignant': [0.2419,0.1812,0.2069,0.2597,0.1809,0.2087,0.1794,0.2196,0.235,0.203,0.1528,0.1842,0.2397,0.1847,0.2069,0.2303,0.1586,0.2164,0.1582,0.2521,0.1769,0.1995,0.304,0.2252,0.1697,0.1926,0.1739,0.2183,0.2301,0.2248,0.1853,0.1998,0.1896,0.1885,0.1565,0.172,0.1784,0.1895,0.231,0.1974,0.1746,0.1907,0.2128,0.2092,0.1616,0.1917,0.2027,0.1949,0.1905,0.1953,0.1582,0.1927,0.1662,0.1798,0.2152,0.2906,0.1829,0.1634,0.2132,0.2075,0.1953,0.1717,0.1855,0.2095,0.1879,0.1609,0.1925,0.2556,0.2157,0.2096,0.2129,0.1966,0.2655,0.1761,0.1946,0.2202,0.1931,0.216,0.1692,0.1585,0.1957,0.1861,0.2678,0.1971,0.1741,0.2027,0.1801,0.1893,0.1538,0.1598,0.1966,0.1794,0.18,0.2395,0.1618,0.1727,0.1621,0.2384,0.1943,0.1737]},
    'fractal_dimension_mean': {'benign': [0.05766,0.06811,0.06905,0.05863,0.06503,0.06043,0.05718,0.05888,0.05953,0.0611,0.05907,0.05501,0.07187,0.0696,0.06757,0.06963,0.06899,0.05667,0.08046,0.05653,0.0898,0.05914,0.06641,0.05997,0.07005,0.06902,0.05968,0.06404,0.06346,0.05866,0.05318,0.0571,0.06635,0.0689,0.06582,0.07818,0.05677,0.06322,0.06481,0.0652,0.06066,0.06529,0.07065,0.0707,0.07769,0.07782,0.07633,0.06194,0.07163,0.06113,0.06402,0.05823,0.0589,0.06467,0.06471,0.05723,0.05945,0.05865,0.06072,0.06422,0.06562,0.06235,0.05688,0.07839,0.06493,0.06081,0.0558,0.06087,0.08261,0.09296,0.06105,0.06207,0.06228,0.05268,0.05907,0.05743,0.06453,0.06761,0.05266,0.064,0.0565,0.05955,0.06669,0.05975,0.06724,0.08116,0.05234,0.06133,0.06217,0.06048,0.06095,0.0578,0.05474,0.06287,0.06447,0.05667,0.06013,0.06373,0.06285,0.0731],'malignant': [0.07871,0.05667,0.05999,0.09744,0.05883,0.07613,0.05742,0.07451,0.07389,0.08243,0.05697,0.06082,0.078,0.05338,0.07682,0.07077,0.05922,0.07356,0.05395,0.07032,0.05278,0.0633,0.07413,0.06924,0.05699,0.0654,0.06149,0.06197,0.07799,0.06382,0.06261,0.06515,0.05656,0.06125,0.05504,0.06419,0.05587,0.0687,0.06343,0.06782,0.06177,0.06049,0.06777,0.0631,0.05684,0.05961,0.06758,0.07292,0.0659,0.06654,0.05461,0.06487,0.06566,0.05391,0.06673,0.08142,0.06782,0.07224,0.06022,0.05636,0.05629,0.06097,0.06284,0.05649,0.0639,0.05871,0.07692,0.07039,0.06768,0.07331,0.05025,0.06213,0.06877,0.0613,0.05044,0.06113,0.05796,0.05891,0.05727,0.06065,0.06216,0.06248,0.07371,0.06166,0.05176,0.06082,0.05553,0.05534,0.06365,0.05671,0.07069,0.06323,0.0577,0.07398,0.05549,0.06317,0.05425,0.07542,0.06937,0.06672]},
}
class_counts   = {'Benign': 357, 'Malignant': 212}
training_sizes = {
    'Baseline (37/63)': {'Benign': 285, 'Malignant': 170},
    'Case 1 (20/80)':   {'Benign': 364, 'Malignant': 91},
    'Case 2 (50/50)':   {'Benign': 227, 'Malignant': 228},
    'Case 3 (80/20)':   {'Benign': 91,  'Malignant': 364},
}
us_cases     = 310720
global_cases = 2300000

# ── Color Constants ────────────────────────────────────────────────────────
C_BENIGN    = '#2ecc71'
C_MALIGNANT = '#e74c3c'
C_LR        = '#3498db'
C_SVM       = '#e67e22'
C_RF        = '#9b59b6'
CLF_COLORS  = {'Logistic Regression': C_LR, 'SVM': C_SVM, 'Random Forest': C_RF}
CASES       = list(accuracy.keys())
CLFS        = ['Logistic Regression', 'SVM', 'Random Forest']

# ── Helpers ────────────────────────────────────────────────────────────────
def metric_line_chart(metric_dict, title, y_label, y_range):
    fig = go.Figure()
    for clf in CLFS:
        fig.add_trace(go.Scatter(
            x=CASES, y=[metric_dict[c][clf] for c in CASES],
            mode='lines+markers', name=clf,
            line=dict(color=CLF_COLORS[clf], width=2.5),
            marker=dict(size=8)
        ))
    fig.update_layout(
        title=title, template='plotly_white',
        xaxis_title='Class Distribution', yaxis_title=y_label,
        yaxis=dict(range=y_range), height=380, legend=dict(orientation='h', y=1.12)
    )
    return fig

def confusion_matrix_fig(case):
    fig = make_subplots(rows=1, cols=3, subplot_titles=CLFS)
    cm_colors = [[1, 0], [0, 1]]
    for i, clf in enumerate(CLFS):
        d  = cm_data[case][clf]
        cm = [[d['TN'], d['FP']], [d['FN'], d['TP']]]
        fig.add_trace(go.Heatmap(
            z=cm_colors, x=['Benign', 'Malignant'], y=['Benign', 'Malignant'],
            colorscale=[[0, C_MALIGNANT], [1, C_BENIGN]],
            text=cm, texttemplate='%{text}', showscale=False, hoverinfo='none'
        ), row=1, col=i+1)
    fig.update_layout(template='plotly_white', height=320,
                      title=f'Confusion Matrices — {case}')
    return fig

def kde_trace(data, color, name, show_legend=True):
    arr = np.array(data)
    x   = np.linspace(arr.min(), arr.max(), 300)
    y   = gaussian_kde(arr)(x)
    return go.Scatter(x=x, y=y, mode='lines', name=name,
                      fill='tozeroy', opacity=0.5,
                      line=dict(color=color, width=2), showlegend=show_legend)

# ── Sidebar Nav ────────────────────────────────────────────────────────────
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/8/8e/Northeastern_University_logo.svg/320px-Northeastern_University_logo.svg.png", width=180)
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", [
    "🏠 Overview",
    "🔬 Feature Exploration",
    "📊 Baseline Classification",
    "⚗️ Distribution Experiment",
    "🌍 Real World Impact"
])
st.sidebar.markdown("---")
st.sidebar.markdown("**EECE 5642 — Data Visualization**")
st.sidebar.markdown("Stephany Erhabor · Spring 2025")

# ══════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.title("The Effect of Class Distribution on Breast Cancer Classification")
    st.markdown("#### Wisconsin Breast Cancer Dataset · EECE 5642 Final Project")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Samples",  "569")
    col2.metric("Features",       "30")
    col3.metric("Benign Cases",   f"{class_counts['Benign']} (63%)")
    col4.metric("Malignant Cases",f"{class_counts['Malignant']} (37%)")

    st.markdown("---")
    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("Research Question")
        st.markdown("""
        > *How does the balance of malignant vs. benign cases in training data affect classifier performance?*

        Breast cancer is the most diagnosed cancer in women worldwide. 
        AI-assisted mammogram tools are becoming increasingly prevalent — 
        yet most are trained on data that does not reflect real clinical distributions.

        In real clinical settings, **approximately 80% of confirmed diagnoses are benign** 
        and only 20% are malignant. This project investigates what happens when that ratio shifts.
        """)
    with c2:
        st.subheader("Project Structure")
        st.markdown("""
        | Stage | Focus |
        |-------|-------|
        | **01 — Feature Exploration** | Are there observable structural differences between malignant and benign cells? |
        | **02 — Baseline Classification** | Logistic Regression, SVM, Random Forest on the original dataset |
        | **03 — Distribution Experiment** | Retrain on 20/80, 50/50, 80/20 malignant-to-benign ratios |
        """)

    st.markdown("---")
    st.subheader("Class Distribution in Dataset")
    fig = go.Figure(go.Pie(
        labels=list(class_counts.keys()),
        values=list(class_counts.values()),
        hole=0.45,
        marker_colors=[C_BENIGN, C_MALIGNANT],
        textinfo='label+percent'
    ))
    fig.update_layout(template='plotly_white', height=320,
                      showlegend=False, margin=dict(t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Methodology Notes")
    col1, col2, col3 = st.columns(3)
    col1.info("**Train/Test Split**\n\n80/20 with `stratify=y` to preserve class ratio. Test set is fixed across all experiments.")
    col2.info("**Resampling**\n\nHybrid RandomOverSampler + RandomUnderSampler. Fixed training size (455 samples) across all cases. SMOTE excluded to avoid altering data geometry.")
    col3.info("**Implementation**\n\nResults are pre-computed from Google Colab. This dashboard presents hardcoded outputs for reliable live presentation.")


# ══════════════════════════════════════════════════════════════════════════
# PAGE 2 — FEATURE EXPLORATION
# ══════════════════════════════════════════════════════════════════════════
elif page == "🔬 Feature Exploration":
    st.title("Feature Exploration")
    st.markdown("Comparing the distribution of 10 cell nucleus `_mean` features across Benign and Malignant classes.")

    chart_type = st.radio("Chart Type", ["Histogram", "KDE Curves", "Box Plot"], horizontal=True)
    st.markdown("---")

    features = list(feature_data.keys())
    cols_per_row = 5
    rows = [features[i:i+cols_per_row] for i in range(0, len(features), cols_per_row)]

    for row in rows:
        cols = st.columns(len(row))
        for col, feat in zip(cols, row):
            b = feature_data[feat]['benign']
            m = feature_data[feat]['malignant']
            if chart_type == "Histogram":
                fig = go.Figure()
                fig.add_trace(go.Histogram(x=b, name='Benign', marker_color=C_BENIGN,
                                           opacity=0.6, showlegend=False))
                fig.add_trace(go.Histogram(x=m, name='Malignant', marker_color=C_MALIGNANT,
                                           opacity=0.6, showlegend=False))
                fig.update_layout(barmode='overlay', template='plotly_white',
                                  height=220, margin=dict(t=30, b=20, l=10, r=10),
                                  title=dict(text=feat.replace('_', ' '), font_size=11))
            elif chart_type == "KDE Curves":
                fig = go.Figure()
                fig.add_trace(kde_trace(b, C_BENIGN, 'Benign', False))
                fig.add_trace(kde_trace(m, C_MALIGNANT, 'Malignant', False))
                fig.update_layout(template='plotly_white', height=220,
                                  margin=dict(t=30, b=20, l=10, r=10),
                                  title=dict(text=feat.replace('_', ' '), font_size=11))
            else:
                fig = go.Figure()
                fig.add_trace(go.Box(y=b, name='Benign', marker_color=C_BENIGN, showlegend=False))
                fig.add_trace(go.Box(y=m, name='Malignant', marker_color=C_MALIGNANT, showlegend=False))
                fig.update_layout(template='plotly_white', height=220,
                                  margin=dict(t=30, b=20, l=10, r=10),
                                  title=dict(text=feat.replace('_', ' '), font_size=11))
            col.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Key Observation")
    st.success("Features such as **radius_mean**, **area_mean**, **perimeter_mean**, **concavity_mean**, and **concave points_mean** show strong class separation — malignant tumors are consistently larger and more irregular. Features like **smoothness_mean** and **fractal_dimension_mean** show minimal separation, suggesting they contribute less to classification.")


# ══════════════════════════════════════════════════════════════════════════
# PAGE 3 — BASELINE CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════
elif page == "📊 Baseline Classification":
    st.title("Baseline Classification")
    st.markdown("All three classifiers trained on the original Wisconsin dataset (37% malignant / 63% benign), evaluated on a fixed 20% test set.")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    for col, clf in zip([col1, col2, col3], CLFS):
        acc  = accuracy['Baseline (37/63)'][clf]
        sen  = sensitivity['Baseline (37/63)'][clf]
        spe  = specificity['Baseline (37/63)'][clf]
        auc_ = auc['Baseline (37/63)'][clf]
        col.markdown(f"### {clf}")
        col.metric("Accuracy",    f"{acc:.1%}")
        col.metric("Sensitivity", f"{sen:.1%}")
        col.metric("Specificity", f"{spe:.1%}")
        col.metric("AUC-ROC",     f"{auc_:.4f}")

    st.markdown("---")
    st.subheader("Confusion Matrices — Baseline")
    st.plotly_chart(confusion_matrix_fig('Baseline (37/63)'), use_container_width=True)

    st.markdown("---")
    st.subheader("Radar Chart — Baseline Performance")
    categories = ['Accuracy', 'Sensitivity', 'Specificity', 'AUC-ROC']
    fig = go.Figure()
    for clf in CLFS:
        vals = [
            accuracy['Baseline (37/63)'][clf],
            sensitivity['Baseline (37/63)'][clf],
            specificity['Baseline (37/63)'][clf],
            auc['Baseline (37/63)'][clf]
        ]
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]], theta=categories + [categories[0]],
            fill='toself', name=clf,
            line=dict(color=CLF_COLORS[clf], width=2), opacity=0.6
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(range=[0.85, 1.0])),
        template='plotly_white', height=400,
        legend=dict(orientation='h', y=-0.15)
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Key Finding")
    st.info("**Logistic Regression outperforms SVM and Random Forest** on this dataset — contrary to the conventional expectation that more complex models perform better. This suggests the decision boundary between malignant and benign is largely linear, which is supported by the clear class separation observed in Stage 1.")


# ══════════════════════════════════════════════════════════════════════════
# PAGE 4 — DISTRIBUTION EXPERIMENT
# ══════════════════════════════════════════════════════════════════════════
elif page == "⚗️ Distribution Experiment":
    st.title("Distribution Experiment")
    st.markdown("Three fixed training sets (455 samples each) at different malignant-to-benign ratios. Test set is unchanged across all cases.")

    st.markdown("---")
    st.subheader("Training Set Composition")
    fig = go.Figure()
    for cls, color in [('Benign', C_BENIGN), ('Malignant', C_MALIGNANT)]:
        fig.add_trace(go.Bar(
            x=list(training_sizes.keys()),
            y=[training_sizes[c][cls] for c in training_sizes],
            name=cls, marker_color=color
        ))
    fig.update_layout(barmode='stack', template='plotly_white',
                      height=300, xaxis_title='Case', yaxis_title='Samples',
                      legend=dict(orientation='h', y=1.1))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    metric_choice = st.selectbox("Select Metric", ["Accuracy", "Sensitivity", "Specificity", "AUC-ROC"])
    metric_map = {
        "Accuracy":    (accuracy,    "Accuracy",    [0.85, 1.01]),
        "Sensitivity": (sensitivity, "Sensitivity", [0.80, 1.01]),
        "Specificity": (specificity, "Specificity", [0.80, 1.01]),
        "AUC-ROC":     (auc,         "AUC-ROC",     [0.96, 1.01]),
    }
    d, label, rng = metric_map[metric_choice]
    st.plotly_chart(metric_line_chart(d, f"{label} Across Class Distribution Cases", label, rng),
                    use_container_width=True)

    st.markdown("---")
    st.subheader("Confusion Matrices by Case")
    case_choice = st.selectbox("Select Case", CASES)
    st.plotly_chart(confusion_matrix_fig(case_choice), use_container_width=True)

    st.markdown("---")
    st.subheader("All Metrics Summary Table")
    rows_data = []
    for case in CASES:
        for clf in CLFS:
            rows_data.append({
                'Case': case, 'Classifier': clf,
                'Accuracy':    f"{accuracy[case][clf]:.1%}",
                'Sensitivity': f"{sensitivity[case][clf]:.1%}",
                'Specificity': f"{specificity[case][clf]:.1%}",
                'AUC-ROC':     f"{auc[case][clf]:.4f}",
            })
    import pandas as pd
    st.dataframe(pd.DataFrame(rows_data), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Key Findings")
    col1, col2 = st.columns(2)
    col1.warning("**Sensitivity vs. Specificity Tradeoff**\n\nAs malignant cases increase in training (Case 3), sensitivity reaches 1.000 for all classifiers but specificity drops significantly — classifiers become overly aggressive in predicting malignant.")
    col2.error("**Case 1 is the most dangerous**\n\nCase 1 (20/80) is closest to real clinical distribution yet produces the worst sensitivity across all classifiers. RF drops to 85.7% — meaning 1 in 7 cancer cases would be missed.")


# ══════════════════════════════════════════════════════════════════════════
# PAGE 5 — REAL WORLD IMPACT
# ══════════════════════════════════════════════════════════════════════════
elif page == "🌍 Real World Impact":
    st.title("Real World Impact")
    st.markdown("Translating classifier sensitivity drops into missed diagnoses at scale.")
    st.markdown("---")

    st.subheader("Impact Calculator")
    col1, col2 = st.columns(2)
    with col1:
        selected_clf  = st.selectbox("Classifier",        CLFS)
        selected_case = st.selectbox("Distribution Case", CASES)
    with col2:
        baseline_sen = sensitivity['Baseline (37/63)'][selected_clf]
        case_sen     = sensitivity[selected_case][selected_clf]
        drop         = baseline_sen - case_sen
        us_missed    = int(abs(drop) * us_cases)
        global_missed= int(abs(drop) * global_cases)

        st.metric("Baseline Sensitivity",      f"{baseline_sen:.1%}")
        st.metric("Selected Case Sensitivity", f"{case_sen:.1%}",
                  delta=f"{drop:+.1%}", delta_color="inverse")

    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    col1.metric("Sensitivity Drop",        f"{abs(drop):.1%}")
    col2.metric("Missed Diagnoses — US",   f"{us_missed:,}")
    col3.metric("Missed Diagnoses — Global",f"{global_missed:,}")

    st.markdown("---")
    st.subheader("Sensitivity Across All Cases")
    fig = go.Figure()
    for clf in CLFS:
        fig.add_trace(go.Bar(
            x=CASES, y=[sensitivity[c][clf] for c in CASES],
            name=clf, marker_color=CLF_COLORS[clf], opacity=0.85
        ))
    fig.update_layout(barmode='group', template='plotly_white',
                      height=380, yaxis=dict(range=[0.8, 1.01]),
                      xaxis_title='Case', yaxis_title='Sensitivity',
                      legend=dict(orientation='h', y=1.1))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Global Scale Context")
    st.markdown("""
    - **2.3 million** women diagnosed with breast cancer globally in 2022
    - **310,720** new invasive breast cancer cases in the US in 2024
    - By **2040**, global incidence projected to exceed **3 million** per year
    - A **1% sensitivity drop** = ~23,000 missed diagnoses globally per year
    - Case 1 RF sensitivity drop of **7.2%** = ~**165,600 missed cases** globally per year

    > *"The stakes of training data composition in AI-assisted diagnosis are not academic — they are life and death."*
    """)

    st.markdown("---")
    st.subheader("Why Case 1 Matters Most")
    st.error("""
    **Case 1 (20% malignant / 80% benign) is the closest to real clinical data.**

    In real screening populations, the malignant rate among confirmed biopsies is approximately 20%.
    This means classifiers deployed in clinical settings are most likely to operate under Case 1 conditions —
    which is precisely the scenario that produces the worst sensitivity in our experiment.

    This finding directly motivates the need to study and address class distribution effects
    before deploying AI tools in medical diagnosis.
    """)
