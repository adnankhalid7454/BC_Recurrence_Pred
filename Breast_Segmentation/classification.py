import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report)

from pytorch_tabnet.tab_model import TabNetClassifier

# === CONFIGURATION ===
TEST_FILE = 'mri_breast_test.xlsx'
MODEL_PATH = 'final_best_tabnet_model.zip'
IMPUTER_PATH = 'knn_imputer.pkl'
SCALER_PATH = 'scaler_minmax.pkl'
SELECTOR_PATH = 'feature_selector_kbest.pkl'
TRAIN_MEDIAN_VALUE = 0.5  

# === LOAD TEST DATA ===
test_data = pd.read_excel(TEST_FILE).sample(frac=1, random_state=42)
X_test = test_data.drop(columns='Recurrence')
y_test = (test_data['Recurrence'] > TRAIN_MEDIAN_VALUE).astype(int)

# === LOAD TRAINED TRANSFORMERS ===
imputer = joblib.load(IMPUTER_PATH)
scaler = joblib.load(SCALER_PATH)
selector = joblib.load(SELECTOR_PATH)

# === PREPROCESS TEST DATA ===
X_test_imputed = imputer.transform(X_test)
X_test_scaled = scaler.transform(X_test_imputed)
X_test_selected = selector.transform(X_test_scaled)

# === LOAD TABNET MODEL ===
model = TabNetClassifier()
model.load_model(MODEL_PATH)

# === MAKE PREDICTIONS ===
y_pred = model.predict(X_test_selected)
y_prob = model.predict_proba(X_test_selected)[:, 1]

# === EVALUATION METRICS ===
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)
cm = confusion_matrix(y_test, y_pred)

# === PRINT RESULTS ===
print("\nTabNet Model Performance on Test Set:")
print(f"Accuracy:      {accuracy:.4f}")
print(f"Precision:     {precision:.4f}")
print(f"Recall:        {recall:.4f}")
print(f"F1 Score:      {f1:.4f}")
print(f"ROC AUC Score: {roc_auc:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(cm)

# === PLOT CONFUSION MATRIX ===
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['Actual 0', 'Actual 1'])
plt.title('TabNet Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()
