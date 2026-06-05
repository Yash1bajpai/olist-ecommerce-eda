import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)

# paths
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
PLOTS_DIR  = os.path.join(BASE_DIR, 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

# load saved models and test data
lr_model     = joblib.load(os.path.join(MODELS_DIR, 'logistic_regression.pkl'))
rf_model     = joblib.load(os.path.join(MODELS_DIR, 'random_forest.pkl'))
feature_cols = joblib.load(os.path.join(MODELS_DIR, 'feature_cols.pkl'))

X_test = pd.read_csv(os.path.join(MODELS_DIR, 'X_test.csv'))
y_test = pd.read_csv(os.path.join(MODELS_DIR, 'y_test.csv')).squeeze()

# predictions
y_pred_lr = lr_model.predict(X_test)
y_pred_rf = rf_model.predict(X_test)

# probabilities for ROC-AUC (column 1 = probability of being late)
y_prob_lr = lr_model.predict_proba(X_test)[:, 1]
y_prob_rf = rf_model.predict_proba(X_test)[:, 1]

# classification reports
print("── Logistic Regression ─────────────────────────────")
print(classification_report(y_test, y_pred_lr, target_names=['On Time', 'Late']))

print("── Random Forest ────────────────────────────────────")
print(classification_report(y_test, y_pred_rf, target_names=['On Time', 'Late']))

# ROC-AUC
auc_lr = roc_auc_score(y_test, y_prob_lr)
auc_rf = roc_auc_score(y_test, y_prob_rf)
print(f"ROC-AUC  Logistic Regression : {auc_lr:.4f}")
print(f"ROC-AUC  Random Forest       : {auc_rf:.4f}")

# confusion matrices (side by side)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, model_name, y_pred in zip(
    axes,
    ['Logistic Regression', 'Random Forest'],
    [y_pred_lr, y_pred_rf]
):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues', ax=ax,
        xticklabels=['On Time', 'Late'],
        yticklabels=['On Time', 'Late']
    )
    ax.set_title(f'Confusion Matrix — {model_name}')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'confusion_matrices.png'), dpi=150, bbox_inches='tight')
plt.show()

# ROC curve
plt.figure(figsize=(8, 6))
for model_name, y_prob, auc in [
    ('Logistic Regression', y_prob_lr, auc_lr),
    ('Random Forest',       y_prob_rf, auc_rf)
]:
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.4f})', linewidth=2)

plt.plot([0, 1], [0, 1], 'k--', label='Random Guess', linewidth=1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve — LR vs Random Forest')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'roc_curve.png'), dpi=150, bbox_inches='tight')
plt.show()

# feature importance (random forest)
importances = pd.Series(rf_model.feature_importances_, index=feature_cols).sort_values()

plt.figure(figsize=(10, 6))
importances.plot(kind='barh', color='teal')
plt.xlabel('Importance Score')
plt.title('Random Forest — Feature Importances')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'feature_importance.png'), dpi=150, bbox_inches='tight')
plt.show()

print("\nPlots saved to plots/")
