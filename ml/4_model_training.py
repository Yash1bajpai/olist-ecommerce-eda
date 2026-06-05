import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# paths
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE_DIR, 'master_data.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

# load data
df = pd.read_csv(DATA_PATH)
print(f"Loaded: {df.shape}")

# filter: only delivered orders with actual delivery date
filtered_df = df[
    (df['order_status'] == 'delivered') &
    (df['is_order_delivered_customer_date_missing'] == 0)
].copy()
print(f"After filter: {filtered_df.shape}")

# convert date columns
date_cols = [
    'order_delivered_customer_date',
    'order_estimated_delivery_date',
    'order_purchase_timestamp',
    'order_approved_at'
]
for col in date_cols:
    filtered_df[col] = pd.to_datetime(filtered_df[col])

# target variable: 1 if delivered after estimated date
filtered_df['is_late'] = np.where(
    filtered_df['order_delivered_customer_date'] > filtered_df['order_estimated_delivery_date'],
    1, 0
)
print(f"Class distribution:\n{filtered_df['is_late'].value_counts()}\n")

# feature engineering
filtered_df['approval_days'] = (
    filtered_df['order_approved_at'] - filtered_df['order_purchase_timestamp']
).dt.days
filtered_df['approval_days'] = filtered_df['approval_days'].clip(lower=0)  # negative = data entry error

filtered_df['estimated_delivery_window'] = (
    filtered_df['order_estimated_delivery_date'] - filtered_df['order_purchase_timestamp']
).dt.days

filtered_df['purchase_month']     = filtered_df['order_purchase_timestamp'].dt.month
filtered_df['purchase_dayofweek'] = filtered_df['order_purchase_timestamp'].dt.dayofweek

filtered_df['product_volume_cm3'] = (
    filtered_df['product_length_cm'] *
    filtered_df['product_height_cm'] *
    filtered_df['product_width_cm']
)

feature_cols = [
    'approval_days',
    'estimated_delivery_window',
    'purchase_month',
    'purchase_dayofweek',
    'product_volume_cm3',
    'price',
    'freight_value',
    'product_weight_g'
]

# fill missing values with median (robust to outliers)
for col in ['approval_days', 'product_volume_cm3', 'product_weight_g']:
    filtered_df[col] = filtered_df[col].fillna(filtered_df[col].median())

print(f"NaN check:\n{filtered_df[feature_cols].isna().sum()}\n")

# train/test split — stratify keeps the 92/8 class ratio in both splits
X = filtered_df[feature_cols]
y = filtered_df['is_late']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {X_train.shape} | Test: {X_test.shape}")

# baseline: logistic regression — Pipeline ensures scaler is part of the model artifact
print("\nTraining Logistic Regression...")
lr_model = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(max_iter=5000, random_state=42, class_weight='balanced'))
])
lr_model.fit(X_train, y_train)

# main model: random forest with class balancing
print("Training Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',  # penalizes mistakes on rare class more
    random_state=42
)
rf_model.fit(X_train, y_train)

# save models and test set
joblib.dump(lr_model,     os.path.join(MODELS_DIR, 'logistic_regression.pkl'))
joblib.dump(rf_model,     os.path.join(MODELS_DIR, 'random_forest.pkl'))
joblib.dump(feature_cols, os.path.join(MODELS_DIR, 'feature_cols.pkl'))

X_test.to_csv(os.path.join(MODELS_DIR, 'X_test.csv'), index=False)
y_test.to_csv(os.path.join(MODELS_DIR, 'y_test.csv'), index=False)

print("\nModels saved to models/")
