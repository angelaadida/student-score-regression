"""
=============================================================
Project : Student Math Score Prediction
Author  : Angela
Dataset : Students Performance in Exams
Goal    : Predict student math score using Linear Regression Pipeline
=============================================================
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ─────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────
if os.path.exists('StudentsPerformance.csv'):
    data = pd.read_csv('StudentsPerformance.csv')
elif os.path.exists('StudentsPerformance.xlsx'):
    data = pd.read_excel('StudentsPerformance.xlsx')
else:
    raise FileNotFoundError('Please place StudentsPerformance.csv or .xlsx in this folder!')

print('Dataset Shape:', data.shape)
print(data.head())

# ─────────────────────────────────────────
# 2. SPLIT DATA
# ─────────────────────────────────────────
target = 'math score'
X = data.drop(target, axis=1)
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f'Train: {len(X_train)} | Test: {len(X_test)}')

# ─────────────────────────────────────────
# 3. BUILD PIPELINE
# ─────────────────────────────────────────
num_features = ['reading score', 'writing score']
nom_features = ['race/ethnicity']
ord_features = ['parental level of education', 'gender', 'lunch', 'test preparation course']

education_values = [
    'some high school', 'high school', 'some college',
    "associate's degree", "bachelor's degree", "master's degree"
]
gender_values  = X_train['gender'].unique().tolist()
lunch_values   = X_train['lunch'].unique().tolist()
test_prep_vals = X_train['test preparation course'].unique().tolist()

num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler())
])

ord_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OrdinalEncoder(
        categories=[education_values, gender_values, lunch_values, test_prep_vals]
    ))
])

nom_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', num_transformer, num_features),
    ('nom', nom_transformer, nom_features),
    ('ord', ord_transformer, ord_features),
])

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', LinearRegression())
])

# ─────────────────────────────────────────
# 4. TRAIN MODEL
# ─────────────────────────────────────────
pipeline.fit(X_train, y_train)
print('Model trained successfully!')

# ─────────────────────────────────────────
# 5. EVALUATE
# ─────────────────────────────────────────
y_pred = pipeline.predict(X_test)

print('='*35)
print(f'  MAE : {mean_absolute_error(y_test, y_pred):.4f}')
print(f'  MSE : {mean_squared_error(y_test, y_pred):.4f}')
print(f'  R²  : {r2_score(y_test, y_pred):.4f}')
print('='*35)

# ─────────────────────────────────────────
# 6. VISUALIZATIONS
# ─────────────────────────────────────────

# Actual vs Predicted
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, alpha=0.5, color='steelblue', edgecolors='k', linewidths=0.3)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual Math Score')
plt.ylabel('Predicted Math Score')
plt.title('Actual vs Predicted — Math Score')
plt.legend()
plt.tight_layout()
plt.savefig('actual_vs_predicted.png', dpi=150)
plt.show()

# Residual Plot
residuals = y_test - y_pred
plt.figure(figsize=(8, 4))
plt.scatter(y_pred, residuals, alpha=0.5, color='tomato', edgecolors='k', linewidths=0.3)
plt.axhline(0, color='black', linestyle='--', lw=1.5)
plt.xlabel('Predicted Math Score')
plt.ylabel('Residual')
plt.title('Residual Plot')
plt.tight_layout()
plt.savefig('residual_plot.png', dpi=150)
plt.show()
