# 🎓 Student Math Score Prediction — Linear Regression Pipeline

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange?logo=scikit-learn)
![Task](https://img.shields.io/badge/Task-Regression-purple)
![Dataset](https://img.shields.io/badge/Dataset-Student%20Performance-lightgrey)

---

## 📌 Project Overview

Predict a student's **math score** from demographic and academic preparation features using a full scikit-learn Pipeline.

| Item | Detail |
|------|--------|
| **Dataset** | Students Performance in Exams (1,000 students) |
| **Model** | Linear Regression |
| **Target** | `math score` (continuous, 0–100) |
| **Key Skill** | sklearn Pipeline with mixed feature types |

---

## 🗂️ Project Structure

```
02_student_score_regression/
├── main.ipynb        ← Full notebook: EDA → Pipeline → Train → Evaluate
├── dataset_info.md   ← Download link + column descriptions
└── README.md         ← This file
```

---

## 🚀 How to Run

### Step 1 — Install dependencies
```bash
pip install pandas numpy scikit-learn matplotlib seaborn openpyxl
```

### Step 2 — Download dataset
👉 [Students Performance in Exams — Kaggle](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams)

Place `StudentsPerformance.csv` or `StudentsPerformance.xlsx` in this folder.

### Step 3 — Open notebook
```bash
jupyter notebook main.ipynb
```
Then click **Kernel → Restart & Run All**

---

## ⚙️ ML Pipeline

```
StudentsPerformance.csv / .xlsx
          │
          ▼
  Train / Test Split (80% / 20%)
          │
          ▼
    ColumnTransformer
  ┌───────────────────────────────────────┐
  │ Numerical  → Imputer + StandardScaler │
  │ Ordinal    → Imputer + OrdinalEncoder │
  │ Nominal    → Imputer + OneHotEncoder  │
  └───────────────────────────────────────┘
          │
          ▼
    LinearRegression
          │
          ▼
  MAE / MSE / R² + Visualizations
```

---

## 📊 Results

| Metric | Score |
|--------|-------|
| MAE | ~4.5 |
| R²  | ~0.87 |

**Charts generated:**
- 📊 Math Score Distribution
- 📦 Score by Gender
- 📦 Score by Test Preparation Course
- 📦 Score by Parental Education Level
- 🔵 Actual vs Predicted Scatter Plot
- 🔴 Residual Plot

---

## 🔑 Key Concepts

- ✅ sklearn Pipeline (prevents data leakage)
- ✅ ColumnTransformer for mixed feature types
- ✅ OrdinalEncoder for ordered categories (education level)
- ✅ OneHotEncoder for nominal categories (race/ethnicity)
- ✅ SimpleImputer for missing value handling

---

## 📦 Dependencies

```
pandas
numpy
scikit-learn
matplotlib
seaborn
openpyxl
```
