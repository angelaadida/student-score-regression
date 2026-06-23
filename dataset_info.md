# 📊 Dataset Info — Students Performance in Exams

## 🔗 Download Link

👉 **Kaggle:** https://www.kaggle.com/datasets/spscientist/students-performance-in-exams

> **How to download:**
> 1. Go to the link above
> 2. Click the **Download** button (top right)
> 3. Save file as `StudentsPerformance.csv` or `StudentsPerformance.xlsx`
> 4. Place it in the same folder as `main.ipynb`

---

## 📋 Column Descriptions

| Column | Type | Description |
|--------|------|-------------|
| `gender` | str | male / female |
| `race/ethnicity` | str | Group A, B, C, D, or E |
| `parental level of education` | str | Highest education level of parents |
| `lunch` | str | standard / free or reduced |
| `test preparation course` | str | completed / none |
| `reading score` | int | Reading test score (0–100) |
| `writing score` | int | Writing test score (0–100) |
| `math score` | int | **Target** — Math test score (0–100) |

- **Total rows:** 1,000 students
- **Missing values:** None
- **Task type:** Regression (predict continuous value)

---

## 💡 Key Insights from EDA

- Students who **completed test preparation** score ~5–10 points higher
- **Reading and writing scores** are the strongest predictors of math score
- **Parental education level** positively correlates with student performance
- **Gender** shows slight difference — males tend to score higher in math

---

## ⚠️ Note on File Format

This notebook **automatically detects** your file format:
- ✅ `StudentsPerformance.csv` → works directly
- ✅ `StudentsPerformance.xlsx` → also works (no need to rename!)
