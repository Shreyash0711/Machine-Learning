# 🏦 Loan Amount Prediction using Random Forest

This project builds a machine learning model to **predict personal loan amounts** using customer financial and demographic data.

It includes:
- ✅ A version **with a preprocessing pipeline** (encapsulation of imputation, encoding, and scaling)
- ⚙️ A version **without a pipeline** (manual preprocessing)

---

## 📊 Dataset Overview
- 📦 Source: https://github.com/Sakil786/Calculating-credit-worthiness-for-rural-India-
- 🎯 **Target Variable**: `loan_amount`
- 🧾 **Key Features**: `annual_income`, `monthly_expenses`, `dependents`, `home_ownership`, etc.
- 🛠️ Includes feature engineering like:
  - `income_to_expense_ratio`
  - Combined dependents

---

## ✅ Model Evaluation

### 🔧 Without Pipeline (Manual Preprocessing)
| Metric       | Score             |
|--------------|------------------|
| **MAE**      | ₹1,331.88         |
| **MSE**      | ₹80,169,002.01    |
| **R² Score** | **0.6669**        |

---

### 📦 With Pipeline (Encapsulated Preprocessing)
| Metric       | Score             |
|--------------|------------------|
| **MAE**      | ₹1,312.50         |
| **MSE**      | ₹91,647,516.88    |
| **R² Score** | **0.6005**        |

> ✅ While the pipeline version is more portable and cleaner for deployment, the non-pipeline version slightly outperformed it in this case.

---

## 🧠 Techniques Used

- Feature Engineering
- Imputation with `SimpleImputer`
- One-Hot Encoding
- MinMax Scaling
- RandomForestRegressor from `sklearn`
