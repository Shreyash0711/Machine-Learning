# 🚗 Car Evaluation - Random Forest Classifier

A machine learning project that predicts the **acceptability of cars** (`unacc`, `acc`, `good`, `vgood`) using a **Random Forest classifier** trained on the UCI Car Evaluation dataset.

---

## 📊 Dataset Overview

- **Source**: UCI Machine Learning Repository  
- **Instances**: 1,728  
- **Features**:  
  - `buying`, `maint`, `doors`, `persons`, `lug_boot`, `safety`  
- **Target**: `class` → Acceptability (`unacc`, `acc`, `good`, `vgood`)

---

## ✅ Model Performance

- **Accuracy**: 97.39%
- **Classification Report**:

| Class    | Precision | Recall | F1-score |
|----------|-----------|--------|----------|
| acc      | 0.99      | 0.90   | 0.94     |
| good     | 0.65      | 1.00   | 0.79     |
| unacc    | 0.99      | 1.00   | 1.00     |
| vgood    | 1.00      | 0.94   | 0.97     |

