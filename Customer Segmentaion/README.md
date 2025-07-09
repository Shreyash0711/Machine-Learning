# 🧠 Customer Segmentation Using K-Means Clustering

This project applies **unsupervised machine learning** techniques to segment customers based on their demographics and purchasing behavior. Using the **Mall Customer Segmentation Dataset**, we cluster customers into actionable groups for targeted marketing.

---

## 📌 Overview

- **Goal:** Identify distinct customer segments for personalized marketing strategies.
- **Technique:** K-Means Clustering with Silhouette Score validation and t-SNE for visualization.
- **Dataset:** [Mall Customers Dataset (Kaggle)](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation)

---

## 🛠 Features Used

- `Gender`
- `Age`
- `Annual Income (k$)`
- `Spending Score (1-100)`

---

## 📈 Workflow

1. **Data Preprocessing**
   - Encode Gender
   - Standardize features using `StandardScaler`

2. **Clustering**
   - K-Means Clustering (Optimal K = 10)
   - Elbow Method and Silhouette Score for K selection

3. **Evaluation**
   - Silhouette Score for each K (2 to 10)
   - Final Score for K=10: **0.405**

4. **Visualization**
   - Seaborn: Age vs Spending Score
   - Plotly: Interactive 2D plot
   - t-SNE: Cluster visualization in 2D

5. **Segment Profiling**
   - Created 10 customer personas based on Age, Income, and Spending

---

## 📊 Cluster Segments

| Cluster | Label                   | Age   | Income | Spending |
|--------|--------------------------|-------|--------|----------|
| 0      | Mature Balanced          | 59.3  | 54.3   | 49.2     |
| 1      | Conservative Elders      | 52.6  | 47.5   | 42.2     |
| 2      | Affluent Enthusiasts     | 33.3  | 87.1   | 82.7     |
| 3      | Young Spenders           | 24.8  | 42.0   | 62.0     |
| 4      | Budget Big Spenders      | 25.5  | 25.7   | 80.5     |
| 5      | High Earners Spenders    | 32.2  | 86.0   | 81.7     |
| 6      | Affluent but Frugal      | 39.5  | 85.2   | 14.1     |
| 7      | Average Millennials      | 28.5  | 51.4   | 43.0     |
| 8      | Low-Income Seniors       | 52.0  | 25.9   | 17.3     |
| 9      | Wealthy Non-Spenders     | 44.6  | 92.3   | 21.6     |

---

## ✅ Business Insights

- **VIP Segments:** Cluster 2 & 5 (High income + High spend)
- **Growth Potential:** Cluster 6 & 9 (High income but low spend)
- **Discount Seekers:** Cluster 3 & 4 (Low income but spend heavily)
- **Low ROI Group:** Cluster 8 (Low income, low spend)
