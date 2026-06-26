# 🚗 Car Price Prediction using Linear Regression (Geely Auto)

## 🎯 Project Overview
This project builds a predictive Linear Regression model to understand the factors driving automobile pricing in the US market. The objective is to help Geely Auto, an aspiring market entrant, identify which vehicle features (e.g., horsepower, engine size, car body style) significantly impact pricing so they can design and position their cars competitively.

---

## 🛠️ Data Science Workflow

1. **Data Cleaning & Exploration:** Handled categorical naming anomalies (e.g., fixing typos in car company names) and analyzed feature correlations.
2. **Feature Engineering:** Created dummy variables for categorical specifications like fuel type, aspiration, and engine location.
3. **Model Construction:** Used a combination of **Recursive Feature Elimination (RFE)** and manual pruning via **P-values** and **VIF (Variance Inflation Factor)** to keep the model highly interpretable.
4. **Residual Analysis & Validation:** Verified the assumptions of linear regression (error terms are normally distributed with constant variance) to ensure model reliability.

---

## 📦 Key Libraries Used
* `pandas` & `numpy` (Data manipulation)
* `matplotlib` & `seaborn` (Data visualization)
* `sklearn` (RFE and model evaluation)
* `statsmodels` (Detailed OLS regression summaries)
