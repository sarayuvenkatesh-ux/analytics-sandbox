**Lead Scoring Optimization using Logistic Regression**

**🎯 Project Overview**

This project builds a predictive logistics regression model to assign a lead score between 0 and 100 to incoming marketing leads. The objective is to help the sales team identify "Hot Leads" (highly likely to convert), boosting the company's overall lead conversion rate from a baseline of around 30% up to an efficient target of 80%.

**🛠️ Machine Learning Workflow**

The Python notebook steps through a comprehensive end-to-end data science pipeline:

[Data Cleaning & Missing Values] ➔ [EDA & Outlier Treatment] ➔ [Feature Scaling & Dummy Variables] ➔ [RFE Feature Selection] ➔ [Model Evaluation (ROC, Precision-Recall)]

**Data Cleaning & Preprocessing:**

Handled missing/null values and replaced generic placeholders (like 'Select') with actionable data data points.

Treated numerical outliers to protect model stability.

**Feature Engineering:**

Categorical variables were converted into numerical formats using dummy variables.

Scaled features using StandardScaler to ensure coefficients are comparable.

**Model Building & Selection:**

Utilized Recursive Feature Elimination (RFE) alongside Variance Inflation Factor (VIF) checking to systematically eliminate multi-collinearity and optimize features.

Built the final GLM Logistic Regression model using statsmodels.

**Model Evaluation & Optimization:**

Balanced model sensitivity, specificity, and accuracy by plotting an ROC Curve and calculating the Precision-Recall tradeoff.

Selected an optimal cutoff threshold to isolate true conversion likelihood.

**📦 Key Libraries Used**

Data Processing: pandas, numpy

Data Visualization: matplotlib, seaborn

Statistical Modeling: statsmodels.api, sklearn.linear_model

Evaluation Metrics: sklearn.metrics (confusion_matrix, precision_recall_curve, roc_curve)
