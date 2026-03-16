# Walmart Store Sales Forecasting with XGBoost

Retail sales forecasting is challenging because many factors influence demand: seasonality, holidays, promotions, weather, and economic conditions.  

This project builds an **end-to-end machine learning pipeline** to forecast **weekly Walmart store sales** using **Python, pandas, and XGBoost**.

The goal is to predict **`Weekly_Sales` for each `(Store, Dept, Date)` combination** and analyze which variables influence sales patterns.

---

# 1. Problem Overview

The dataset contains historical Walmart sales data with information about stores, departments, and external economic indicators.

Key characteristics of the dataset:

- 45 Walmart stores
- 99 departments
- Weekly sales values
- Store metadata (type and size)
- External economic factors
- Promotion markdown data
- Holiday indicators

The task is a **regression problem** where the model learns from historical data and predicts future weekly sales.

---

# 2. Data and Features

The model uses the following features:

- **Store** – Store identifier (encoded numerically)
- **Dept** – Department identifier (encoded numerically)
- **Weekly_Sales** – Target variable
- **Type** – Store type (A, B, C encoded as numeric values)
- **Size** – Store size in square feet
- **Temperature** – Average weekly temperature
- **Fuel_Price** – Regional fuel price
- **MarkDown1 – MarkDown5** – Promotional markdown variables
- **CPI** – Consumer Price Index
- **Unemployment** – Local unemployment rate
- **IsHoliday** – Binary indicator for holidays
- **Year** – Extracted from the Date column
- **Month** – Extracted from the Date column
- **Date_only** – Day index extracted from the Date column

The original `Date` column is transformed into calendar features and then removed.

---

# 3. Target Transformation

Weekly sales values are highly skewed and sometimes contain **negative values due to returns**.

To stabilize the learning process, a **signed logarithmic transformation** is applied:

```
y_trans = sign(y) * log(1 + |y|)
```

This transformation helps by:

- Preserving negative values
- Reducing the effect of extreme promotional spikes
- Making the learning problem easier for the model

Predictions are later converted back to the original scale using the inverse transformation.

---

# 4. Data Preprocessing

Preprocessing steps include:

- Cleaning missing values
- Converting categorical values to numeric format
- Converting `IsHoliday` to binary values (0 or 1)
- Encoding categorical features such as `Type`
- Extracting calendar features from the `Date` column
- Dropping unused columns

All preprocessing steps are implemented using a **scikit-learn Pipeline** so the same transformations are applied consistently to training and test data.

Feature scaling is not required because **tree-based models such as XGBoost do not depend on feature scaling**.

---

# 5. Model

The model used in this project is **XGBoost Regressor**, a gradient boosting tree algorithm designed for structured tabular data.

Example configuration:

```python
from xgboost import XGBRegressor

model = XGBRegressor(
    objective="reg:squarederror",
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
```

These parameters provide a balanced starting configuration and can be further tuned using hyperparameter search methods.

---

# 6. Training and Evaluation

The dataset is split into **training and test sets**.

The model is trained on the **transformed target variable**.

Evaluation metrics used:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

Results obtained on the test set:

```
MAE  ≈ 0.272
MSE  ≈ 0.209
RMSE ≈ 0.458
R²   ≈ 0.953
```

These metrics indicate that the model explains roughly **95% of the variance** in the transformed sales values.

---

# 7. Visual Diagnostics

To analyze prediction performance, several plots are used:

- True vs Predicted line plots (on sampled test data)
- Scatter plot of actual vs predicted sales
- Error distribution plots
- Residual analysis

Sampling 200–500 points helps keep visualizations readable without overcrowding the charts.

---

# 8. How to Run the Project

Install the required dependencies:

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn
```

Example workflow:

```python
import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("train.csv")

# Target transformation
y = df["Weekly_Sales"]
y_trans = np.sign(y) * np.log1p(np.abs(y))

X = df.drop(columns=["Weekly_Sales"])

# Train model
model.fit(X_train, y_train_trans)

# Predict
y_pred_trans = model.predict(X_test)

# Inverse transform predictions
y_pred = np.sign(y_pred_trans) * np.expm1(np.abs(y_pred_trans))
```

---

# 9. Possible Improvements

Future improvements may include:

- Time-based cross validation instead of random splitting
- Adding more detailed holiday features
- Hyperparameter optimization
- Separate models for each store or department
- Testing other gradient boosting models
- Trying deep learning or time-series forecasting models

---

# 10. Key Takeaways

- XGBoost performs well on structured tabular datasets.
- Target transformation improves stability when sales data is highly skewed.
- Calendar features and promotional markdown variables are strong predictors of retail demand.
- Proper preprocessing and feature engineering significantly improve model performance.

This project demonstrates a complete workflow for **machine learning based retail sales forecasting**, from preprocessing and feature engineering to modeling and evaluation.