# Predictive-Analytics-Pipeline-for-Sports-Talent-Valuation

This project builds a supervised learning pipeline to predict the market value of professional football (soccer) players using their physical attributes, performance metrics, and club-level information. The goal is to create a robust, interpretable, and scalable model that can assist clubs, scouts, or analysts in estimating fair player transfer prices.

---

## 📌 Project Objective

- Build a predictive model to estimate a player's transfer market value.
- Compare the performance of multiple regression algorithms.
- Analyze the most influential features affecting player value.
- Provide an interface for dynamic prediction of unseen player profiles.

---

## 📁 Dataset

- **Size**: ~18,000 records
- **Features Include**:
  - Player attributes (age, height, weight)
  - Performance stats (rating, position, appearances)
  - Club and nationality data
  - Market value (target variable)

---

## 🛠️ Tech Stack

- **Language**: Python 3.x
- **Libraries**:  
  `scikit-learn`, `xgboost`, `pandas`, `numpy`, `matplotlib`, `seaborn`
- **Environment**: Jupyter Notebook

---

## 🔍 Exploratory Data Analysis (EDA)

- Distribution analysis of numerical features (histograms, box plots)
- Correlation heatmaps for feature relationships
- Detection of outliers and missing values
- Feature scaling and encoding

---

## 🤖 Machine Learning Models

Models evaluated using an 80/20 train-test split:
- Linear Regression
- Random Forest Regressor
- XGBoost Regressor ✅ *(Best Performer)*
- Gradient Boosting Regressor
- Support Vector Regressor (SVR)
- AdaBoost Regressor

### ✅ Best Model: **XGBoost**
- **R² Score**: `0.92`
- **MAE**: Low error rate across wide price ranges
- **Interpretability**: Feature importance plot for key drivers

---

## 📈 Performance Metrics

- **R² Score**
- **Mean Absolute Error (MAE)**
- **Root Mean Squared Error (RMSE)**
- **Cross-validated Scores** (optional extension)

---

## 🎯 Key Features

- Comparative analysis of regression models
- Feature importance ranking to interpret what drives player value
- Custom player input form for dynamic price predictions
- Modular and reusable pipeline structure

---

## 🚀 Future Enhancements

- Hyperparameter tuning using GridSearchCV or Optuna
- Deployment via **Streamlit** or **Flask**
- Integration of deep learning models (ANN, LSTM)
- Data enrichment from live APIs (e.g., FIFA ratings, Transfermarkt)

---
