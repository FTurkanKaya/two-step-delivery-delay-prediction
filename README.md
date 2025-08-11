# Two-Step Delivery Delay Prediction

## 📌 Overview  
This project was developed as part of a **Data Science training program provided by euroTech Study**.  
It focuses on a **machine learning-based delivery delay prediction** system for e-commerce orders.

Using the [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce), the project aims to:  
1. **Classify** whether an order will be delivered late.  
2. **Predict** the number of days the delivery will be delayed (if a delay is expected).

By applying a **two-step predictive modeling approach**, we aim to support logistics optimization, reduce delays, and improve customer satisfaction.

---

## 🎯 Project Goals  
- Understand e-commerce order and delivery patterns.  
- Build two machine learning models:  
  - **Classification model** → Predict if an order will be late.  
  - **Regression model** → Predict delay duration in days.  
- Extract actionable insights for business and logistics decision-making.

---

## 📂 Dataset Information  
The dataset contains multiple CSV files, including:

| File | Description |
|------|-------------|
| `olist_orders_dataset.csv` | Order purchase, shipping limit, and delivery timestamps. |
| `olist_order_items_dataset.csv` | Product details for each order. |
| `olist_order_reviews_dataset.csv` | Customer reviews and ratings. |
| `olist_customers_dataset.csv` | Customer location data. |
| `olist_sellers_dataset.csv` | Seller location data. |
| `olist_geolocation_dataset.csv` | Geolocation mapping for postal codes. |
| `olist_order_payments_dataset.csv` | Payment methods and amounts. |

Data preprocessing involves merging relevant tables, handling missing values, creating delivery time metrics, and computing distances between customers and sellers.

---

## 🛠️ Technologies Used  
- **Python**  
- **Pandas / NumPy** for data manipulation  
- **Matplotlib / Seaborn** for visualization  
- **Scikit-learn** for machine learning  
- **Jupyter Notebook** for analysis and development  
- **Kaggle API** for dataset access

---

## 📊 Machine Learning Workflow  

```mermaid
flowchart TD
    A[📥 Data Acquisition] --> B[🧹 Data Preprocessing - Merge datasets, handle missing values, feature engineering]
    B --> C[📊 Exploratory Data Analysis - Delivery time analysis, delay patterns, visualization]
    C --> D[🧠 Step 1: Classification Model - Predict if delayed, accuracy, precision, recall, F1-score]
    D --> E[⏳ Step 2: Regression Model - Predict delay duration (days), RMSE, MAE, R2 score]
    E --> F[📈 Insights & Reporting - Key delay factors, recommendations]

## Data Preparation

Merge datasets into one analytical table.

Calculate delivery times and determine delays.

Feature engineering: distances, product categories, payment types, etc.

Step 1: Delay Classification

Target: On-time (0) vs. Delayed (1).

Algorithms: Logistic Regression, Random Forest, Gradient Boosting.

Step 2: Delay Duration Prediction

Target: Delay days (numeric).

Algorithms: Linear Regression, Random Forest Regressor, XGBoost.

Model Evaluation

Classification: Accuracy, Precision, Recall, F1-score.

Regression: RMSE, MAE, R² score.

🚀 Expected Outcomes
Early detection of delays for proactive customer communication.

Accurate estimation of delay duration to optimize delivery operations.

Identification of key features influencing delivery performance.
