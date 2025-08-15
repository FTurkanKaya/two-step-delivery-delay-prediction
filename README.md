
## 📊 Machine Learning Workflow  

```mermaid
flowchart TD
    A[📥 Data Acquisition] --> B[🧹 Data Preprocessing - Merge datasets, handle missing values, feature engineering]
    B --> C[📊 Exploratory Data Analysis - Delivery time analysis, delay patterns, visualization]
    C --> D[🧠 Step 1: Classification Model - Predict if delayed, accuracy, precision, recall, F1-score]
    D --> E[⏳ Step 2: Regression Model - Predict delay duration (days), RMSE, MAE, R2 score]
    E --> F[📈 Insights & Reporting - Key delay factors, recommendations]
### Data Preprocessing

Merge datasets into a unified table.

Calculate delivery times and delays.

Create features such as distance, product category, and payment type.

### Step 1: Delay Classification

Target: on_time (0) vs. delayed (1).

Models tested: Logistic Regression, Random Forest, Gradient Boosting.

### Step 2: Delay Duration Prediction

Target: delay_days (numeric).

Models tested: Linear Regression, Random Forest Regressor, XGBoost.

## Evaluation Metrics

Classification: Accuracy, Precision, Recall, F1-score.

Regression: RMSE, MAE, R² score.

## Data Preparation
