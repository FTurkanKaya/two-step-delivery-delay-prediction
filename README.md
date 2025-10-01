
# Two-Step Delivery Delay Prediction (Classification Only)

A machine learning project for predicting e-commerce delivery delays:
Classifying whether an order will be delayed.

Built using the Brazilian E-Commerce Public Dataset by Olist and visualized with Power BI dashboards for interactive exploration of the results.

## Overview

This project was developed as part of a Data Science training program provided by Study. It focuses on a machine learning-based delivery delay prediction system for e-commerce orders.

Using the Brazilian E-Commerce Public Dataset by Olist, the project aims to:

Classify whether an order will be delivered late.

Provide interactive dashboards in Power BI to explore key insights and support business decisions.

By applying predictive modeling, we aim to support logistics optimization, reduce delays, and improve customer satisfaction.

## Project Goals

Understand e-commerce order and delivery patterns.

Build a classification model to predict late deliveries.

Extract actionable insights for business and logistics decision-making.

Visualize key findings and model results in Power BI dashboards for interactive analysis.

## Dataset Information

The dataset contains multiple CSV files, including:

File	Description
olist_orders_dataset.csv	Order purchase, shipping limit, and delivery timestamps
olist_order_items_dataset.csv	Product details for each order
olist_order_reviews_dataset.csv	Customer reviews and ratings
olist_customers_dataset.csv	Customer location data
olist_sellers_dataset.csv	Seller location data
olist_geolocation_dataset.csv	Geolocation mapping for postal codes
olist_order_payments_dataset.csv	Payment methods and amounts

##Data preprocessing involves:

Merging relevant tables into a single analytical table

Handling missing values

Creating delivery time metrics and flags (is_late, delay_class)

Computing distances between customers and sellers

Feature engineering (product categories, payment types, seller counts, etc.)

## Technologies Used

Python: Pandas, NumPy for data manipulation

Matplotlib / Seaborn: Visualizations

Scikit-learn: Machine learning models

Jupyter Notebook: Analysis and development

Power BI: Interactive dashboards for visual exploration of model outputs and business insights

Kaggle API: Dataset access

## Machine Learning Workflow
Step 0: Data Preparation

Merge datasets into one analytical table

Calculate delivery times and determine delays

Feature engineering: distances, product categories, payment types, etc.

Step 1: Delay Classification

Target: On-time (0) vs. Delayed (1)

Algorithms: Logistic Regression, Random Forest, Gradient Boosting, XGBoost, LightGBM

Evaluation Metrics: Accuracy, Precision, Recall, F1-score, ROC-AUC

<img width="300" height="200" alt="image" src="https://github.com/user-attachments/assets/531aad6c-92e1-4dfd-accf-5be4cde293c5" /> <br>

<img width="300" height="200" alt="image" src="https://github.com/user-attachments/assets/81c24cf9-1c4b-41ab-88bb-048ebaeebcf0" /> <br>

<img width="300" height="200" alt="image" src="https://github.com/user-attachments/assets/96298e30-ac57-4def-aed2-e4eda85f36bd" /><br>



## Power BI Dashboards

Interactive visualizations of delay distributions, customer/seller behavior, distance effects, product categories, and payment types

Comparison of model predictions vs. actual outcomes

Filtering and slicing by city, product, seller, and payment type to uncover actionable insights

Supports business users in monitoring logistics performance and prioritizing at-risk orders

<img width="228" height="90" alt="image" src="https://github.com/user-attachments/assets/93ac566c-bcf8-47bb-a158-059f6cfa503a" />

<img width="203" height="98" alt="image" src="https://github.com/user-attachments/assets/b2d30721-a046-4093-aba1-0a3088cb6f64" />

<img width="203" height="94" alt="image" src="https://github.com/user-attachments/assets/a6cc4bbb-bc9a-4461-80b6-65ce988090bd" /><br>

<img width="209" height="92" alt="image" src="https://github.com/user-attachments/assets/53de320b-0fd6-4f29-a86c-d3771241a0f3" />

<img width="179" height="98" alt="image" src="https://github.com/user-attachments/assets/1ad05957-bfc5-426a-88e0-de2d2e88796b" />

<img width="226" height="98" alt="image" src="https://github.com/user-attachments/assets/7c44acfd-ed25-440b-8000-f7e4b05239b9" /><br>





## Expected Outcomes

Early detection of delays for proactive customer communication

Identification of key features influencing delivery performance

Interactive Power BI dashboards enable stakeholders to explore insights dynamically and make data-driven decisions.
