

# 🚲 Bike Rental Demand Predictor

**Predict daily bike rentals using Random Forest Regression — a real-world, CV-ready machine learning project with a user-friendly web interface.**

---

## Table of Contents

* [Project Overview](#project-overview)
* [Motivation](#motivation)
* [Dataset](#dataset)
* [Problem Statement](#problem-statement)
* [Features](#features)
* [Model & Methodology](#model--methodology)
* [Implementation](#implementation)
* [Application](#application)
* [Installation](#installation)
* [Usage](#usage)
* [Results](#results)
* [Author](#author)

---

## Project Overview

This project forecasts **bike rental demand** for urban areas using **Random Forest Regression**, leveraging historical rental data and weather conditions. The model predicts rentals with high accuracy and the results can help in **resource allocation, planning, and urban mobility optimization**.

---

## Motivation

* Demonstrates **end-to-end machine learning skills**: data preprocessing, feature engineering, modeling, evaluation, and deployment.
* Provides insights into **factors influencing bike rentals**, such as weather, day of the week, and holidays.
* Enhances CV with a **practical, real-world predictive model**.

---

## Dataset

* **Source:** [UCI Bike Sharing Dataset](https://archive.ics.uci.edu/ml/datasets/bike+sharing+dataset)
* **Contents:** Hourly/daily bike rentals with features including:

  * Date & time
  * Weather conditions: temperature, humidity, wind speed, weather situation
  * Holiday & working day indicators
  * Season
* **Target Variable:** `cnt` (total bike rentals)

---

## Problem Statement

Predict the total number of bike rentals for a given day/hour using relevant features.

**Use Cases:**

* Optimize bike distribution across stations
* Improve urban transportation planning
* Provide insights into **weather and seasonal impacts** on bike usage

---

## Features

| Feature      | Description                               |
| ------------ | ----------------------------------------- |
| `season`     | Season of the year (Spring, Summer, etc.) |
| `year`       | Year (2011, 2012)                         |
| `month`      | Month (1–12)                              |
| `weekday`    | Day of the week (0–6)                     |
| `holiday`    | Whether the day is a holiday (0/1)        |
| `workingday` | Whether it’s a working day (0/1)          |
| `weather`    | Weather condition code                    |
| `temp`       | Normalized temperature                    |
| `atemp`      | Feels-like temperature                    |
| `hum`        | Humidity                                  |
| `windspeed`  | Wind speed                                |
| `cnt`        | Target: total rentals                     |

---

## Model & Methodology

**Model:** Random Forest Regressor

* Handles numeric and categorical data efficiently
* Reduces overfitting via ensemble learning
* Provides **feature importance** insights

**Methodology:**

1. Data Cleaning & Missing Value Handling
2. Feature Engineering (Date to year/month/day, categorical encoding)
3. Data Splitting: Training & Test sets
4. Model Training & Hyperparameter Tuning
5. Model Evaluation (RMSE, R² Score)
6. Model Deployment via Streamlit

---

## Implementation

```python
from sklearn.ensemble import RandomForestRegressor
import joblib

# Train model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

# Save model
joblib.dump(rf_model, "bike_rental_rf_model.pkl")
```

* Data visualizations: rental trends, seasonality, correlation heatmaps
* Streamlit app for interactive predictions

---

## Application

* Input parameters: weather, date, season, holiday/working day
* Generates **predicted bike rental count** instantly
* Scenario analysis to explore **“what-if” conditions**
* Suitable for **urban planners, transportation services, and bike-sharing platforms**

---

## Installation

```bash
# Clone repository
git clone https://github.com/siddhant754962/bike-rental-demand-predictor.git
cd bike-rental-demand-predictor

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

```bash
# Run Streamlit web app
streamlit run app.py
```

* Enter inputs manually or fetch live weather data
* Click **Predict** to see results

---

## Results

* **Accuracy:** R² Score \~0.85–0.90
* **RMSE:** \~700–900 rentals
* Features such as **temperature, season, and weekday** strongly influence bike demand
* Professional **web app deployment** enhances CV impact

---

## Author

**Siddhant Patel**

* GitHub: [github.com/siddhant754962](https://github.com/siddhant754962)
* Email: [kumarsidhant144@gmail.com](mailto:kumarsidhant144@gmail.com)







