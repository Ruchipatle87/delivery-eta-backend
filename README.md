# Delivery ETA Backend

A FastAPI service that serves a self-trained machine learning model predicting food delivery time, based on real-world delivery operations data.

This is the backend half of a two-part project. See the frontend here: [delivery-eta-frontend](https://github.com/Ruchipatle87/delivery-eta-frontend)

---

## What this does

Given order details (distance, weather, traffic, rider info, etc.), this API returns a predicted delivery time in minutes — powered by a Random Forest Regressor trained from scratch, not a third-party AI API.

---

## Problem

Predicting delivery time accurately helps dispatch systems set realistic customer expectations and identify operational bottlenecks (traffic, weather, rider load). This project builds that prediction pipeline end-to-end: data → trained model → deployed API.

---

## Dataset

Zomato Delivery Operations Analytics Dataset (Kaggle) — ~45,000 real delivery records including distance, weather, road traffic density, rider ratings, vehicle type, and actual delivery time.

---

## Approach

1. Feature engineering — calculated real distance between restaurant and delivery location using the haversine formula; extracted order hour and day-of-week from timestamps
2. Preprocessing — handled missing values, label-encoded categorical features
3. Model comparison — trained and evaluated three models:

| Model | MAE | RMSE | R2 |
|---|---|---|---|
| Linear Regression | 5.64 | 7.01 | 0.443 |
| Random Forest | 3.13 | 3.95 | 0.823 |
| XGBoost | 3.18 | 4.01 | 0.817 |

4. Hyperparameter tuning — GridSearchCV confirmed Random Forest's default settings were already near-optimal (tuned MAE: 3.13, R2: 0.824)
5. Explainability — used SHAP to confirm road traffic density, rider ratings, and distance were the strongest predictors, validating the model against real-world intuition

Model chosen: Random Forest Regressor (best MAE/R2 trade-off, more stable than XGBoost on this dataset size)

---

## Tech stack

- FastAPI — REST API framework
- scikit-learn — model training (Random Forest, Linear Regression)
- XGBoost — comparison model
- SHAP — model explainability
- Pandas / NumPy — data processing
- Pydantic — request validation

---

## API

### POST /predict

Request body:

{ "delivery_person_age": 25, "delivery_person_rating": 4.5, "weather": "Sunny", "traffic": "Medium", "vehicle_condition": 1, "order_type": "Snack", "vehicle_type": "motorcycle", "multiple_deliveries": 1, "festival": "No", "city": "Urban", "distance_km": 5.2, "order_day_of_week": 2, "order_hour": 19 }

Response:

{ "predicted_time_minutes": 14.8 }

Invalid categorical values (e.g. an unrecognized weather type) return a clean 400 error listing valid options, instead of crashing.

---

## Running locally

pip install -r requirements.txt
uvicorn app:app --reload

Visit http://127.0.0.1:8000/docs for interactive API testing (Swagger UI).

---

## Project structure

delivery-time-api/
- app.py (FastAPI app + prediction endpoint)
- delivery_time_model.pkl (trained Random Forest model)
- label_encoders.pkl (saved encoders for categorical fields)
- requirements.txt
