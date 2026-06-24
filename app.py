from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("delivery_time_model.pkl")
encoders = joblib.load("label_encoders.pkl")

class OrderInput(BaseModel):
    delivery_person_age: float
    delivery_person_rating: float
    weather: str
    traffic: str
    vehicle_condition: int
    order_type: str
    vehicle_type: str
    multiple_deliveries: float
    festival: str
    city: str
    distance_km: float
    order_day_of_week: int
    order_hour: int

def safe_encode(encoder, value, field_name):
    if value not in encoder.classes_:
        valid_options = ", ".join(encoder.classes_)
        raise HTTPException(
            status_code=400,
            detail=f"Invalid value '{value}' for '{field_name}'. Must be one of: {valid_options}"
        )
    return encoder.transform([value])[0]

@app.post("/predict")
def predict(order: OrderInput):
    row = [
        order.delivery_person_age,
        order.delivery_person_rating,
        safe_encode(encoders['Weather_conditions'], order.weather, "weather"),
        safe_encode(encoders['Road_traffic_density'], order.traffic, "traffic"),
        order.vehicle_condition,
        safe_encode(encoders['Type_of_order'], order.order_type, "order_type"),
        safe_encode(encoders['Type_of_vehicle'], order.vehicle_type, "vehicle_type"),
        order.multiple_deliveries,
        safe_encode(encoders['Festival'], order.festival, "festival"),
        safe_encode(encoders['City'], order.city, "city"),
        order.distance_km,
        order.order_day_of_week,
        order.order_hour
    ]
    prediction = model.predict([row])[0]
    return {"predicted_time_minutes": round(float(prediction), 1)}