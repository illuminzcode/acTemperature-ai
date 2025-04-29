# filename: ac_temp_ai_v2.py

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import joblib
import numpy as np
#from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
import os

app = FastAPI()

MODEL_PATH = "ac_temp_model.joblib"

class TrainData(BaseModel):
    crowd: List[int]
    ac_powers: List[List[float]]
    target_temp: List[float]

class AnalyzeData(BaseModel):
    crowd: int
    ac_powers: List[float]
    feedback: Optional[float] = None  # If user provides actual temp achieved

@app.post("/train")
def train(data: TrainData):
    X = []
    for people, acs in zip(data.crowd, data.ac_powers):
        total_ac_power = sum(acs)
        X.append([people, total_ac_power])

    y = data.target_temp

    #model = LinearRegression()
    model = SGDRegressor(max_iter=1000, tol=1e-3)
    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)

    return {"message": "Model trained successfully."}

@app.post("/analyze")
def analyze(data: AnalyzeData):
    if not os.path.exists(MODEL_PATH):
        return {"error": "Model not trained yet."}

    model = joblib.load(MODEL_PATH)

    total_ac_power = sum(data.ac_powers)
    input_features = np.array([[data.crowd, total_ac_power]])
    predicted_temp = model.predict(input_features)[0]
    predicted_temp = float(predicted_temp)
    if predicted_temp < 16 or predicted_temp > 30:
        predicted_temp = 24  # default fallback

    # Logic to check if AC setup is too weak
    avg_power_per_person = total_ac_power / max(data.crowd, 1)  # avoid division by 0

    if avg_power_per_person < 0.08:  # ~80W per person minimum needed
        suggestion = "AC setup is too weak for the crowd. Add more AC units or higher capacity units."
    else:
        suggestion = "AC setup is sufficient."

    # If user gives feedback, retrain slightly (continuous learning)
    if data.feedback is not None:
        X_new = np.array([[data.crowd, total_ac_power]])
        y_new = np.array([data.feedback])
        model.partial_fit(X_new, y_new)  # small update
        joblib.dump(model, MODEL_PATH)

    return {
        "recommended_temp_setting": round(predicted_temp, 2),
        "ac_suggestion": suggestion
    }
