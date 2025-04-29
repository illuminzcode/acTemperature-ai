# 🧠 Smart AC Temperature Predictor (AI-Driven)

This project uses machine learning to recommend optimal AC temperature settings based on:
- Number of people in the room
- Number of AC units
- Total power of AC units

It provides a FastAPI interface to:
- Train the model with new data
- Analyze crowd data and return recommended AC temperature

---

## 🚀 Features

- ✅ FastAPI endpoints for training and prediction
- ✅ Uses `SGDRegressor` for incremental learning (`partial_fit`)
- ✅ Saves and loads model using `joblib`
- ✅ Input validation and safe defaults
- ✅ Real-time model improvement with feedback loop

---

## 🧩 Example Use Case

A smart building system tracks:
- Room occupancy
- AC specs (count and total wattage)

And automatically adjusts the AC temperature setting via this API.

---

## 📦 Tech Stack

- Python 3.10+
- FastAPI
- scikit-learn (SGDRegressor)
- joblib
- Uvicorn

---

## 📁 Project Structure

acTemperature/ 
├── ac_temp_ai.py # Main FastAPI app 
├── model.joblib # Trained ML model 
├── requirements.txt # Python dependencies 
└── README.md # This file


/analyze — Predict recommended temperature
POST /analyze
{
  "num_people": 15,
  "num_ac": 3,
  "total_ac_power": 4500
}

Response:
{
  "recommended_temp_setting": 21.5,
  "ac_suggestion": "AC setup is sufficient."
}

🛠️ Setup Instructions

# 1. Clone the repo
git clone https://github.com/yourname/acTemperature-ai.git
cd acTemperature-ai

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the server
uvicorn ac_temp_ai:app --reload