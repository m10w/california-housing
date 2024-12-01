from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import numpy as np
from typing import Literal
import pandas as pd

# Load the preprocessing pipeline
pipeline = joblib.load('./models/best_model/preprocessing_pipeline_20241201-114702_v1.joblib')
# Load the trained model or pipeline
model = joblib.load('./models/best_model/best_model_lgb_20241201-114702_v1.joblib')

# Initialize FastAPI app
app = FastAPI()


# Define the input data schema
class InputData(BaseModel):
    longitude: float
    latitude: float
    housing_median_age: float
    total_rooms: float
    total_bedrooms: float
    population: float
    households: float
    median_income: float
    ocean_proximity: Literal['<1H OCEAN', 'NEAR BAY', 'NEAR OCEAN', 'INLAND', 'ISLAND']

# Define the root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the housing price prediction API"}


# Test endpoint
@app.post("/validate/")
def validate_input(data: InputData):
    return {"message": f"Input is valid: {data}"}

def add_engineered_features(data: pd.DataFrame) -> pd.DataFrame:
    # Calculate engineered features

    data['rooms_per_household'] = data['total_rooms'] / data['households'] # A measure of average house size (assumption: more rooms per household = larger house)
    data['bedrooms_per_rooms'] = data['total_bedrooms'] / data['total_rooms']
    data['income_per_population'] = data['median_income'] / data['population'] # Average median income per population
    data['rooms_per_population'] = data['total_rooms'] / data['population']
    data['households_per_population'] = data['households'] / data['population'] # Average household size

    return data

@app.post("/predict/")
def preprocess_and_predict(data: InputData):
    # Convert InputData to a DataFrame
    input_dict = data.dict()  # Convert Pydantic model to dictionary
    input_df = pd.DataFrame([input_dict])  # Convert to single-row DataFrame

    # Add engineered features to the DataFrame
    input_df = add_engineered_features(input_df)

    # Apply preprocessing pipeline to the entire dataset
    preprocessed_features = pipeline.transform(input_df)

    # Make prediction using the processed dataset
    prediction = model.predict(preprocessed_features)

    return {"prediction": prediction.tolist()}