# ---------------- Predict Pipeline ----------------
"""
Predict Pipeline: Load trained model and preprocessor to make predictions on new data.
"""

import os
import sys
import pandas as pd
from dataclasses import dataclass
from src.exception import CustomException
from src.utils import load_object
from src.logger import logging

@dataclass
class CustomData:
    Age: float
    Gender: str
    Location: str
    IncomeLevel: str
    PurchaseFrequency: float
    AvgOrderValue: float
    RecencyDays: float
    TenureDays: float
    Churned: int

    def get_data_as_data_frame(self) -> pd.DataFrame:
        """
        Converts the CustomData instance into a pandas DataFrame for prediction.
        """
        try:
            data_dict = {
                "Age": [self.Age],
                "Gender": [self.Gender],
                "Location": [self.Location],
                "IncomeLevel": [self.IncomeLevel],
                "PurchaseFrequency": [self.PurchaseFrequency],
                "AvgOrderValue": [self.AvgOrderValue],
                "RecencyDays": [self.RecencyDays],
                "TenureDays": [self.TenureDays],
                "Churned": [self.Churned]
            }
            return pd.DataFrame(data_dict)
        except Exception as e:
            raise CustomException(e, sys)


class PredictPipeline:
    def __init__(self):
        try:
            self.model_path = os.path.join("artifacts", "model.pkl")
            self.preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

            logging.info("Loading model and preprocessor for predictions...")
            self.model = load_object(self.model_path)
            self.preprocessor = load_object(self.preprocessor_path)
        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, features: pd.DataFrame):
        """
        Predict CLV using trained model and preprocessor.
        """
        try:
            data_scaled = self.preprocessor.transform(features)
            predictions = self.model.predict(data_scaled)
            return predictions
        except Exception as e:
            raise CustomException(e, sys)
