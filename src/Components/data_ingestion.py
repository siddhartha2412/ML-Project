import os
import sys
from pathlib import Path
from dataclasses import dataclass
import pandas as pd
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging

# Set project root dynamically
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join("artifacts", "raw.csv")
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")

class DataIngestion:
    def __init__(self, config: DataIngestionConfig = DataIngestionConfig()):
        self.config = config

    def _find_raw_csv(self) -> Path:
        try:
            candidate_dirs = [
                PROJECT_ROOT / "raw data",
                PROJECT_ROOT / "raw_data",
                PROJECT_ROOT / "data",
            ]
            for d in candidate_dirs:
                if d.exists() and d.is_dir():
                    preferred_file = d / "customer_clv_data.csv"
                    if preferred_file.exists():
                        return preferred_file
                    csv_files = sorted(d.glob("*.csv"))
                    if csv_files:
                        return csv_files[0]
            raise FileNotFoundError(
                f"No CSV found in directories: {[str(p) for p in candidate_dirs]}"
            )
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_ingestion(self):
        try:
            logging.info("Starting data ingestion...")
            csv_path = self._find_raw_csv()
            logging.info(f"Using raw data file: {csv_path}")
            df = pd.read_csv(csv_path)
            os.makedirs(os.path.dirname(self.config.raw_data_path), exist_ok=True)
            df.to_csv(self.config.raw_data_path, index=False)
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            train_set.to_csv(self.config.train_data_path, index=False)
            test_set.to_csv(self.config.test_data_path, index=False)
            logging.info("Data ingestion completed successfully.")
            return self.config.train_data_path, self.config.test_data_path
        except Exception as e:
            raise CustomException(e, sys)
