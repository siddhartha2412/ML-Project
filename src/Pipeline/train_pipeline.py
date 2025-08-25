from src.Components.data_ingestion import DataIngestion
from src.Components.data_transformation import DataTransformation
from src.Components.model_trainer import ModelTrainer
from src.logger import logging

if __name__ == "__main__":
    logging.info("Pipeline started.")

    ingestion = DataIngestion()
    train_data, test_data = ingestion.initiate_data_ingestion()

    transformation = DataTransformation()
    train_arr, test_arr, preprocessor_path = transformation.initiate_data_transformation(train_data, test_data)

    trainer = ModelTrainer()
    model_score = trainer.initiate_model_trainer(train_arr, test_arr)

    print(f"Training complete ✅ | Model score: {model_score}")
    logging.info("Pipeline completed successfully.")
