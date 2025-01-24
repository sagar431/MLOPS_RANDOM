import numpy as np
import pandas as pd
import pickle
import os
from sklearn.ensemble import GradientBoostingClassifier
import yaml
import logging

# logging configuration
logger = logging.getLogger('model_building')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('model_building_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise

def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        logger.debug('Data loaded from %s', file_path)
        # Print column names for debugging
        logger.debug(f'Columns in dataset: {df.columns.tolist()}')
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise

def save_model(model, file_path: str) -> None:
    """Save the trained model to a file."""
    try:
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
        logger.debug('Model saved to %s', file_path)
    except Exception as e:
        logger.error('Error occurred while saving the model: %s', e)
        raise

def main():
    try:
        # Load parameters
        params = load_params('params.yaml')
        n_estimators = params['model_building']['n_estimators']
        learning_rate = params['model_building']['learning_rate']

        # Load training data
        train_data = load_data('./data/processed/train_tfidf.csv')
        
        # Assuming the last column is the target variable
        X = train_data.iloc[:, :-1]  # All columns except the last one
        y = train_data.iloc[:, -1]   # Last column
        
        logger.debug(f'Features shape: {X.shape}')
        logger.debug(f'Target shape: {y.shape}')

        # Initialize and train the model
        model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=42
        )
        model.fit(X, y)
        logger.debug('Model training completed')

        # Save the trained model
        save_model(model, 'models/model.pkl')

    except Exception as e:
        logger.error('Failed to complete the model building process: %s', e)
        print(f"Error: {e}")
        raise

if __name__ == '__main__':
    main()
