import numpy as np
import torch
import itertools
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import shap
import plotly.graph_objects as go
import plotly.express as px
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from collections import defaultdict
import itertools
from sklearn.model_selection import RandomizedSearchCV
from catboost import CatBoostRegressor, Pool
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor

class LightGBMTrainer:
    def __init__(self, dataprep=None, n_estimators=100, random_state=42, max_depth=None, 
                 min_samples_split=2, horizons=[], cv=3, n_iter=50, scoring='neg_mean_absolute_error', 
                 verbose=1, random_search=True, param_distributions=None, model_folder='lgbm_models'):
        """
        Initializes the trainer with hyperparameters for LightGBM models and sets up the model saving directory.
        
        Parameters:
            dataprep (DataPreparationRF): Instance of DataPreparationRF used during data preparation.
            n_estimators (int): Number of boosting rounds.
            random_state (int): Seed used by the random number generator.
            max_depth (int or None): Maximum tree depth for base learners.
            min_samples_split (int): Minimum number of samples required to split an internal node.
            horizons (list): List of actual horizon values corresponding to timesteps.
            cv (int): Number of cross-validation folds for hyperparameter search.
            n_iter (int): Number of parameter settings that are sampled in RandomizedSearchCV.
            scoring (str): Scoring metric for hyperparameter search.
            verbose (int): Controls the verbosity: the higher, the more messages.
            random_search (bool): If True, use RandomizedSearchCV; else, use GridSearchCV.
            param_distributions (dict): Dictionary with parameter names (`str`) as keys and distributions or lists of parameters to try.
            model_folder (str): Directory path to save and load LightGBM models.
        """
        self.data_prep = dataprep
        self.models = {}  # Dictionary to store best LightGBM models per timestep
        self.best_params = {}  # Dictionary to store best parameters per timestep
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.horizons = horizons
        self.cv = cv
        self.n_iter = n_iter
        self.scoring = scoring
        self.verbose = verbose
        self.random_search = random_search
        self.model_folder = model_folder
        
        # Create the model directory if it doesn't exist
        if not os.path.exists(self.model_folder):
            os.makedirs(self.model_folder)
            print(f"Created model directory at: {self.model_folder}")
        else:
            print(f"Model directory already exists at: {self.model_folder}")
        
        # Default hyperparameter distributions if none provided
        if param_distributions is None:
            self.param_distributions = {
                'num_leaves': [31, 50, 70],
                'learning_rate': [0.01, 0.05, 0.1],
                'n_estimators': [ 300, 400, 500],
                'max_depth': [ 20, 30],
                'min_child_samples': [20, 40],
                'subsample': [ 0.8,1],
                'colsample_bytree': [ 0.8],
                'reg_alpha': [0, 1],
                'reg_lambda': [0, 1]
            }
        else:
            self.param_distributions = param_distributions

    def train(self, X_train, y_train_dict):
        """
        Trains a separate LightGBM model for each timestep with hyperparameter tuning and saves the models.
        
        Parameters:
            X_train (numpy.ndarray): Training features with shape (samples, timesteps, features).
            y_train_dict (dict): Dictionary with timestep as key and y_train array as value.
        """
        for t in range(len(self.horizons)):
            print(f"\n--- Training LightGBM for Timestep {self.horizons[t]} ---")
            
            # Initialize the LightGBM Regressor
            lgbm = LGBMRegressor(random_state=self.random_state)
            
            # Initialize RandomizedSearchCV
            random_search = RandomizedSearchCV(
                estimator=lgbm,
                param_distributions=self.param_distributions,
                n_iter=self.n_iter,
                cv=self.cv,
                verbose=self.verbose,
                random_state=self.random_state,
                n_jobs=-1,
                scoring=self.scoring
            )
            
            # Fit RandomizedSearchCV
            print("Performing hyperparameter search...")
            random_search.fit(X_train[:, t, :], y_train_dict)
            
            # Retrieve the best model and parameters
            best_lgbm = random_search.best_estimator_
            self.best_params[t] = random_search.best_params_
            
            print(f"Best Parameters for Timestep {self.horizons[t]}: {random_search.best_params_}")
            
            # Optionally, evaluate on training data (could use validation set instead)
            y_pred_scaled = best_lgbm.predict(X_train[:, t, :])
            y_pred = self.data_prep.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
            y_train_unscaled = self.data_prep.scaler_y.inverse_transform(y_train_dict.reshape(-1, 1)).flatten()
            
            mse = mean_squared_error(y_train_unscaled, y_pred)
            mae = mean_absolute_error(y_train_unscaled, y_pred)
            print(f"Timestep {self.horizons[t]} - Training MSE: {mse:.4f}, Training MAE: {mae:.4f}")
            
            # Save the trained model to the specified folder
            model_filename = f"lgbm_{self.horizons[t]}.pkl"
            model_path = os.path.join(self.model_folder, model_filename)
            with open(model_path, 'wb') as f:
                pickle.dump(best_lgbm, f)
            print(f"Saved model for Timestep {self.horizons[t]} at {model_path}")

    def evaluate(self, X_test, y_test_dict):
        """
        Loads each LightGBM model from the folder and evaluates it on the test set, printing MSE and MAE.
        
        Parameters:
            X_test (numpy.ndarray): Testing features with shape (samples, timesteps, features).
            y_test_dict (dict): Dictionary with timestep as key and y_test array as value.
        
        Returns:
            y_pred_dict (dict): Dictionary with timestep as key and y_pred array as value.
        """
        y_pred_dict = {}
        mae_per_timestep = {}
        
        for t in range(len(self.horizons)):
            # print(f"\n--- Evaluating LightGBM for Timestep {self.horizons[t]} ---")
            model_filename = f"lgbm_{self.horizons[t]}.pkl"
            model_path = os.path.join(self.model_folder, model_filename)
            
            if not os.path.exists(model_path):
                print(f"No saved model found for Timestep {self.horizons[t]} at {model_path}. Skipping evaluation.")
                continue
            
            # Load the model from the file
            with open(model_path, 'rb') as f:
                lgbm = pickle.load(f)
            print(f"Loaded model for Timestep {self.horizons[t]} from {model_path}")
            
            y_pred_scaled = lgbm.predict(X_test[:, t, :])
            y_pred = self.data_prep.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
            y_test_unscaled = self.data_prep.scaler_y.inverse_transform(y_test_dict.reshape(-1, 1)).flatten()
            
            mse = mean_squared_error(y_test_unscaled, y_pred)
            mae = mean_absolute_error(y_test_unscaled, y_pred)
            # print(f"Timestep {self.horizons[t]} - Test MSE: {mse:.4f}, Test MAE: {mae:.4f}")
            
            y_pred_dict[t] = y_pred
            mae_per_timestep[t] = mae
        
        # Plot MAE per timestep
        if mae_per_timestep:
            sorted_timesteps = sorted(mae_per_timestep.keys())
            mae_values = [mae_per_timestep[t] for t in sorted_timesteps]
            
            plt.figure(figsize=(12, 6))
            plt.plot([self.horizons[t] for t in sorted_timesteps], mae_values, marker='o', linestyle='-')
            plt.xticks([self.horizons[t] for t in sorted_timesteps][::4])
            plt.xlabel('Time Horizon')
            plt.ylabel('Mean Absolute Error (MAE)')
            plt.title('MAE for Each Time Horizon')
            plt.grid(True)
            plt.tight_layout()
            plt.show()
        else:
            print("No timesteps were evaluated.")
        
        return mae_values

    def plot_feature_importances(self, feature_names, top_n=20):
        """
        Plots feature importances for each timestep's LightGBM model.
        
        Parameters:
            feature_names (list): List of feature names.
            top_n (int): Number of top features to display.
        """
        for t in range(len(self.horizons)):
            model_filename = f"lgbm_timestep_{self.horizons[t]}.pkl"
            model_path = os.path.join(self.model_folder, model_filename)
            
            if not os.path.exists(model_path):
                print(f"No saved model found for Timestep {self.horizons[t]} at {model_path}. Skipping plotting.")
                continue
            
            # Load the model from the file
            with open(model_path, 'rb') as f:
                lgbm = pickle.load(f)
            print(f"\n--- Plotting Feature Importances for Timestep {self.horizons[t]} ---")
            
            importances = lgbm.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            plt.figure(figsize=(10, 6))
            plt.title(f"Feature Importances for Timestep {self.horizons[t]}")
            plt.bar(range(top_n), importances[indices][:top_n], color="b", align="center")
            plt.xticks(range(top_n), [feature_names[i] for i in indices[:top_n]], rotation=90)
            plt.xlim([-1, top_n])
            plt.ylabel('Importance Score')
            plt.tight_layout()
            plt.show()

class LGBMRollingForecaster:
    def __init__(self, model_folder, data_prep, scaled_flight_features, time_horizons):
        """
        :param model_folder: Directory where the LightGBM models are saved.
        :param data_prep: DataPreparation instance used during training (contains scalers).
        :param scaled_flight_features: Transformed (scaled) features for a specific flight (numpy array).
        :param time_horizons: List of time horizons (e.g., [-300, -295, ..., 0]).
        """
        self.model_folder = model_folder
        self.data_prep = data_prep
        self.scaled_features = scaled_flight_features  # Shape: (timesteps, features)
        self.time_horizons = time_horizons  # List of horizons corresponding to timesteps
        self.current_time_index = 0  # Index of the current time step in the features
        self.predictions = []  # Store predictions at each step
        self.models = self.load_models()  # Load all LightGBM models

    def reset(self):
        self.current_time_index = 0  # Index of the current time step in the features
        self.predictions = []  # Store predictions at each step

    def load_models(self):
        """
        Load the LightGBM models for each timestep.
        Returns a dictionary with horizon as key and model as value.
        """
        models = {}
        for t_idx, horizon in enumerate(self.time_horizons):
            model_filename = f"lgbm_{horizon}.pkl"
            model_path = os.path.join(self.model_folder, model_filename)
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    models[horizon] = pickle.load(f)
                # print(f"Loaded model for horizon {horizon} from {model_path}")
            # else:
                # print(f"Model file {model_path} not found. Skipping horizon {horizon}.")
        return models

    def rolling_forecast(self):
        """
        Perform rolling forecasts using the LightGBM models.
        """
        prediction = None
        num_timesteps = self.scaled_features.shape[0]
        for t in range(num_timesteps):
            # Update current time index
            self.current_time_index = t

            # Get the features at the current timestep
            current_features = self.scaled_features[t, :]  # Shape: (features,)
            if np.isnan(current_features).any():
                break
            def myround(x, base=5):
                return base * round(x/base)
            # Get the corresponding horizon
            # print(f'{t=}')
            # print(f'{(self.predictions[-1] if prediction else 0)=}')
            horizon=str(max(-300, myround(-300 + 5 *t - (self.predictions[-1] if prediction else 0))))
            # print(f'ttiltakeof = {(-300 + 5 *t - (self.predictions[-1] if prediction else 0))}')
            # print(f'{horizon=}')
            # Check if model exists for this horizon
            # print(f'{self.models=}')
            if horizon not in self.models:
                horizon = str(0)
                # print(f"No model found for horizon {horizon}. Skipping prediction.")

            # Get the model for the current horizon
            model = self.models[horizon]

            # Reshape features for prediction
            features_for_prediction = current_features.reshape(1, -1)  # Shape: (1, features)

            # Make prediction
            prediction_scaled = model.predict(features_for_prediction)
            # Inverse transform the prediction
            prediction = self.data_prep.scaler_y.inverse_transform(prediction_scaled.reshape(-1, 1)).flatten()[0]

            # Store the prediction
            self.predictions.append(prediction)

        return self.predictions