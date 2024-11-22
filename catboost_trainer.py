
import matplotlib.pyplot as plt
import os
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV
from catboost import CatBoostRegressor, Pool
import numpy as np


class CatBoostTrainer:
    def __init__(self, dataprep=None, iterations=1000, random_state=42, depth=6, 
                 learning_rate=0.1, l2_leaf_reg=3, early_stopping_rounds=50, 
                 horizons=[], cv=3, n_iter=50, scoring='neg_mean_absolute_error', 
                 verbose=100, random_search=True, param_distributions=None, model_folder='catboost_models'):
        """
        Initializes the trainer with hyperparameters for CatBoost models and sets up the model saving directory.
        
        Parameters:
            dataprep (DataPreparationRF): Instance of DataPreparationRF used during data preparation.
            iterations (int): Number of boosting iterations.
            random_state (int): Seed used by the random number generator.
            depth (int): Depth of the tree.
            learning_rate (float): Learning rate.
            l2_leaf_reg (float): L2 regularization term on weights.
            early_stopping_rounds (int): Number of rounds with no improvement to stop training.
            horizons (list): List of actual horizon values corresponding to timesteps.
            cv (int): Number of cross-validation folds for hyperparameter search.
            n_iter (int): Number of parameter settings that are sampled in RandomizedSearchCV.
            scoring (str): Scoring metric for hyperparameter search.
            verbose (int): Controls the verbosity: the higher, the more messages.
            random_search (bool): If True, use RandomizedSearchCV; else, use GridSearchCV.
            param_distributions (dict): Dictionary with parameter names (`str`) as keys and distributions or lists of parameters to try.
            model_folder (str): Directory path to save and load CatBoost models.
        """
        self.data_prep = dataprep
        self.models = {}  # Dictionary to store best CatBoost models per timestep
        self.best_params = {}  # Dictionary to store best parameters per timestep
        self.iterations = iterations
        self.random_state = random_state
        self.depth = depth
        self.learning_rate = learning_rate
        self.l2_leaf_reg = l2_leaf_reg
        self.early_stopping_rounds = early_stopping_rounds
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
                'iterations': [500, 1000, 1500],
                'depth': [4, 6, 8, 10],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'l2_leaf_reg': [1, 3, 5, 7, 9],
                'border_count': [32, 64, 128],
                'bagging_temperature': [0, 1, 2],
                'random_strength': [1, 2, 3],
                # 'scale_pos_weight': [1, 2, 3],
                # 'bagging_fraction': [0.6, 0.8, 1.0],
                'feature_border_type': ['Median', 'Uniform', 'GreedyLogSum']
            }
            self.param_distributions = {
                'iterations': [500],
                'depth': [4],
                'learning_rate': [0.01],
                'l2_leaf_reg': [1, 9],
                'border_count': [32],
                'bagging_temperature': [ 2],
                'random_strength': [ 3],
                # 'scale_pos_weight': [1, 2, 3],
                # 'bagging_fraction': [0.6, 0.8, 1.0],
                'feature_border_type': [ 'GreedyLogSum']
            }
        else:
            self.param_distributions = param_distributions

    def train(self, X_train, y_train_dict, categorical_features=[]):
        """
        Trains a separate CatBoost model for each timestep with hyperparameter tuning and saves the models.
        
        Parameters:
            X_train (numpy.ndarray): Training features with shape (samples, timesteps, features).
            y_train_dict (dict): Dictionary with timestep as key and y_train array as value.
            categorical_features (list): List of categorical feature indices or names.
        """
        for t in range(X_train.shape[1]):
            print(f"\n--- Training CatBoost for Timestep {self.horizons[t]} ---")
            
            # Initialize the CatBoost Regressor
            catboost = CatBoostRegressor(
                random_seed=self.random_state,
                verbose=False,
                early_stopping_rounds=self.early_stopping_rounds
            )
            
            # Initialize RandomizedSearchCV
            random_search = RandomizedSearchCV(
                estimator=catboost,
                param_distributions=self.param_distributions,
                n_iter=self.n_iter,
                cv=self.cv,
                verbose=self.verbose,
                random_state=self.random_state,
                n_jobs=-1,
                scoring=self.scoring
            )
            
            # Prepare data for CatBoost
            # If categorical features are provided as indices, pass them; else, leave empty
            if isinstance(categorical_features, list) and categorical_features:
                # Assuming categorical_features are indices
                cat_features = categorical_features
            else:
                cat_features = []
            
            # Fit RandomizedSearchCV
            print("Performing hyperparameter search with early stopping...")
            try:
                random_search.fit(
                    X_train[:, t, :],
                    y_train_dict,
                    eval_set=(X_train[:, t, :], y_train_dict),
                    cat_features=cat_features,
                    verbose=False
                )
            except Exception as e:
                print(f"Error during training for Timestep {self.horizons[t]}: {e}")
                continue
            
            # Retrieve the best model and parameters
            best_catboost = random_search.best_estimator_
            self.best_params[t] = random_search.best_params_
            
            print(f"Best Parameters for Timestep {self.horizons[t]}: {random_search.best_params_}")
            
            # Evaluate on training data
            y_pred_scaled = best_catboost.predict(X_train[:, t, :])
            y_pred = self.data_prep.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
            y_train_unscaled = self.data_prep.scaler_y.inverse_transform(y_train_dict.reshape(-1, 1)).flatten()
            
            mse = mean_squared_error(y_train_unscaled, y_pred)
            mae = mean_absolute_error(y_train_unscaled, y_pred)
            print(f"Timestep {self.horizons[t]} - Training MSE: {mse:.4f}, Training MAE: {mae:.4f}")
            
            # Save the trained model to the specified folder
            model_filename = f"catboost_{self.horizons[t]}.pkl"
            model_path = os.path.join(self.model_folder, model_filename)
            with open(model_path, 'wb') as f:
                pickle.dump(best_catboost, f)
            print(f"Saved model for Timestep {self.horizons[t]} at {model_path}")

    def evaluate(self, X_test, y_test_dict, categorical_features=[]):
        """
        Loads each CatBoost model from the folder and evaluates it on the test set, printing MSE and MAE.
        
        Parameters:
            X_test (numpy.ndarray): Testing features with shape (samples, timesteps, features).
            y_test_dict (dict): Dictionary with timestep as key and y_test array as value.
            categorical_features (list): List of categorical feature indices or names.
        
        Returns:
            y_pred_dict (dict): Dictionary with timestep as key and y_pred array as value.
        """
        y_pred_dict = {}
        mae_per_timestep = {}
        
        for t in range(X_test.shape[1]):
            # print(f"\n--- Evaluating CatBoost for Timestep {self.horizons[t]} ---")
            model_filename = f"catboost_{self.horizons[t]}.pkl"
            model_path = os.path.join(self.model_folder, model_filename)
            
            if not os.path.exists(model_path):
                print(f"No saved model found for Timestep {self.horizons[t]} at {model_path}. Skipping evaluation.")
                continue
            
            # Load the model from the file
            try:
                with open(model_path, 'rb') as f:
                    catboost = pickle.load(f)
                print(f"Loaded model for Timestep {self.horizons[t]} from {model_path}")
            except Exception as e:
                print(f"Error loading model for Timestep {self.horizons[t]}: {e}")
                continue
            
            # Prepare data for CatBoost
            if isinstance(categorical_features, list) and categorical_features:
                cat_features = categorical_features
            else:
                cat_features = []
            
            # Predict
            try:
                y_pred_scaled = catboost.predict(X_test[:, t, :])
            except Exception as e:
                print(f"Error during prediction for Timestep {self.horizons[t]}: {e}")
                continue
            
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
        Plots feature importances for each timestep's CatBoost model.
        
        Parameters:
            feature_names (list): List of feature names.
            top_n (int): Number of top features to display.
        """
        for t in range(len(self.horizons)):
            model_filename = f"catboost_{self.horizons[t]}.pkl"
            model_path = os.path.join(self.model_folder, model_filename)
            
            if not os.path.exists(model_path):
                print(f"No saved model found for Timestep {self.horizons[t]} at {model_path}. Skipping plotting.")
                continue
            
            # Load the model from the file
            try:
                with open(model_path, 'rb') as f:
                    catboost = pickle.load(f)
                print(f"\n--- Plotting Feature Importances for Timestep {self.horizons[t]} ---")
            except Exception as e:
                print(f"Error loading model for Timestep {self.horizons[t]}: {e}")
                continue
            
            # Get feature importances
            importances = catboost.get_feature_importance(Pool(X_test[:, t, :], label=y_test_dict[t]))
            indices = np.argsort(importances)[::-1]
            
            plt.figure(figsize=(10, 6))
            plt.title(f"Feature Importances for Timestep {self.horizons[t]}")
            plt.bar(range(top_n), importances[indices][:top_n], color="g", align="center")
            plt.xticks(range(top_n), [feature_names[i] for i in indices[:top_n]], rotation=90)
            plt.xlim([-1, top_n])
            plt.ylabel('Importance Score')
            plt.tight_layout()
            plt.show()


class CATRollingForecaster:
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
            model_filename = f"catboost_{horizon}.pkl"
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