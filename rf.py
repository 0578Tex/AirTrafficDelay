import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV
  
class RandomForestTrainer:
    def __init__(self, dataprep=None, random_state=42,
                 horizons=[], cv=3, n_iter=50, scoring='neg_mean_absolute_error', 
                 verbose=1, random_search=True, param_distributions=None):

        self.data_prep = dataprep
        self.models = {}  # Dictionary to store best RF models per timestep
        self.best_params = {}  # Dictionary to store best parameters per timestep
        self.random_state = random_state
        self.horizons = horizons
        self.cv = cv
        self.n_iter = n_iter
        self.scoring = scoring
        self.verbose = verbose
        self.random_search = random_search
        self.model_folder = r"C:\Users\iLabs_6\Documents\Tex\rf"

        
        # Default hyperparameter distributions if none provided
        if param_distributions is None:
            self.param_distributions = {
                'n_estimators': [200],
                'max_depth': [20],
                'max_features':[0.7],
                'max_samples': [0.8],
                'bootstrap': [True],
                # 'max_features': ['auto','log2']
            }
        else:
            self.param_distributions = param_distributions


    def train(self, X_train, y_train_dict):
        """
        Trains a separate RF model for each timestep with hyperparameter tuning.
        
        Parameters:
            X_train (numpy.ndarray): Training features with shape (samples, timesteps, features).
            y_train_dict (dict): Dictionary with timestep as key and y_train array as value.
        """
        for t in range(X_train.shape[1]):
            print(f"\n--- Training Random Forest for Timestep {self.horizons[t]} ---")
            
            # Initialize the Random Forest Regressor
            rf = RandomForestRegressor(random_state=self.random_state)
            
            # Initialize RandomizedSearchCV
            random_search = RandomizedSearchCV(
                estimator=rf,
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
            best_rf = random_search.best_estimator_
            self.best_params[t] = random_search.best_params_
            
            print(f"Best Parameters for Timestep {self.horizons[t]}: {random_search.best_params_}")
            
            # Optionally, evaluate on training data (could use validation set instead)
            y_pred_scaled = best_rf.predict(X_train[:, t, :])
            y_pred = self.data_prep.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
            y_train_unscaled = self.data_prep.scaler_y.inverse_transform(y_train_dict.reshape(-1, 1)).flatten()
            
            mse = mean_squared_error(y_train_unscaled, y_pred)
            mae = mean_absolute_error(y_train_unscaled, y_pred)
            print(f"Timestep {self.horizons[t]} - Training MSE: {mse:.4f}, Training MAE: {mae:.4f}")

            model_filename = f"rf_{self.horizons[t]}.pkl"
            model_path = os.path.join(self.model_folder, model_filename)
            with open(model_path, 'wb') as f:
                pickle.dump(best_rf, f)
            print(f"Saved model for Timestep {self.horizons[t]} at {model_path}")


    def evaluate(self, X_test, y_test_dict):
        """
        Evaluates each RF model on the test set and prints MSE and MAE.
        
        Parameters:
            X_test (DataFrame): Testing features.
            y_test_dict (dict): Dictionary with timestep as key and y_test array as value.
        
        Returns:
            y_pred_dict (dict): Dictionary with timestep as key and y_pred array as value.
        """
        y_pred_dict = {}
        mae_per_timestep = {}
        for t in range(X_test.shape[1]):
            # print(f"Evaluating Random Forest for Timestep {t}...")
            model_filename = f"rf_{self.horizons[t]}.pkl"
            model_path = os.path.join(self.model_folder, model_filename)
            
            if not os.path.exists(model_path):
                print(f"No saved model found for Timestep {self.horizons[t]} at {model_path}. Skipping evaluation.")
                continue
            
            # Load the model from the file
            with open(model_path, 'rb') as f:
                rf = pickle.load(f)
            # print(f"Loaded model for Timestep {self.horizons[t]} from {model_path}")
            
            y_pred_scaled = rf.predict(X_test[:,t,:])
            y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
            y_test_unscaled = self.scaler_y.inverse_transform(y_test_dict.reshape(-1, 1)).flatten()
            
            mse = mean_squared_error(y_test_unscaled, y_pred)
            mae = mean_absolute_error(y_test_unscaled, y_pred)
            # print(f"Timestep {t} - Test MSE: {mse:.4f}, Test MAE: {mae:.4f}")
            
            y_pred_dict[t] = y_pred
            mae_per_timestep[t] = mae
        
        # Plot MAE per timestep
        sorted_timesteps = sorted(mae_per_timestep.keys())
        mae_values = [mae_per_timestep[t] for t in sorted_timesteps]
        
        plt.figure(figsize=(12, 6))
        plt.plot(range(-300,5,5 ), mae_values, marker='o', linestyle='-')
        plt.xlabel('Time Horizon')
        plt.ylabel('Mean Absolute Error (MAE)')
        plt.title('MAE for Each Time Horizon')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        return mae_values
        

    def plot_feature_importances(self, feature_names, top_n=20):
        """
        Plots feature importances for each timestep's RF model.
        
        Parameters:
            feature_names (list): List of feature names.
            top_n (int): Number of top features to display.
        """
        for t, rf in self.models.items():
            importances = rf.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            plt.figure(figsize=(10, 6))
            plt.title(f"Feature Importances for Timestep {t}")
            plt.bar(range(top_n), importances[indices][:top_n], color="r", align="center")
            plt.xticks(range(top_n), [feature_names[i] for i in indices[:top_n]], rotation=90)
            plt.xlim([-1, top_n])
            plt.tight_layout()
            plt.show()



class RFRollingForecaster:
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

    def load_models(self):
        """
        Load the LightGBM models for each timestep.
        Returns a dictionary with horizon as key and model as value.
        """
        models = {}
        for t_idx, horizon in enumerate(self.time_horizons):
            model_filename = f"rf_{horizon}.pkl"
            model_path = os.path.join(self.model_folder, model_filename)
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    models[horizon] = pickle.load(f)
                print(f"Loaded model for horizon {horizon} from {model_path}")
            else:
                print(f"Model file {model_path} not found. Skipping horizon {horizon}.")
        return models

    def rolling_forecast(self):
        """
        Perform rolling forecasts using the LightGBM models.
        """
        num_timesteps = self.scaled_features.shape[0]
        for t in range(num_timesteps):
            # Update current time index
            self.current_time_index = t

            # Get the features at the current timestep
            current_features = self.scaled_features[t, :]  # Shape: (features,)

            # Get the corresponding horizon
            if t < len(self.time_horizons):
                horizon = self.time_horizons[t]
            else:
                print(f"No corresponding horizon for timestep {t}. Stopping forecast.")
                break

            # Check if model exists for this horizon
            if horizon not in self.models:
                print(f"No model found for horizon {horizon}. Skipping prediction.")
                self.predictions.append(np.nan)
                continue

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

