import torch
import torch.nn as nn
import math
import os
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # Even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_length, d_model)
        Returns:
            Tensor with positional encodings added
        """
        x = x + self.pe[:, :x.size(1), :]
        return x

class TransformerRegressor(nn.Module):
    def __init__(self, input_size, d_model=128, nhead=8, num_encoder_layers=4, 
                 dim_feedforward=256, dropout=0.1, activation='relu'):
        super(TransformerRegressor, self).__init__()
        self.d_model = d_model
        
        # Input linear layer to match d_model
        self.input_linear = nn.Linear(input_size, d_model)
        
        # Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer Encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
        
        # Output linear layer
        self.output_linear = nn.Linear(d_model, 1)  # Assuming single target variable
    
    def forward(self, src):
        """
        Args:
            src: Tensor of shape (batch_size, seq_length, input_size)
        Returns:
            out: Tensor of shape (batch_size, 1)
        """
        src = self.input_linear(src)  # (batch_size, seq_length, d_model)
        src = self.pos_encoder(src)    # (batch_size, seq_length, d_model)
        src = src.permute(1, 0, 2)     # (seq_length, batch_size, d_model)
        memory = self.transformer_encoder(src)  # (seq_length, batch_size, d_model)
        memory = memory.permute(1, 0, 2)        # (batch_size, seq_length, d_model)
        out = memory.mean(dim=1)                # (batch_size, d_model)
        out = self.output_linear(out)           # (batch_size, 1)
        return out
    

class EarlyStopper:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')
        self.best_epoch = 0

    def early_stop(self, validation_loss, epoch):
        if validation_loss < self.min_validation_loss - self.min_delta:
            self.min_validation_loss = validation_loss
            self.counter = 0
            self.best_epoch = epoch
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print(f"Early stopping triggered at epoch {epoch}. Best epoch: {self.best_epoch} with validation loss: {self.min_validation_loss:.4f}")
                return True
        return False

class TransformerTrainer:
    def __init__(self, dataprep, horizons, device='cpu', model_folder='transformer_models',
                 d_model=128, nhead=8, num_encoder_layers=4, dim_feedforward=256, 
                 dropout=0.1, activation='relu', learning_rate=1e-3, 
                 batch_size=32, num_epochs=30, early_stopping_patience=10):

        self.dataprep = dataprep
        self.horizons = horizons
        self.device = torch.device(device)
        self.model_folder = model_folder
        os.makedirs(self.model_folder, exist_ok=True)
        
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = activation
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.early_stopping_patience = early_stopping_patience
        
        # Initialize models and optimizers dictionaries
        self.models = {}
        self.optimizers = {}
        self.criterion = nn.MSELoss()
    
    def train(self, X_train, y_train_dict, X_val=None, y_val_dict=None):
        """
        Trains a Transformer model for each timestep.
        
        Parameters:
            X_train (numpy.ndarray): Training features (samples, timesteps, features).
            y_train_dict (dict): Dictionary of training targets per timestep.
            X_val (numpy.ndarray, optional): Validation features.
            y_val_dict (dict, optional): Dictionary of validation targets per timestep.
        """
        for t, horizon in enumerate(self.horizons):
            print(f"\n--- Training Transformer for Timestep {horizon} ---")
            
            # Define the model
            input_size = X_train.shape[2]
            model = TransformerRegressor(
                input_size=input_size,
                d_model=self.d_model,
                nhead=self.nhead,
                num_encoder_layers=self.num_encoder_layers,
                dim_feedforward=self.dim_feedforward,
                dropout=self.dropout,
                activation=self.activation
            ).to(self.device)
            
            # Define optimizer
            optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
            
            # Prepare data
            X_train_tensor = torch.tensor(X_train[:, t, :], dtype=torch.float32).unsqueeze(1)
            y_train_tensor = torch.tensor(y_train_dict[:,0,0], dtype=torch.float32).unsqueeze(1)
            print(f'{X_train_tensor.shape=}   {y_train_tensor.shape=}')
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True) 
            
            if X_val is not None and y_val_dict is not None:
                X_val_tensor = torch.tensor(X_val[:, t, :], dtype=torch.float32).unsqueeze(1)
                y_val_tensor = torch.tensor(y_val_dict[:,0,0], dtype=torch.float32).unsqueeze(1)
                val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
                val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
            else:
                val_loader = None
            
            early_stopper = EarlyStopper(patience=self.early_stopping_patience)
            best_val_loss = float('inf')
            best_model_state = None
            
            for epoch in range(self.num_epochs):
                model.train()
                running_train_loss = 0.0
                
                for batch_X, batch_y in train_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = self.criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    
                    running_train_loss += loss.item()
                
                avg_train_loss = running_train_loss / len(train_loader)
                print(f"Epoch {epoch+1}/{self.num_epochs} - Training Loss: {avg_train_loss:.4f}")
                
                # Validation
                if val_loader:
                    model.eval()
                    running_val_loss = 0.0
                    with torch.no_grad():
                        for val_X, val_y in val_loader:
                            val_X = val_X.to(self.device)
                            val_y = val_y.to(self.device)
                            val_outputs = model(val_X)
                            val_loss = self.criterion(val_outputs, val_y)
                            running_val_loss += val_loss.item()
                    avg_val_loss = running_val_loss / len(val_loader)
                    print(f"Epoch {epoch+1}/{self.num_epochs} - Validation Loss: {avg_val_loss:.4f}")
                    
                    # Early Stopping
                    if early_stopper.early_stop(avg_val_loss, epoch):
                        print("Early stopping triggered.")
                        break
                    
                    # Save the best model
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        best_model_state = model.state_dict()
                else:
                    # If no validation, consider training loss for early stopping
                    if avg_train_loss < best_val_loss:
                        best_val_loss = avg_train_loss
                        best_model_state = model.state_dict()
            
            # Load the best model state
            if best_model_state:
                model.load_state_dict(best_model_state)
            
            # Save the model
            model_filename = f"transformer_{horizon}.pt"
            model_path = os.path.join(self.model_folder, model_filename)
            torch.save(model.state_dict(), model_path)
            print(f"Saved Transformer model for Timestep {horizon} at {model_path}")
            
            # Store the model
            self.models[horizon] = model
    
    def evaluate(self, X_test, y_test_dict):
        """
        Evaluates the Transformer models on the test set.
        
        Parameters:
            X_test (numpy.ndarray): Testing features (samples, timesteps, features).
            y_test_dict (dict): Dictionary of testing targets per timestep.
        
        Returns:
            y_pred_dict (dict): Dictionary of predictions per timestep.
        """
        y_pred_dict = {}
        mae_per_timestep = {}
        mse_per_timestep = {}
        
        for t, horizon in enumerate(self.horizons):
            print(f"\n--- Evaluating Transformer for Timestep {horizon} ---")
            model_filename = f"transformer_{horizon}.pt"
            model_path = os.path.join(self.model_folder, model_filename)
            
            if not os.path.exists(model_path):
                print(f"No saved model found for Timestep {horizon} at {model_path}. Skipping evaluation.")
                continue
            
            # Load the model
            model = TransformerRegressor(
                input_size=X_test.shape[2],
                d_model=self.d_model,
                nhead=self.nhead,
                num_encoder_layers=self.num_encoder_layers,
                dim_feedforward=self.dim_feedforward,
                dropout=self.dropout,
                activation=self.activation
            ).to(self.device)
            model.load_state_dict(torch.load(model_path))
            model.eval()
            # X_train_tensor = torch.tensor(X_train[:, t, :], dtype=torch.float32).unsqueeze(1)
            # y_train_tensor = torch.tensor(y_train_dict[:,0,0], dtype=torch.float32).unsqueeze(1)
            # Prepare test data
            X_test_tensor = torch.tensor(X_test[:, t, :], dtype=torch.float32).to(self.device).unsqueeze(1)
            y_test_tensor = torch.tensor(y_test_dict[:,0,0], dtype=torch.float32).unsqueeze(1).to(self.device)
            
            with torch.no_grad():
                predictions = model(X_test_tensor).cpu().numpy().flatten()
            
            # Inverse transform predictions and targets
            y_pred = self.dataprep.scaler_y.inverse_transform(predictions.reshape(-1, 1)).flatten()
            y_true = self.dataprep.scaler_y.inverse_transform(y_test_dict[:,0,0].reshape(1, -1)).flatten()
            print(f'{y_pred.shape=}')
            print(f'{y_true.shape=}')
            # Calculate metrics
            mse = mean_squared_error(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            print(f"Timestep {horizon} - Test MSE: {mse:.4f}, Test MAE: {mae:.4f}")
            
            y_pred_dict[horizon] = y_pred
            mae_per_timestep[horizon] = mae
            mse_per_timestep[horizon] = mse
        
        # Plot MAE across all timesteps
        if mae_per_timestep:
            print(f'{mae_per_timestep=}')
            # sorted_horizons = sorted(mae_per_timestep.keys())
            mae_values = [mae_per_timestep[h] for h in mae_per_timestep.keys()]
            # print(f'{sorted_horizons=}')
            print(f'{mae_values=}')
            plt.figure(figsize=(12, 6))
            plt.plot( mae_per_timestep.keys(), mae_values, marker='o', linestyle='-')
            plt.xlabel('Time Horizon (minutes)', fontsize=14)
            plt.ylabel('Mean Absolute Error (MAE)', fontsize=14)
            plt.title('MAE for Each Time Horizon - Transformer', fontsize=16)
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            plt.show()
        else:
            print("No timesteps were evaluated.")
        
        return y_pred_dict
