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
from captum.attr import IntegratedGradients
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch.optim.lr_scheduler as lr_scheduler  # Add this import at the top


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, max_relative_position=20):
        super(MultiHeadAttention, self).__init__()
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.max_relative_position = max_relative_position

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, hidden_size)

        # Relative position embeddings
        vocab_size = 2 * max_relative_position + 1
        self.relative_position_embeddings = nn.Embedding(vocab_size, self.head_dim)

    def forward(self, query, key, value, mask=None):
        N, seq_length, hidden_size = query.shape

        # Linear projections
        Q = self.query(query)
        K = self.key(key)
        V = self.value(value)

        # Split into multiple heads
        Q = Q.view(N, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(N, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(N, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled Dot-Product Attention
        energy_content = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (N, num_heads, seq_length, seq_length)

        # Compute relative positions
        positions = torch.arange(seq_length, device=query.device).unsqueeze(0)
        relative_positions = positions.transpose(0, 1) - positions  # Shape: (seq_length, seq_length)

        # Clamp the relative positions to the maximum range
        relative_positions_clamped = torch.clamp(relative_positions, -self.max_relative_position, self.max_relative_position) + self.max_relative_position

        # Get the relative position embeddings
        relative_position_embeddings = self.relative_position_embeddings(relative_positions_clamped)  # Shape: (seq_length, seq_length, head_dim)

        # Adjust dimensions for batch and heads
        relative_position_embeddings = relative_position_embeddings.unsqueeze(0).repeat(N * self.num_heads, 1, 1, 1)
        Q_reshaped = Q.contiguous().view(N * self.num_heads, seq_length, self.head_dim)
        energy_positional = torch.einsum('bqd,bqkd->bqk', Q_reshaped, relative_position_embeddings)

        energy_positional = energy_positional.view(N, self.num_heads, seq_length, seq_length)

        # Combine content-based and positional energies
        energy = (energy_content + energy_positional) / (self.head_dim ** 0.5)

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            energy = energy.masked_fill(mask == 0, float('-inf'))

        # Causal Masking: Prevent attention to future positions
        causal_mask = torch.tril(torch.ones((seq_length, seq_length), device=query.device)).bool()
        energy = energy.masked_fill(causal_mask == 0, float('-inf'))

        attention = torch.softmax(energy, dim=-1)
        attention = torch.matmul(attention, V)

        # Concatenate heads
        attention = attention.transpose(1, 2).contiguous().view(N, seq_length, hidden_size)

        out = self.fc_out(attention)
        return out, attention

class LSTMModelAttention(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size, num_layers, dropout=0.3, num_heads=4, max_relative_position=20):
        super(LSTMModelAttention, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers, batch_first=True, dropout=dropout)
        self.multihead_attention = MultiHeadAttention(hidden_layer_size, num_heads, max_relative_position=max_relative_position)
        self.fc = nn.Linear(hidden_layer_size, output_size)
        
    def forward(self, x, seq_lengths, h_n=None, c_n=None):
        # Pack the sequence
        try:
            x_packed = nn.utils.rnn.pack_padded_sequence(x, seq_lengths.cpu(), batch_first=True, enforce_sorted=False)
        except:
            x_packed = nn.utils.rnn.pack_padded_sequence(x, seq_lengths, batch_first=True, enforce_sorted=False)

        # LSTM forward pass
        if h_n is not None and c_n is not None:
            packed_output, (h_n, c_n) = self.lstm(x_packed, (h_n, c_n))
        else:
            packed_output, (h_n, c_n) = self.lstm(x_packed)

        # Unpack the sequence
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        # Create mask
        max_seq_len = lstm_out.size(1)
        seq_lengths = torch.tensor(seq_lengths, device='cuda') if (isinstance(seq_lengths, np.ndarray) or isinstance(seq_lengths, list)) else seq_lengths
        seq_lengths_expanded = seq_lengths.unsqueeze(1)  # Shape: (batch_size, 1)
        mask = torch.arange(max_seq_len, device='cuda').unsqueeze(0) < seq_lengths_expanded
        # Apply Multi-Head Attention with Padding Mask
        attn_out, attention_weights = self.multihead_attention(lstm_out, lstm_out, lstm_out, mask=mask)

        # Pass through fully connected layer
        out = self.fc(attn_out)  # Shape: (batch_size, seq_len, output_size)

        return out, attention_weights

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

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size, num_layers, dropout=0.5):
        super(LSTMModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size, hidden_layer_size, num_layers, batch_first=True, dropout=dropout
        )
        self.fc = nn.Linear(hidden_layer_size, output_size)

    def forward(self, x, seq_lengths, h_n=None, c_n=None):
        # Pack the sequence
        try:
            x_packed = nn.utils.rnn.pack_padded_sequence(
                x, seq_lengths.cpu(), batch_first=True, enforce_sorted=False
            )
        except:
            x_packed = nn.utils.rnn.pack_padded_sequence(
                x, seq_lengths, batch_first=True, enforce_sorted=False
            )
        # LSTM forward pass
        if h_n is not None and c_n is not None:
            packed_output, (h_n, c_n) = self.lstm(x_packed, (h_n, c_n))
        else:
            packed_output, (h_n, c_n) = self.lstm(x_packed)

        # Unpack the sequence
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True, total_length=x.size(1)
        )  # Shape: (batch_size, seq_len, hidden_size)

        # Pass through fully connected layer
        out = self.fc(lstm_out)  # Shape: (batch_size, seq_len, output_size)

        return out, None  # Return None for attention weights


class LSTMModelSimpleAttention(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size, num_layers, dropout=0.5):
        super(LSTMModelSimpleAttention, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size, hidden_layer_size, num_layers, batch_first=True, dropout=dropout
        )
        self.attention = SimpleAttention(hidden_layer_size)
        self.fc = nn.Linear(hidden_layer_size, output_size)

    def forward(self, x, seq_lengths, h_n=None, c_n=None):
        # Pack the sequence
        try:
            x_packed = nn.utils.rnn.pack_padded_sequence(
                x, seq_lengths.cpu(), batch_first=True, enforce_sorted=False
            )
        except:
           x_packed = nn.utils.rnn.pack_padded_sequence(
                x, seq_lengths, batch_first=True, enforce_sorted=False
            )
        # LSTM forward pass
        if h_n is not None and c_n is not None:
            packed_output, (h_n, c_n) = self.lstm(x_packed, (h_n, c_n))
        else:
            packed_output, (h_n, c_n) = self.lstm(x_packed)

        # Unpack the sequence
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True, total_length=x.size(1)
        )  # Shape: (batch_size, seq_len, hidden_size)

        # Create mask
        max_seq_len = lstm_out.size(1)
        seq_lengths = torch.tensor(seq_lengths, device='cuda') if (isinstance(seq_lengths, np.ndarray) or isinstance(seq_lengths, list)) else seq_lengths
        seq_lengths_expanded = seq_lengths.unsqueeze(1)  # Shape: (batch_size, 1)
        mask = torch.arange(max_seq_len, device='cuda').unsqueeze(0) < seq_lengths_expanded
        # Apply attention
        context_vector, attention_weights = self.attention(lstm_out, mask)

        # Pass through fully connected layer
        out = self.fc(context_vector)  # Shape: (batch_size, output_size)

        # Expand the output to match sequence length
        out = out.unsqueeze(1).repeat(1, max_seq_len, 1)  # Shape: (batch_size, seq_len, output_size)

        return out, attention_weights

class SimpleAttention(nn.Module):
    def __init__(self, hidden_size):
        super(SimpleAttention, self).__init__()
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, lstm_out, mask=None):
        # lstm_out: (batch_size, seq_len, hidden_size)
        # Compute attention scores
        attention_scores = self.attention(lstm_out).squeeze(-1)  # (batch_size, seq_len)

        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))

        # Compute attention weights
        attention_weights = torch.softmax(attention_scores, dim=1)  # (batch_size, seq_len)

        # Compute context vector
        context_vector = torch.bmm(
            attention_weights.unsqueeze(1), lstm_out
        )  # (batch_size, 1, hidden_size)
        context_vector = context_vector.squeeze(1)  # (batch_size, hidden_size)

        return context_vector, attention_weights


class LSTMModelTrainerAttention:
    def __init__(self, input_size, data_prep, model_type='varattention', log_dir='runs3/lstm_attention'):
        self.input_size = input_size
        self.model_type = model_type
        self.data_prep = data_prep
        self.train_loader = None
        self.test_loader = None
        self.time_steps = None
        self.writer = SummaryWriter(log_dir=log_dir)
        self.final_attention_weights = None  # To store final attention weights

    def set_data_loaders(self, train_loader, test_loader, time_steps):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.time_steps = time_steps

    @staticmethod
    def custom_loss(outputs, targets, seq_lengths, lambda_mse=0.5, lambda_mae=0.3, lambda_accum=0.2):
        """
        Enhanced custom loss function combining MSE, MAE, and accumulated error with dynamic weighting.

        :param outputs: Predicted values, shape [batch_size, seq_len, 1]
        :param targets: True values, shape [batch_size, seq_len, 1]
        :param seq_lengths: Actual sequence lengths before padding, list or tensor
        :param lambda_mse: Weight for Mean Squared Error component
        :param lambda_mae: Weight for Mean Absolute Error component
        :param lambda_accum: Weight for Accumulated Error component
        :return: Combined loss scalar
        """
        batch_size, seq_len, _ = outputs.size()
        
        # Ensure seq_lengths is a PyTorch Tensor on the same device as outputs
        if not torch.is_tensor(seq_lengths):
            seq_lengths = torch.tensor(seq_lengths, dtype=torch.long, device=outputs.device)
        
        # Create a mask based on sequence lengths
        mask = torch.arange(seq_len, device=outputs.device).unsqueeze(0) < seq_lengths.unsqueeze(1)
        mask = mask.float()
        
        # Handle NaN values in targets
        targets = torch.nan_to_num(targets, nan=0.0)
        
        # Squeeze the last dimension if necessary
        outputs = outputs.squeeze(-1)
        targets = targets.squeeze(-1)
        
        # Apply mask
        outputs_masked = outputs * mask
        targets_masked = targets * mask
        
        # Compute MSE
        mse_loss = torch.sum((outputs_masked - targets_masked) ** 2) / (torch.sum(mask) + 1e-10)
        
        # Compute MAE
        mae_loss = torch.sum(torch.abs(outputs_masked - targets_masked)) / (torch.sum(mask) + 1e-10)
        
        # Compute accumulated error (for error propagation)
        cumulative_errors = torch.cumsum(outputs_masked - targets_masked, dim=1)
        accumulated_error_loss = torch.sum((cumulative_errors) ** 2 * mask) / (torch.sum(mask) + 1e-10)
        
        # Total loss with dynamic weighting
        loss = lambda_mse * mse_loss + lambda_mae * mae_loss + lambda_accum * accumulated_error_loss
        
        return loss
        

    
    def train_and_evaluate_model(self, params, device, l1_penalty=0.0, l2_penalty=0.0, initial_shift=2, shift_increment=1, shift_interval=10, final_shift=6, noise_std=0.0, dropout=0.3):

        input_size, hidden_layer_size, output_size, num_layers, learning_rate, num_epochs, _, _, dropout = params

        # Initialize the model based on model_type
        if self.model_type == 'varattention':
            model = LSTMModelAttention(
                input_size, hidden_layer_size, output_size, num_layers, num_heads=5, dropout=dropout  # Adjust num_heads as needed
            ).to(device)
        elif self.model_type == 'simpleattention':
            model = LSTMModelSimpleAttention(
                input_size, hidden_layer_size, output_size, num_layers, dropout=dropout
            ).to(device)
        elif self.model_type == 'lstm':
            model = LSTMModel(
                input_size, hidden_layer_size, output_size, num_layers, dropout=dropout
            ).to(device)
        else:
            raise ValueError("Invalid model_type. Choose from 'lstm', 'simpleattention', or 'varattention'.")
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total trainable parameters: {total_params}")
        print(f'Start training with model {self.model_type}')

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_penalty)
        criterion = nn.MSELoss()
        early_stopper = EarlyStopper(patience=3, min_delta=0.001)

        best_model_state = None
        best_val_loss = float('inf')
        best_epoch = 0

        train_losses = []
        val_losses = []

        for epoch in range(num_epochs):
            model.train()
            running_train_loss = 0.0

            # Determine current max_shift based on epoch
            current_max_shift =4
            print(f"Epoch {epoch+1}: Current max_shift = {current_max_shift}")

            # Training loop
            for X_batch, y_batch in self.train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                # Apply sequence shifting with current_max_shift
                X_batch_shifted, y_batch_shifted = self.shift_sequences(X_batch, y_batch, current_max_shift)

                # Compute the sequence lengths for the batch
                seq_lengths = (X_batch_shifted.sum(dim=2) != 0).sum(dim=1)

                # Forward pass
                y_pred, _ = model(X_batch_shifted, seq_lengths)

                # Compute loss
                loss = criterion(y_pred, y_batch_shifted)
                optimizer.zero_grad()
                loss.backward()

                # Apply gradient clipping (optional)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                running_train_loss += loss.item()

            avg_train_loss = running_train_loss / len(self.train_loader)
            train_losses.append(avg_train_loss)

            # Log training loss to TensorBoard
            self.writer.add_scalar('Loss/Train', avg_train_loss, epoch)

            # Validation loop
            model.eval()
            running_val_loss = 0.0
            with torch.no_grad():
                for X_batch_val, y_batch_val in self.test_loader:
                    X_batch_val, y_batch_val = X_batch_val.to(device), y_batch_val.to(device)
                    X_batch_val_shifted, y_batch_val_shifted = self.shift_sequences(X_batch_val, y_batch_val, current_max_shift)

                    # Compute the sequence lengths for the batch
                    seq_lengths_val = (X_batch_val_shifted.sum(dim=2) != 0).sum(dim=1)

                    # Forward pass
                    y_val_pred, _ = model(X_batch_val_shifted, seq_lengths_val)

                    # Compute validation loss
                    val_loss = criterion(y_val_pred, y_batch_val_shifted)
                    running_val_loss += val_loss.item()

            avg_val_loss = running_val_loss / len(self.test_loader)
            val_losses.append(avg_val_loss)

            # Log validation loss to TensorBoard
            self.writer.add_scalar('Loss/Validation', avg_val_loss, epoch)

            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

            # Early stopping check
            if early_stopper.early_stop(avg_val_loss, epoch):
                print("Early stopping triggered.")
                break

            # Save the best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = model.state_dict()
                best_epoch = epoch

        if best_model_state is not None:
            print(f"Restoring the best model from epoch {best_epoch+1} with validation loss: {best_val_loss:.4f}")
            model.load_state_dict(best_model_state)

        # Close the TensorBoard writer
        self.writer.close()
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss', linestyle='--')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.show()

        return model, avg_train_loss, best_val_loss


    def shift_sequences(self, X_batch, y_batch, max_shift):

        batch_size, seq_len, feature_dim = X_batch.size()
        shifted_X_batch = torch.zeros_like(X_batch)
        shifted_y_batch = torch.zeros_like(y_batch)

        for i in range(batch_size):
            # Define possible shifts

            shifts = list(range(-max_shift, max_shift + 1))
            
            # Define weights: higher weights for smaller shifts
            # Example: shift=0 has weight=max_shift +1, shift=±1 has weight=max_shift, ..., shift=±max_shift has weight=1
            weights = [max_shift + 1 - abs(shift) for shift in shifts]
            total_weight = sum(weights)
            probabilities = [w / total_weight for w in weights]

            # Sample a shift based on the defined probabilities
            sampled_shift = np.random.choice(shifts, p=probabilities)

            if sampled_shift > 0:
                # Shift forward, pad at the beginning by repeating the first valid feature
                shifted_X_batch[i, sampled_shift:] = X_batch[i, :-sampled_shift]
                shifted_y_batch[i, sampled_shift:] = y_batch[i, :-sampled_shift]

                # Repeat the first valid feature for padding
                shifted_X_batch[i, :sampled_shift] = X_batch[i, 0].unsqueeze(0)
                shifted_y_batch[i, :sampled_shift] = y_batch[i, 0].unsqueeze(0)

            elif sampled_shift < 0:
                # Shift backward, pad at the end by repeating the last valid feature
                abs_shift = abs(sampled_shift)
                shifted_X_batch[i, :-abs_shift] = X_batch[i, abs_shift:]
                shifted_y_batch[i, :-abs_shift] = y_batch[i, abs_shift:]

                # Repeat the last valid feature for padding
                shifted_X_batch[i, -abs_shift:] = X_batch[i, -1].unsqueeze(0)
                shifted_y_batch[i, -abs_shift:] = y_batch[i, -1].unsqueeze(0)

            else:
                # No shift
                shifted_X_batch[i] = X_batch[i]
                shifted_y_batch[i] = y_batch[i]

        return shifted_X_batch, shifted_y_batch


    def hyperparameter_search(self, max_shift, output_size=1):
        hidden_layer_sizes =[110]
        num_layers_list = [5]
        learning_rates = [0.0005]
        num_epochs_list = [50]
        dropout=[0.1]
        l1_penalties = [0]
        l2_penalties = [0]
        max_shift=[5]
        num_heads = [4]

        hyperparameter_combinations = list(itertools.product(
            [self.input_size], hidden_layer_sizes, [output_size], num_layers_list, learning_rates, num_epochs_list, max_shift, num_heads, dropout
        ))

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'device = {device}')
        best_val_loss = float('inf')
        best_params = None
        best_model = None

        for params in hyperparameter_combinations:
            print(f'params = {params}')
            # try:
            model, train_loss, val_loss = self.train_and_evaluate_model(params, device)

            print(f"Params: {params}, Val Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_params = params
                best_model = model
            # except Exception as e:
            #     print(f"Params: {params}, gave error {e}")

                # continue

        print(f"Best Params: {best_params}, Best Val Loss: {best_val_loss:.4f}")
        return best_model

    def evaluate_and_plot(self, model, X_test_tensor, y_test_tensor, scaler_y, horizons):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        X_test_tensor = X_test_tensor.to(device)
        y_test_tensor = y_test_tensor.to(device)
        model = model.to(device)
        model.eval()

        with torch.no_grad():
            try:
                seq_lengths_val = (X_test_tensor.sum(dim=2) != 0).sum(dim=1).cpu().numpy()
            except:
                seq_lengths_val = torch.tensor(
                    (X_test_tensor.sum(dim=2) != 0).sum(dim=1), device=X_test_tensor.device
                )
            y_test_pred, _ = model(X_test_tensor, seq_lengths_val)  # Shape: (batch_size, time_steps, output_size)

        # print(f'{self.time_steps}')
        self.time_steps = 61  # Ensure this matches the actual time_steps

        # Inverse transform the scaled predictions and true values for comparison
        y_test_pred_inverse = scaler_y.inverse_transform(
            y_test_pred.detach().cpu().numpy().reshape(-1, 1)
        ).reshape(-1, self.time_steps)

        y_test_inverse = scaler_y.inverse_transform(
            y_test_tensor.detach().cpu().numpy().reshape(-1, 1)
        ).reshape(-1, self.time_steps)

        # Compute absolute error per timestep
        absolute_error_per_timestep = np.abs(y_test_inverse - y_test_pred_inverse)

        # Compute MAE per timestep
        mae_per_timestep = np.mean(absolute_error_per_timestep, axis=0)
        # print(f'{mae_per_timestep=}')
        
        # Plot MAE per timestep
        plt.figure(figsize=(10, 6))
        plt.plot(horizons, mae_per_timestep, marker='o')
        plt.xlabel('Timestep')
        plt.xticks(horizons, rotation='vertical')
        plt.ylabel('Mean Absolute Error (MAE)')
        plt.title('MAE for Each Timestep')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Compute error per timestep
        error_per_timestep = y_test_inverse - y_test_pred_inverse

        # Compute mean error per timestep
        mean_error_per_timestep = np.mean(error_per_timestep, axis=0)
        # print(f'{mean_error_per_timestep=}')

        # Compute standard deviation of error per timestep
        std_error_per_timestep = np.std(error_per_timestep, axis=0)
        # print(f'{std_error_per_timestep=}')

        # Plot mean error per timestep with standard deviation as fill
        plt.figure(figsize=(10, 6))
        plt.plot(horizons, mean_error_per_timestep, marker='o', label='Mean Error')
        plt.fill_between(
            horizons,
            mean_error_per_timestep - std_error_per_timestep,
            mean_error_per_timestep + std_error_per_timestep,
            color='b',
            alpha=0.2,
            label='±1 Std Dev'
        )
        plt.xlabel('Timestep')
        plt.xticks(horizons, rotation='vertical')
        plt.ylabel('Mean Error')
        plt.title('Mean Error for Each Timestep with Standard Deviation')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        return y_test_pred_inverse, y_test_inverse

    def plotCBAS(self, y_pred, y_test, cbaslabels, horizons):
        error_dict = {}

        for t, time in enumerate(horizons):
            y_p = y_pred[:, t]
            y_t = y_test[:, t]
            ttcbas = cbaslabels[:, t]

            for i in range(len(y_t)):
                time_to_cbas = ttcbas[i].astype('timedelta64[s]').astype(int) / 60
                error = abs(y_p[i] - y_t[i])
                if time_to_cbas not in error_dict:
                    error_dict[time_to_cbas] = []
                error_dict[time_to_cbas].append(error)


        bucketed_errors = defaultdict(list)
        for time_in_seconds, errors in error_dict.items():
            time_in_minutes = time_in_seconds   # Convert seconds to minutes
            bucket = int(time_in_minutes // 5) * 5   # Group into 5-minute buckets
            bucketed_errors[bucket].extend(errors)   # Aggregate errors within the bucket

        # Compute the mean error for each bucket
        mean_errors_by_bucket = {bucket: np.mean(errors) for bucket, errors in bucketed_errors.items()}

        # Sort the buckets
        sorted_buckets = sorted(mean_errors_by_bucket.items())

        # Extract sorted times and mean errors for plotting
        times = [item[0] for item in sorted_buckets]
        errors = [item[1] for item in sorted_buckets]

        # Plot the results
        plt.figure(figsize=(10, 6))
        plt.plot(times[:-10], errors[:-10], marker='o', linestyle='-')
        plt.title("Mean Prediction Error vs Time to CBAS Entry (bucketed in 5-minute intervals)")
        plt.xlabel("Time to CBAS Entry (minutes)")
        plt.ylabel("Mean Prediction Error (minutes)")
        plt.grid(True)
        plt.show()

    def plot_error_evolution(self, model, X_single_sample, y_single_sample, scaler_y, horizons):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        X_single_sample = X_single_sample.to(device)
        y_single_sample = y_single_sample.to(device)
        model = model.to(device)
        model.eval()

        with torch.no_grad():
            # Compute the sequence length for the single sample
            seq_lengths_single = (X_single_sample.sum(dim=2) != 0).sum(dim=1).cpu().numpy()

            # Get the model prediction for the single sample
            y_single_pred, _ = model(X_single_sample, seq_lengths_single)

        # Inverse transform the scaled predictions and true values for comparison
        y_single_pred_inverse = scaler_y.inverse_transform(
            y_single_pred.detach().cpu().numpy().reshape(-1, 1)
        ).reshape(-1, len(horizons))

        y_single_true_inverse = scaler_y.inverse_transform(
            y_single_sample.detach().cpu().numpy().reshape(-1, 1)
        ).reshape(-1, len(horizons))
        print(f'{y_single_pred_inverse=}')
        # Calculate the absolute error at each timestep
        absolute_error = np.abs(y_single_true_inverse - y_single_pred_inverse)
        # print(f'{y_single_true_inverse=}')
        # Plot the error evolution over time (horizons)
        plt.figure(figsize=(10, 6))
        plt.plot(horizons, absolute_error.flatten(), marker='o', linestyle='-')
        plt.xlabel('Timestep')
        plt.xticks(horizons[::10], rotation='vertical')
        plt.ylabel('Absolute Error')
        plt.title('Error Evolution for Single Sample Over Time')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        return y_single_pred_inverse, y_single_true_inverse, absolute_error
    

    def compute_baseline_mae(self, model, X_test_tensor, y_test_tensor, scaler_y, horizons):
        """
        Compute the baseline MAE on the original test set.

        :param model: Trained LSTM model
        :param X_test_tensor: Test features tensor, shape [N, T, F]
        :param y_test_tensor: Test targets tensor, shape [N, T, 1]
        :param scaler_y: Scaler used for target variable
        :param horizons: List of time horizons
        :return: Baseline MAE per timestep
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()

        X_test = X_test_tensor.to(device)
        y_test = y_test_tensor.to(device)

        with torch.no_grad():
            seq_lengths_val = (X_test.sum(dim=2) != 0).sum(dim=1)
            y_pred, _ = model(X_test, seq_lengths_val)  # Shape: (batch_size, time_steps, output_size)

        # Inverse transform the predictions and true values
        y_pred_inverse = scaler_y.inverse_transform(
            y_pred.detach().cpu().numpy().reshape(-1, 1)
        ).reshape(-1, len(horizons))

        y_test_inverse = scaler_y.inverse_transform(
            y_test.detach().cpu().numpy().reshape(-1, 1)
        ).reshape(-1, len(horizons))

        # Compute MAE per timestep
        absolute_error_per_timestep = np.abs(y_test_inverse - y_pred_inverse)
        baseline_mae = np.mean(absolute_error_per_timestep, axis=0)  # Shape: (T,)

        return baseline_mae
        
    def print_and_plot_performance(self, model, X_test_tensor, y_test_tensor, scaler_y, horizons):
        """
        Computes and prints performance metrics (MAE, RMSE, R²) and generates comprehensive plots
        to visualize the model's performance, including error and standard deviation per timestep.

        :param model: Trained LSTM model
        :param X_test_tensor: Test features tensor, shape [N, T, F]
        :param y_test_tensor: Test targets tensor, shape [N, T, 1]
        :param scaler_y: Scaler used for target variable
        :param horizons: List of time horizons
        """
        # Set Seaborn style for publication-quality plots
        sns.set(style="whitegrid", context="talk", palette="deep")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()

        # Move data to device
        X_test = X_test_tensor.to(device)
        y_test = y_test_tensor.to(device)

        with torch.no_grad():
            # Compute sequence lengths
            seq_lengths_val = (X_test.sum(dim=2) != 0).sum(dim=1)

            # Get model predictions
            y_pred, _ = model(X_test, seq_lengths_val)  # Shape: (batch_size, time_steps, output_size)

        # Inverse transform predictions and true values
        y_pred_np = y_pred.detach().cpu().numpy().reshape(-1, 1)
        y_pred_inverse = scaler_y.inverse_transform(y_pred_np).reshape(-1, len(horizons))

        y_test_np = y_test.detach().cpu().numpy().reshape(-1, 1)
        y_test_inverse = scaler_y.inverse_transform(y_test_np).reshape(-1, len(horizons))

        # Compute overall performance metrics
        mae = mean_absolute_error(y_test_inverse.flatten(), y_pred_inverse.flatten())
        rmse = np.sqrt(mean_squared_error(y_test_inverse.flatten(), y_pred_inverse.flatten()))
        r2 = r2_score(y_test_inverse.flatten(), y_pred_inverse.flatten())

        print("==== Overall Performance Metrics ====")
        print(f"Mean Absolute Error (MAE): {mae:.4f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        print(f"R² Score: {r2:.4f}\n")

        # Compute metrics per timestep
        mae_per_timestep = mean_absolute_error(y_test_inverse, y_pred_inverse, multioutput='raw_values')
        rmse_per_timestep = np.sqrt(mean_squared_error(y_test_inverse, y_pred_inverse, multioutput='raw_values'))
        r2_per_timestep = r2_score(y_test_inverse, y_pred_inverse, multioutput='raw_values')

        # Compute Error and Standard Deviation per timestep
        error_per_timestep = y_pred_inverse - y_test_inverse  # Shape: (samples, timesteps)
        mean_error_per_timestep = np.mean(error_per_timestep, axis=0)
        std_error_per_timestep = np.std(error_per_timestep, axis=0)

        # Create a DataFrame for per-timestep metrics
        metrics_df = pd.DataFrame({
            'Timestep': horizons,
            'MAE': mae_per_timestep,
            'RMSE': rmse_per_timestep,
            'R2_Score': r2_per_timestep,
            'Mean_Error': mean_error_per_timestep,
            'Std_Error': std_error_per_timestep
        })

        print("==== Per-Timestep Performance Metrics ====")
        print(metrics_df)
        print("\n")

        # Plot MAE and RMSE per timestep
        plt.figure(figsize=(14, 6))
        sns.lineplot(data=metrics_df, x='Timestep', y='MAE', marker='o', label='MAE')
        sns.lineplot(data=metrics_df, x='Timestep', y='RMSE', marker='s', label='RMSE')
        plt.xlabel('Timestep')
        plt.ylabel('Error')
        plt.title('MAE and RMSE per Timestep')
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Plot R² Score per timestep
        plt.figure(figsize=(14, 6))
        sns.lineplot(data=metrics_df, x='Timestep', y='R2_Score', marker='^', color='green', label='R² Score')
        plt.xlabel('Timestep')
        plt.ylabel('R² Score')
        plt.title('R² Score per Timestep')
        plt.legend()
        plt.tight_layout()
        plt.show()

        # # Plot Actual vs Predicted with Density (Hexbin)
        # plt.figure(figsize=(8, 8))
        # sns.kdeplot(x=y_test_inverse.flatten(), y=y_pred_inverse.flatten(), cmap="Blues", fill=True, thresh=0.05, levels=100)
        # plt.plot([y_test_inverse.min(), y_test_inverse.max()],
        #         [y_test_inverse.min(), y_test_inverse.max()],
        #         'r--', lw=2, label='Ideal Fit')
        # plt.xlabel('Actual Values')
        # plt.ylabel('Predicted Values')
        # plt.title('Actual vs Predicted Values with Density')
        # plt.legend()
        # plt.tight_layout()
        # plt.show()

        # Residuals
        residuals = y_test_inverse.flatten() - y_pred_inverse.flatten()

        # Plot Residuals Distribution
        plt.figure(figsize=(14, 6))

        # Histogram with KDE
        plt.subplot(1, 2, 1)
        sns.histplot(residuals, bins=50, kde=True, color='blue')
        plt.xlabel('Residuals')
        plt.ylabel('Frequency')
        plt.title('Residuals Distribution')

        # KDE Plot
        plt.subplot(1, 2, 2)
        sns.kdeplot(residuals, shade=True, color='red')
        plt.xlabel('Residuals')
        plt.ylabel('Density')
        plt.title('Residuals Density Plot')

        plt.tight_layout()
        plt.show()

        # Boxplot of Residuals
        plt.figure(figsize=(8, 6))
        sns.boxplot(x=residuals, color='lightgreen')
        plt.xlabel('Residuals')
        plt.title('Residuals Boxplot')
        plt.tight_layout()
        plt.show()

        # Correlation Matrix between Metrics
        plt.figure(figsize=(8, 6))
        corr_matrix = metrics_df[['MAE', 'RMSE', 'R2_Score', 'Mean_Error', 'Std_Error']].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Correlation Matrix of Performance Metrics')
        plt.tight_layout()
        plt.show()

        # **Plot Mean Error with Standard Deviation as a Line with Shaded Background**
        plt.figure(figsize=(14, 6))
        sns.lineplot(data=metrics_df, x='Timestep', y='Mean_Error', color='salmon', label='Mean Error')
        plt.fill_between(metrics_df['Timestep'],
                        metrics_df['Mean_Error'] - metrics_df['Std_Error'],
                        metrics_df['Mean_Error'] + metrics_df['Std_Error'],
                        color='salmon', alpha=0.2, label='Standard Deviation')
        plt.xlabel('Timestep')
        plt.ylabel('Mean Error (Predicted - Actual)')
        plt.title('Mean Error with Standard Deviation per Timestep')
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

        # Summary of Metrics
        print("==== Summary of Metrics ====")
        print(metrics_df.describe())

        return metrics_df, residuals
    

    def compute_permutation_importance(self, model, X_test_tensor, y_test_tensor, scaler_y, horizons, bnames, metric='mae'):

        import copy
        from tqdm import tqdm
        import pandas as pd
        import numpy as np

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()

        # Define loss function
        if metric == 'mae':
            loss_fn = nn.L1Loss()
        elif metric == 'mse':
            loss_fn = nn.MSELoss()
        else:
            raise ValueError("Unsupported metric. Choose 'mae' or 'mse'.")

        # Move data to device
        X_test = X_test_tensor.to(device)
        y_test = y_test_tensor.to(device)

        # Compute baseline loss
        with torch.no_grad():
            seq_lengths_val = (X_test.sum(dim=2) != 0).sum(dim=1)
            y_pred, _ = model(X_test, seq_lengths_val)
            y_pred_inverse = scaler_y.inverse_transform(
                y_pred.detach().cpu().numpy().reshape(-1, 1)
            ).reshape(-1, len(horizons))
            y_test_inverse = scaler_y.inverse_transform(
                y_test.detach().cpu().numpy().reshape(-1, 1)
            ).reshape(-1, len(horizons))
            absolute_error = np.abs(y_test_inverse - y_pred_inverse)
            baseline_mae = np.mean(absolute_error, axis=0)  # per timestep

        print(f"Baseline {metric.upper()} on Test Set: {baseline_mae.mean():.4f}")

        # Initialize importance DataFrame
        importance_df = pd.DataFrame(index=bnames, columns=horizons)
        print(f'{bnames=}')
        # Iterate over each feature and timestep
        for feature_idx, feature in tqdm(enumerate(bnames), total=len(bnames), desc="Features"):
            print(f'{feature=}')
            for t_idx, t in enumerate(horizons):
                # Create a copy of X_test_tensor
                X_permuted = X_test.clone()

                # Permute the feature at timestep t across all samples
                # Shuffling along the batch dimension (dim=0)
                X_permuted[:, t_idx, feature_idx] = X_permuted[:, t_idx, feature_idx][torch.randperm(X_permuted.size(0))]

                # Compute predictions on permuted data
                with torch.no_grad():
                    y_pred_permuted, _ = model(X_permuted, seq_lengths_val)
                    y_pred_permuted_inverse = scaler_y.inverse_transform(
                        y_pred_permuted.detach().cpu().numpy().reshape(-1, 1)
                    ).reshape(-1, len(horizons))
                    # Compute MAE on permuted data
                    if metric == 'mae':
                        permuted_mae = np.mean(np.abs(y_test_inverse - y_pred_permuted_inverse), axis=0)
                    else:
                        permuted_mae = np.mean((y_test_inverse - y_pred_permuted_inverse) ** 2, axis=0)

                # Compute the difference in MAE
                mae_diff = permuted_mae[t_idx] - baseline_mae[t_idx]

                # Store the difference in the DataFrame
                importance_df.at[feature, t] = mae_diff

        # Convert DataFrame to float
        importance_df = importance_df.astype(float)

        return importance_df
    

    def plot_permutation_importance(self, importance_df, top_n=10):
        """
        Plot the permutation feature importance as a stacked area plot with relative importance percentages.

        :param importance_df: DataFrame containing feature importance scores
        :param top_n: Number of top features to display in the plot
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd

        # Sum the importance over all timesteps for each feature
        importance_df['total_importance'] = importance_df.sum(axis=1)
        
        # Select the top N features based on total importance
        top_features = importance_df.sort_values('total_importance', ascending=False).head(top_n).index
        importance_df_top = importance_df.loc[top_features]
        
        # Sum the importance of all other features
        other_features = importance_df.index.difference(top_features)
        importance_df_other = importance_df.loc[other_features].drop('total_importance', axis=1)
        importance_df_other = pd.DataFrame(importance_df_other.sum(axis=0)).T
        importance_df_other.index = ['Other']
        
        # Combine the top features and 'Other'
        importance_df_top = importance_df_top.drop('total_importance', axis=1)
        importance_df_combined = pd.concat([importance_df_top, importance_df_other])
        
        # Compute total importance per timestep
        total_importance_per_timestep = importance_df_combined.sum(axis=0)
        
        # Avoid division by zero
        total_importance_per_timestep[total_importance_per_timestep == 0] = 1e-8
        
        # Compute relative importance percentages per timestep
        importance_df_relative = importance_df_combined.divide(total_importance_per_timestep, axis=1) * 100

        # Prepare data for stacking
        timesteps = importance_df_relative.columns.astype(float)
        features = importance_df_relative.index
        importance_values = importance_df_relative.values

        # Sort timesteps in case they're not in order
        sorted_indices = np.argsort(timesteps)
        timesteps = timesteps[sorted_indices]
        importance_values = importance_values[:, sorted_indices]
        
        # Plot stacked area chart
        plt.figure(figsize=(12, 8))
        plt.stackplot(timesteps, importance_values, labels=features)
        
        plt.xlabel('Timestep')
        plt.ylabel('Relative Importance (%)')
        plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
        plt.tight_layout()
        plt.show()

    def plot_permutation_importance_with_highlights_interactive(self, importance_df, highlight_features):
        """
        Plot the permutation feature importance heatmap and highlight the evolution of specific features interactively using Plotly.
        
        :param importance_df: DataFrame containing feature importance scores (features x timesteps)
        :param highlight_features: List of feature names to highlight and analyze
        """
        import numpy as np

        # Validate that highlight_features exist in the DataFrame
        missing_features = [f for f in highlight_features if f not in importance_df.index]
        if missing_features:
            print(f"Warning: The following features are not in the importance DataFrame and will be skipped: {missing_features}")
            highlight_features = [f for f in highlight_features if f in importance_df.index]
        
        if not highlight_features:
            print("No valid features to highlight. Exiting the function.")
            return
        
        # Heatmap
        fig = go.Figure(data=go.Heatmap(
            z=importance_df.values,
            x=importance_df.columns,
            y=importance_df.index,
            colorscale='Viridis',
            colorbar=dict(title='Increase in MAE'),
            hoverongaps=False
        ))
        
        # Add rectangles for highlighted features
        for feature in highlight_features:
            if feature in importance_df.index:
                feature_idx = importance_df.index.get_loc(feature)
                fig.add_trace(go.Scatter(
                    x=importance_df.columns,
                    y=[feature]*len(importance_df.columns),
                    mode='markers',
                    marker=dict(color='red', size=6),
                    name=f'Highlighted: {feature}'
                ))
        
        fig.update_layout(
            title='Permutation Feature Importance Over Time Steps',
            xaxis_title='Timestep',
            yaxis_title='Feature',
            height=800,
            width=1200
        )
        
        fig.show()
        
        # Line Plot for Highlighted Features
        fig_line = go.Figure()
        for feature in highlight_features:
            if feature in importance_df.index:
                fig_line.add_trace(go.Scatter(
                    x=importance_df.columns,
                    y=importance_df.loc[feature],
                    mode='lines+markers',
                    name=feature
                ))
        
        fig_line.update_layout(
            title='Evolution of Permutation Feature Importance for Selected Features',
            xaxis_title='Timestep',
            yaxis_title='Increase in MAE',
            height=600,
            width=800
        )
        
        fig_line.show()

class LSTMRollingForecaster:
    def __init__(self, model, data_prep, scaled_flight_features, time_horizons, device='cuda'):

        self.model = model.to(device)
        self.data_prep = data_prep
        self.scaled_features = scaled_flight_features.unsqueeze(0)  # Add batch dimension
        self.device = device
        self.time_horizons = [int(h) for h in time_horizons]
        self.max_seq_length = 61  # len(self.time_horizons)
        self.data_time_index = 0  # Current time index in the data
        self.current_features = None  # Features used for the current prediction
        self.predicted_takeoff_time = 300  # Initial predicted takeoff time
        self.start_time_index = 0  # Index from where the model starts using historical data
        self.model_time_index = 0  # Time index used in the model's output

        # Initialize current_features with the first time step
        self.current_features = self.scaled_features[:, :1, :]

        # Get the index of 't_to_eobt' feature (assumed to be the last feature)
        self.t_to_eobt_feature_index = -1  # Adjust if necessary

    def predict_next(self):
        current_tensor = self.current_features.to(self.device)
        seq_lengths = [current_tensor.size(1)]  # Since batch size is 1

        with torch.no_grad():
            output, _ = self.model(current_tensor, seq_lengths)

        self.model_time_index = max(0, min(self.model_time_index, 60))
        prediction = output[:, self.model_time_index, :].squeeze(-1).cpu().numpy()
        prediction_inverse = self.data_prep.scaler_y.inverse_transform(prediction.reshape(-1, 1)).reshape(-1)
        return prediction_inverse[-1]

    
    def update_features(self, start_time):

        # Adjust the sequence window for any rolling shifts or delays.
        if self.predicted_takeoff_time <= -300:
            # Use only the current feature if takeoff time is within initial range
            self.current_features = self.scaled_features[:, self.data_time_index:self.data_time_index + 1, :]
        else:
            if start_time == self.data_time_index:
                start_time = self.data_time_index

            # Dynamically adjust the indices to reflect the correct data window
            indices = list(range(start_time, self.data_time_index + 1))
            indices = indices[-self.max_seq_length:]  # Limit to max_seq_length
            
            if not indices:
                indices = [0]
            
            # Padding logic to ensure that the sequence length remains fixed
            if len(indices) < 61:
                padding_length = self.max_seq_length - len(indices)
                indices += [indices[-1]] * padding_length  # Pad with the last valid index
            indices = [i if i >= 0 else 0 for i in indices]
            indices = [i if 0 <= i < self.scaled_features.size(1) else -1 for i in indices]

            # print(f'{indices=}')
            # Set current features based on the prepared indices
            self.current_features = self.scaled_features[:, indices, :]

        return True
    
    def rolling_forecast(self):

        predictions = []
        current_horizon = 300  # Starting horizon (adjust as needed)

        while True:

            success = self.update_features(self.start_time_index)
            if torch.isnan(self.current_features).any():
                break
            prediction = self.predict_next()

            predictions.append(prediction)
            self.predicted_takeoff_time = current_horizon + prediction

            # Update start_time_index based on predicted takeoff time
            if self.predicted_takeoff_time >= 300:
                self.start_time_index = self.data_time_index
            else:
                # if  int(np.average(predictions[-1:])/5) >= self.start_time_index:
                self.start_time_index = int(np.round(np.average(predictions[-5:])/5,0)  ) 

            current_horizon -= 5  # Decrement current horizon

            if self.data_time_index >= self.scaled_features.shape[1]:
                break
            self.data_time_index += 1 
            self.model_time_index = max(0, min(self.data_time_index - self.start_time_index, 61))
            # print(f'{predictions=}')
            # print(f'{self.start_time_index=}')
        return predictions



def calculate_and_plot_errors(y_r, X_real, best_model, data_prep, ETOT_horizons, cbaslabels, start_idx=-1500, end_idx=-1400, target_length=61, bucket_size=5):
    from collections import defaultdict
    import numpy as np
    import matplotlib.pyplot as plt
    from tqdm import tqdm

    ap = []
    signed_errors = []
    error_dict = defaultdict(list)
    signed_error_dict = defaultdict(list)

    for fnr in tqdm(np.where((y_r <= 120))[0][start_idx:end_idx]):
        # Create rolling forecaster instance
        recursive = LSTMRollingForecaster(best_model, data_prep, X_real[fnr], ETOT_horizons)
        pred = recursive.rolling_forecast()
        
        # Calculate errors per timestep
        error_per_timestep = (pred[-target_length:] - y_r[fnr])[::-1]
        absolute_error_per_timestep = np.abs(error_per_timestep)
        
        ap.append(absolute_error_per_timestep)
        signed_errors.append(error_per_timestep)
        
        # Reverse for alignment and fetch relevant CBAS labels
        clabel = [x.astype('timedelta64[s]').astype(int) / 60 for x in cbaslabels[fnr] if not np.isnat(x)][-target_length:][::-1]
        absolute_error_per_timestep = absolute_error_per_timestep[-len(clabel):]
        error_per_timestep = error_per_timestep[-len(clabel):]
        
        # Populate the error dictionaries
        for i in range(len(absolute_error_per_timestep)):
            time_to_cbas = np.round(clabel[i], 0)
            error_dict[time_to_cbas].append(absolute_error_per_timestep[i])
            signed_error_dict[time_to_cbas].append(error_per_timestep[i])
    
    # Calculate MAE and mean signed error per timestep
    mae_per_timestep = np.nanmean(ap, axis=0)
    std_mae_per_timestep = np.nanstd(ap, axis=0)
    mean_signed_error_per_timestep = np.nanmean(signed_errors, axis=0)
    std_signed_error_per_timestep = np.nanstd(signed_errors, axis=0)
    
    # Group errors into time buckets
    bucketed_errors = defaultdict(list)
    bucketed_signed_errors = defaultdict(list)
    for time_in_seconds, errors in error_dict.items():
        time_in_minutes = time_in_seconds
        bucket = int(time_in_minutes // bucket_size) * bucket_size
        bucketed_errors[bucket].extend(errors)
        bucketed_signed_errors[bucket].extend(signed_error_dict[time_in_seconds])
    
    # Compute mean errors by bucket
    mean_errors_by_bucket = {bucket: np.nanmean(errors) for bucket, errors in bucketed_errors.items()}
    std_errors_by_bucket = {bucket: np.nanstd(errors) for bucket, errors in bucketed_errors.items()}
    mean_signed_errors_by_bucket = {bucket: np.nanmean(errors) for bucket, errors in bucketed_signed_errors.items()}
    std_signed_errors_by_bucket = {bucket: np.nanstd(errors) for bucket, errors in bucketed_signed_errors.items()}
    
    # Plot MAE per timestep
    plt.figure(figsize=(10, 6))
    plt.plot(ETOT_horizons[:target_length], mae_per_timestep[::-1], marker='o', label="Mean Absolute Error")
    plt.fill_between(ETOT_horizons[:target_length], 
                     mae_per_timestep[::-1] - std_mae_per_timestep[::-1], 
                     mae_per_timestep[::-1] + std_mae_per_timestep[::-1], 
                     alpha=0.2, label="Std Dev")
    plt.xlabel('Timestep')
    plt.xticks(ETOT_horizons[:target_length], rotation='vertical')
    plt.ylabel('Mean Absolute Error (MAE)')
    plt.title('MAE for Each Timestep')
    plt.tight_layout()
    plt.grid(True)
    plt.legend()
    plt.show()
    
    # Plot Mean Signed Error per Timestep
    plt.figure(figsize=(10, 6))
    plt.plot(ETOT_horizons[:target_length], mean_signed_error_per_timestep[::-1], marker='o', color='orange', label="Mean Signed Error")
    plt.fill_between(ETOT_horizons[:target_length], 
                     mean_signed_error_per_timestep[::-1] - std_signed_error_per_timestep[::-1], 
                     mean_signed_error_per_timestep[::-1] + std_signed_error_per_timestep[::-1], 
                     alpha=0.2, color='orange', label="Std Dev")
    plt.xlabel('Timestep')
    plt.xticks(ETOT_horizons[:target_length], rotation='vertical')
    plt.ylabel('Mean Signed Error')
    plt.title('Mean Signed Error for Each Timestep')
    plt.tight_layout()
    plt.grid(True)
    plt.legend()
    plt.show()
    
    # Plot Mean Absolute Error by Time Buckets
    sorted_buckets = sorted(mean_errors_by_bucket.items())
    times = [-item[0] for item in sorted_buckets]
    errors = [item[1] for item in sorted_buckets]
    std_errors = [std_errors_by_bucket[item[0]] for item in sorted_buckets]
    
    plt.figure(figsize=(10, 6))
    plt.plot(times, errors, marker='o', linestyle='-', label="Mean Absolute Error")
    # plt.fill_between(times, 
    #                  np.array(errors) - np.array(std_errors), 
    #                  np.array(errors) + np.array(std_errors), 
    #                  alpha=0.2, label="Std Dev")
    plt.title("Mean Prediction Absolute Error vs Time to CBAS Entry (bucketed in 5-minute intervals)")
    plt.xlabel("Time to CBAS Entry (minutes)")
    plt.ylabel("Mean Prediction Absolute Error (minutes)")
    plt.grid(True)
    plt.legend()
    plt.show()
    
    # Plot Mean Signed Error by Time Buckets
    sorted_signed_buckets = sorted(mean_signed_errors_by_bucket.items())
    times_signed = [-item[0] for item in sorted_signed_buckets]
    signed_errors = [item[1] for item in sorted_signed_buckets]
    std_signed_errors = [std_signed_errors_by_bucket[item[0]] for item in sorted_signed_buckets]
    
    plt.figure(figsize=(10, 6))
    plt.plot(times_signed, signed_errors, marker='o', linestyle='-', color='orange', label="Mean Signed Error")
    # plt.fill_between(times_signed, 
    #                  np.array(signed_errors) - np.array(std_signed_errors), 
    #                  np.array(signed_errors) + np.array(std_signed_errors), 
    #                  alpha=0.2, color='orange', label="Std Dev")
    plt.title("Mean Prediction Signed Error vs Time to CBAS Entry (bucketed in 5-minute intervals)")
    plt.xlabel("Time to CBAS Entry (minutes)")
    plt.ylabel("Mean Prediction Signed Error (minutes)")
    plt.grid(True)
    plt.legend()
    plt.show()
    
    return mae_per_timestep, mean_errors_by_bucket, mean_signed_errors_by_bucket
