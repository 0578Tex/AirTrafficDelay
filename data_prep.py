import numpy as np
import torch
import itertools
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import pandas as pd
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.stats import ks_2samp
from constants import *
import seaborn as sns
import pickle
import os
from collections import defaultdict
from tqdm import tqdm
import itertools
from flights import Flight, Flights

class DataPreparation:
    def __init__(self, fixed_columns=None, prefix='_Tmin_', mean_threshold=0.1, var_threshold=0.1):

        self.fixed_columns = fixed_columns or []
        self.prefix = prefix
        self.mean_threshold = mean_threshold
        self.var_threshold = var_threshold
        self.scaler_X_fixed = StandardScaler()
        self.scalers_X_time_varying = {}  # Dictionary to store scalers for each base feature
        self.scaler_y = StandardScaler()
        self.input_size = None
        self.time_horizons = None
        self.cbaslabels = None
        self.feature_settings = {}
        self.feature_name_to_idx = None  # Store feature mapping
        self.binary_features = ['modeltyp_EST', 'modeltyp_CAL', 'modeltyp_ACT', 'fltstate_FI', 'fltstate_SI', 'fltstate_other']
        
        # Store scaler for 't_to_eobt' feature separately
        self.t_to_eobt_scaler = StandardScaler()


    def fit_transform_data(self, X, y, split_ratio=0.8, mode='lstm'):

        # Step 1: Separate time-varying and fixed features
        time_varying_columns, time_CBAS_columns, X_fixed, horizons = self._separate_features(X)
        print(f'{time_varying_columns=}')
        print(f'{self.fixed_columns=}')
        N = len(X)
        # print(f'{X_fixed.describe()}')
        # Step 2: Extract time horizons and sort them
        self._extract_time_horizons(time_varying_columns)

        # Step 3: Standardize fixed features
        X_fixed_scaled = self.scaler_X_fixed.fit_transform(X_fixed)

        # Step 4: Standardize time-varying features based on base feature names
        X_time_varying_scaled_df = self._scale_time_varying_features(X, time_varying_columns, fit=True)
        self.X_time_varying_scaled_df = X_time_varying_scaled_df
        # Step 5: Build time-varying sequences and store feature mapping
        X_time_varying_scaled, self.feature_name_to_idx = self._build_time_varying_sequences_from_df(
            X_time_varying_scaled_df, time_varying_columns, N, self.time_horizons
        )

        # Step 6: Concatenate fixed and time-varying features
        X_final = self._concatenate_fixed_time_varying(X_fixed_scaled, X_time_varying_scaled, horizons)

        # Step 7: Standardize y
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1))

        # Step 8: Train-test split
        X_train, X_test, y_train, y_test = self._train_test_split(X_final, y_scaled, split_ratio)

        # Step 9: Extract CBAS labels and maintain structure
        split_index = int(N * split_ratio)
        self.cbaslabels = self._extract_cbas_labels(X, time_CBAS_columns, split_index=split_index)

        # Step 10: Create PyTorch tensors and DataLoaders
        self._create_tensors_and_loaders(X_train, X_test, y_train, y_test)

        # Store feature settings for consistency checks during transform
        self.feature_settings = {
            "fixed_columns": self.fixed_columns,
            "time_varying_columns": time_varying_columns,
            "time_horizons": self.time_horizons,
            "scalers_X_time_varying": list(self.scalers_X_time_varying.keys())
        }

        # Data preparation succeeded
        if mode == 'rf':
            return X_train, X_test, y_train, y_test 

        return self.X_train_tensor, self.y_train_tensor, self.X_test_tensor, self.y_test_tensor, self.time_horizons, self.cbaslabels

    def transform_data(self, X, update_t_to_eobt_values=None, split_ratio=0.8):

        # Step 1: Separate time-varying and fixed features
        time_varying_columns, time_CBAS_columns, X_fixed, horizons = self._separate_features(X)
        print(f'{time_varying_columns=}')
        N = len(X)
        X_fixed = X_fixed.fillna(0)

        # Step 2: Handle unseen fixed columns
        X_fixed_scaled = self._handle_unseen_fixed_columns(X_fixed)

        # Step 3: Standardize fixed features
        X_fixed_scaled = self.scaler_X_fixed.transform(X_fixed_scaled)

        # Step 4: Standardize time-varying features
        X_time_varying_scaled_df = self._scale_time_varying_features(X, time_varying_columns, fit=False, update_t_to_eobt_values=update_t_to_eobt_values)

        # Step 5: Build time-varying sequences using stored feature mapping
        X_time_varying_scaled, _ = self._build_time_varying_sequences_from_df(
            X_time_varying_scaled_df, time_varying_columns, N, horizons, feature_name_to_idx=self.feature_name_to_idx
        )

        cbaslabels = self._extract_cbas_labels(X, time_CBAS_columns, split_index=1)

        # Step 6: Concatenate fixed and time-varying features
        X_final = self._concatenate_fixed_time_varying(X_fixed_scaled, X_time_varying_scaled, horizons)

        # Convert to PyTorch tensor
        X_tensor = torch.tensor(X_final, dtype=torch.float32)
        return X_tensor, horizons, cbaslabels

    def _handle_unseen_fixed_columns(self, X_fixed):

        # Identify columns present in training but missing in test
        missing_fixed = set(self.fixed_columns) - set(X_fixed.columns)
        if missing_fixed:
            print(f"**Missing Fixed Columns in Test Data:** {missing_fixed}")
            # Fill missing columns with zeros
            for col in missing_fixed:
                X_fixed[col] = 0.0

        # Identify columns present in test but not in training
        extra_fixed = set(X_fixed.columns) - set(self.fixed_columns)
        if extra_fixed:
            print(f"**Extra Fixed Columns in Test Data (will be dropped):** {extra_fixed}")
            # Drop extra columns
            X_fixed = X_fixed[self.fixed_columns]

        return X_fixed

    def _separate_features(self, X):

        # Identify time-varying columns based on prefix and exclude CBAS columns
        skip = ['CBAS', 'cbas', 'eobt', 'atot']
        time_varying_columns = [col for col in X.columns if self.prefix in col and col not in skip]
        unique_values_after_prefix = set([col.split(self.prefix)[1] for col in time_varying_columns])
        horizons = sorted(unique_values_after_prefix, key=int)
        # Identify CBAS columns (if any)
        time_CBAS_columns = [col for col in X.columns if self.prefix in col and ('CBAS' in col or 'cbas' in col)]

        # Keep fixed columns consistent with the ones used during training
        if not self.fixed_columns:
            self.fixed_columns = [col for col in X.columns if col not in time_varying_columns and col not in skip]

        X_fixed = X[self.fixed_columns].copy()  # shape: (N, num_fixed_features)
        return time_varying_columns, time_CBAS_columns, X_fixed, horizons

    def _extract_time_horizons(self, time_varying_columns):

        self.time_horizons = sorted(set([col.split(self.prefix)[1] for col in time_varying_columns]), key=int)
        self.time_steps = len(self.time_horizons)

    def _scale_time_varying_features(self, X, time_varying_columns, fit=True, update_t_to_eobt_values=None):
        X_time_varying_df = X[time_varying_columns].copy()
        X_time_varying_scaled_df = pd.DataFrame(index=X_time_varying_df.index)
        base_feature_names = sorted(set([col.split(self.prefix)[0] for col in time_varying_columns]))
        self.base_feature_names= base_feature_names

        self.basenames = self.fixed_columns + base_feature_names

        for feature in base_feature_names:
            # Get all columns for this base feature
            feature_columns = [col for col in time_varying_columns if col.startswith(feature + self.prefix)]
            if not feature_columns:
                continue  # Skip if the feature columns are missing (e.g., in new data)

            # Get data for this feature
            feature_data = X_time_varying_df[feature_columns]
            # print(f'{feature=} { feature in ['etodepdelay', 'TSATdelay', 'TOBTdelay']=}   { np.nanmean(feature_data)}')
            if feature in ['etodepdelay', 'TSATdelay', 'TOBTdelay'] and  np.nanmean(feature_data) < 0:
                print(f'{feature} switched')
                feature_data = -feature_data

            # Skip scaling for binary features
            if feature in self.binary_features:
                X_time_varying_scaled_df[feature_columns] = feature_data
                continue

            # Flatten the data
            feature_data_values = feature_data.values.flatten()

            if fit:
                # Fit scaler ignoring NaNs
                scaler = StandardScaler()
                scaler.fit(feature_data_values[~np.isnan(feature_data_values)].reshape(-1, 1))
                self.scalers_X_time_varying[feature] = scaler
            else:
                # Use existing scaler
                scaler = self.scalers_X_time_varying.get(feature)
                if scaler is None:
                    print(f"**Scaler for feature '{feature}' not found. Skipping scaling for this feature.**")
                    X_time_varying_scaled_df[feature_columns] = feature_data
                    continue  # Skip scaling for this feature

            # Transform data while preserving NaNs
            scaled_values = np.full_like(feature_data_values, np.nan)
            non_nan_indices = ~np.isnan(feature_data_values)
            scaled_values[non_nan_indices] = scaler.transform(feature_data_values[non_nan_indices].reshape(-1, 1)).flatten()
            feature_data_scaled = scaled_values.reshape(feature_data.shape)

            # Assign scaled data back to the DataFrame
            X_time_varying_scaled_df[feature_columns] = pd.DataFrame(feature_data_scaled, columns=feature_columns, index=feature_data.index)

        return X_time_varying_scaled_df

    def _build_time_varying_sequences_from_df(self, X_time_varying_scaled_df, time_varying_columns, N, horizons, feature_name_to_idx=None):
        # Ensure that 't_to_eobt' appears as the last feature in each time step
        if feature_name_to_idx is None:
            # Identify and sort features, placing 't_to_eobt' at the end
            base_feature_names = sorted(set([col.split(self.prefix)[0] for col in time_varying_columns]))

            feature_name_to_idx = {name: idx for idx, name in enumerate(base_feature_names)}
            self.feature_name_to_idx = feature_name_to_idx
        else:
            feature_name_to_idx = self.feature_name_to_idx

        # Define the shape of the sequence array
        num_time_varying_features_per_timestep = len(feature_name_to_idx)
        X_time_varying_scaled = np.zeros((N, len(horizons), num_time_varying_features_per_timestep))

        # Populate the sequence array
        for idx_t, t in enumerate(horizons):
            # Collect columns for the current time horizon
            cols_t = [col for col in X_time_varying_scaled_df.columns if col.endswith(f'{self.prefix}{t}')]
            if not cols_t:
                continue  # Skip if no columns are present for this timestep

            # Assign feature values to their designated positions
            for col in cols_t:
                base_feature = col.split(self.prefix)[0]
                feature_idx = feature_name_to_idx.get(base_feature)
                if feature_idx is None:
                    print(f"**Base feature '{base_feature}' not found in feature_name_to_idx mapping. Skipping column '{col}'.**")
                    continue
                X_time_varying_scaled[:, idx_t, feature_idx] = X_time_varying_scaled_df[col].values

        return X_time_varying_scaled, feature_name_to_idx


    def _concatenate_fixed_time_varying(self, X_fixed_scaled, X_time_varying_scaled, horizons):
        # Repeat fixed features across all time horizons
        X_fixed_repeated = np.repeat(X_fixed_scaled[:, np.newaxis, :], len(horizons), axis=1)

        X_final = np.concatenate([X_fixed_repeated, X_time_varying_scaled], axis=2)  # shape: (N, T, num_total_features)

        self.input_size = X_final.shape[2]
        return X_final

    def _train_test_split(self, X_final, y_scaled, split_ratio):

        split_index = int(len(X_final) * split_ratio)
        X_train, X_test = X_final[:split_index], X_final[split_index:]
        y_train, y_test = y_scaled[:split_index], y_scaled[split_index:]
        return X_train, X_test, y_train, y_test

    def _extract_cbas_labels(self, X, time_CBAS_columns, split_index):

        y_cbas_label = X[time_CBAS_columns].values  # Shape: (N, num_cbas_labels)
        return y_cbas_label[split_index:]

    def _create_tensors_and_loaders(self, X_train, X_test, y_train, y_test):

        # Convert to PyTorch tensors
        self.X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        self.X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

        y_train_repeated = np.repeat(y_train, self.time_steps, axis=1).reshape(-1, self.time_steps, 1)
        y_test_repeated = np.repeat(y_test, self.time_steps, axis=1).reshape(-1, self.time_steps, 1)

        self.y_train_tensor = torch.tensor(y_train_repeated, dtype=torch.float32)
        self.y_test_tensor = torch.tensor(y_test_repeated, dtype=torch.float32)

        # Create DataLoader instances for batching
        train_dataset = TensorDataset(self.X_train_tensor, self.y_train_tensor)
        test_dataset = TensorDataset(self.X_test_tensor, self.y_test_tensor)

        self.train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    def inverse_transform_single_flight(self, X_scaled_single_flight):
        """
        Reverts the transformation for a single flight/sequence.

        :param X_scaled_single_flight: Scaled and transformed data for a single flight.
                                    Shape: (T, num_total_features)
        :return: Original data for the single flight as a DataFrame.
        """
        import pandas as pd
        import numpy as np

        num_fixed_features = len(self.fixed_columns)
        num_time_steps = X_scaled_single_flight.shape[0]

        # Extract fixed features (same across all time steps)
        fixed_features_scaled = X_scaled_single_flight[0, :num_fixed_features]
        fixed_features_original = self.scaler_X_fixed.inverse_transform(
            fixed_features_scaled.reshape(1, -1)
        ).flatten()
        fixed_features_dict = dict(zip(self.fixed_columns, fixed_features_original))

        # Prepare DataFrame for time-varying features
        base_feature_names = list(self.feature_name_to_idx.keys())
        time_varying_features_df = pd.DataFrame(
            index=self.time_horizons, columns=base_feature_names
        )

        X_time_varying_scaled = X_scaled_single_flight[:, num_fixed_features:]  # Shape: (T, num_time_varying_features)

        for t_idx, t in enumerate(self.time_horizons):
            for base_feature in base_feature_names:
                feature_idx = self.feature_name_to_idx[base_feature]
                scaled_value = X_time_varying_scaled[t_idx, feature_idx]

                if np.isnan(scaled_value) or scaled_value in [np.inf, -np.inf]:
                    original_value = np.nan
                elif base_feature in self.binary_features:
                    original_value = scaled_value  # No scaling was applied to binary features
                else:
                    scaler = self.scalers_X_time_varying.get(base_feature)
                    if scaler is not None:
                        scaled_value = np.array([[scaled_value]])  # Reshape to (1, 1)
                        original_value = scaler.inverse_transform(scaled_value)[0, 0]
                    else:
                        print(f"Warning: No scaler found for feature '{base_feature}'. Assigning scaled value directly.")
                        original_value = scaled_value  # If no scaler, assume no scaling was applied

                # Round the original_value to 2 decimal places
                if not np.isnan(original_value):
                    original_value = np.round(original_value, 2)

                time_varying_features_df.at[t, base_feature] = original_value

        # Flatten the time-varying features to match original column names
        time_varying_features_flat = {}
        for t in self.time_horizons:
            for base_feature in base_feature_names:
                col_name = f"{base_feature}{self.prefix}{t}"
                time_varying_features_flat[col_name] = time_varying_features_df.at[t, base_feature]

        # Combine fixed and time-varying features
        all_features_dict = fixed_features_dict.copy()
        all_features_dict.update(time_varying_features_flat)

        # Convert to DataFrame
        original_features_df = pd.DataFrame([all_features_dict])

        return original_features_df
    

def map_airport_details(row, airport_dict, airport_type):
    airport_code = row[airport_type]
    airport_info = airport_dict.get(airport_code, {})
    return {
        f'{airport_type}_capacity': airport_info.get('capacity', None),
        f'{airport_type}Long': airport_info.get('longitude', None),
        f'{airport_type}Lat': airport_info.get('latitude', None),
    }


def dummies_encode_efd(P, airport = None, save_template=True):
    """
    Perform dummy encoding and ensure column consistency.
    If save_template is True, saves the dummy columns as a template for future runs.
    """
    if airport is None:
        cdm_columns = [col for col in P.columns if 'cdm' in col]
        P = P.drop(cdm_columns, axis=1)
        dum_cols = ["ADEP", "ADES", "day_of_week"]

        # Perform dummy encoding
        new_df3 = pd.get_dummies(P, columns=dum_cols)

        # Save the columns template for future runs
        if save_template:
            save_dummy_template(new_df3.columns.tolist(), DUMMY_TEMPLATE_FILE)

    else:
        drop_columns = ["ADES", "FiledAT", "ACType", "ArrivalDelay"]
        new_df = P.drop(columns=[col for col in drop_columns if col in P.columns], axis=1)
        dum_cols = ["ADEP", "ACOperator", "month", "weekday", "operator", "actype", "flighttype"]

        # Perform dummy encoding
        new_df3 = pd.get_dummies(new_df, columns=dum_cols)

    # Adjust columns to match the template if it exists
    template_columns = load_dummy_template(DUMMY_TEMPLATE_FILE)
    if template_columns:
        # Add missing columns with NaN
        for col in template_columns:
            if col not in new_df3.columns:
                new_df3[col] = np.nan
        # Ensure column order matches the template
        # new_df3 = new_df3[template_columns]

    return new_df3

def time_to_circular(time_in_minutes, period=1440):
    """
    Convert a time in minutes to a circular (sine, cosine) representation.
    `period`: The number of minutes in a full cycle (1440 minutes = 24 hours).
    """
    sin_component = np.sin(2 * np.pi * time_in_minutes / period)
    cos_component = np.cos(2 * np.pi * time_in_minutes / period)
    return sin_component, cos_component



def filtering_data(extended_df, airport='EHAM', save=False):

    dform = "%Y-%m-%d %H:%M:%S"
    airport2 = 'EGLL'
    # print(df)
    df = (
        extended_df.query("ADES== @airport ") # |ADES== @airport" / ADEP@ ==airport   & ADEP == @airport2
        .assign(EOBT=lambda x: pd.to_datetime(x.EOBT, format=dform))
        .assign(ETA=lambda x: pd.to_datetime(x.ETA, format=dform))
        .assign(day_of_week=lambda x: x.EOBT.dt.weekday)  # Add day_of_week column using .weekday() from EOBT
        .sort_values(by='EOBT')  # Sort by FiledOBT after conversion
    )
    if 'timetoCBAS_Tmin_5' in df.columns:
        timeto_columns = [col for col in df.columns if col.startswith('timetoCBAS_Tmin_')]

        # Step 2: Create a function to find the last non-NaN value for each row
        def last_valid_timedelta(row):
            last_valid = row[timeto_columns].dropna().iloc[-1]  # Get the last non-NaN value
            return last_valid

        # Step 3: Apply this function to each row and filter the rows
        df['last_timetoCBAS_Tmin_'] = df.apply(last_valid_timedelta, axis=1)

        # Step 4: Filter rows where the last non-NaN timetoCBAS_Tmin_ is greater than pd.Timedelta(0)
        df = df[df['last_timetoCBAS_Tmin_'] > pd.Timedelta(0)]

        df = df.drop('last_timetoCBAS_Tmin_', axis=1)
    else:
        df = df[(df['timetoCBAS_Tmin_0'] > pd.Timedelta(0)) | (df['timetoCBAS_Tmin_0'].isna())]
    # df = df[(df['timetoCBAS_Tmin_0'] > pd.Timedelta(0)) | (df['timetoCBAS_Tmin_0'].isna())]
    df = df.join(df.apply(lambda row: pd.Series(map_airport_details(row, airport_dict, 'ADEP')), axis=1))
    df = df.join(df.apply(lambda row: pd.Series(map_airport_details(row, airport_dict, 'ADES')), axis=1))
    # print(f'df = \n{df.describe()}')
    df = df.drop(['sid', 'star'], axis =1)
    df_3 = dummies_encode_efd(df, airport=None, save_template=save)


    y = df["delay"].to_numpy()
    # print(f'yyyyyy =')
    df_3 = df_3.drop(['delay', 'EOBT', 'event', 'CDMStatus', 'TSAT', 'TOBT', 'taxitime', 'regulations', 'dep_status', 'atfmdelay', 'fltstate'], axis=1)
    df_3['ETOT_minutes'] = df_3['ETOT'].dt.hour * 60 + df_3['ETOT'].dt.minute
    df_3 = df_3.drop('ETOT', axis=1) 
    df_3['ETA_minutes'] = np.where(
        df_3['ETA'].isna(), 
        0, 
        df_3['ETA'].dt.hour * 60 + df_3['ETA'].dt.minute
    )    
    df_3 = df_3.drop('ETA', axis=1)
    df_3['sin_ETOT'], df_3['cos_ETOT'] = time_to_circular(df_3['ETOT_minutes'])
    df_3 = df_3.drop('ETOT_minutes', axis=1)  # Drop the intermediate minutes column

    # Apply the circular transformation for ETA_minutes
    df_3['sin_ETA'], df_3['cos_ETA'] = time_to_circular(df_3['ETA_minutes'])
    df_3 = df_3.drop('ETA_minutes', axis=1)  # Drop the intermediate minutes column


    #convert T/F columns to 0/1:
    bool_columns = df_3.select_dtypes(include=['bool']).columns.tolist()
    df_3[bool_columns] = df_3[bool_columns].astype(int)

    X_final = pd.DataFrame(df_3)
    # X_final = pd.concat([X_final, df['EOBT']])
    return X_final, y, df_3.columns


DUMMY_TEMPLATE_FILE = r"C:\Users\iLabs_6\Documents\Tex\realtimetest\templates\template.pkl"

# Function to save column headers template
def save_dummy_template(dummy_columns, template_file):
    with open(template_file, 'wb') as file:
        pickle.dump(dummy_columns, file)

# Function to load column headers template
def load_dummy_template(template_file):
    if os.path.exists(template_file):
        with open(template_file, 'rb') as file:
            dummy_columns = pickle.load(file)
        return dummy_columns
    return None


def process_flight_data(f, period=1440):
    """
    Processes flight data to extract specific information:
    time, day, ADEP, distance, longitude, and latitude.
    
    Args:
        f (DataFrame): Input data frame containing flight information.
        period (int): The period in minutes (default 1440 for a day).
        
    Returns:
        dict: Dictionary containing extracted flight data.
    """
    # Compute time
    sin_component = f['sin_ETOT']
    cos_component = f['cos_ETOT']
    angle = np.arctan2(sin_component, cos_component)
    angle = np.where(angle < 0, angle + 2 * np.pi, angle)
    time_in_minutes = (angle / (2 * np.pi)) * period
    time = time_in_minutes[0]
    
    # Extract day
    daycols = f[[col for col in f.columns if 'day' in col]]
    day = daycols.max().idxmax()
    
    # Extract ADEP
    adepdf = f[[col for col in f.columns if ("ADEP_" in col and col != "ADEP_capacity")]]
    adep = adepdf.max().idxmax()
    
    # Extract distance
    distance = f['distance'][0]
    
    # Extract longitude
    longitude = f['ADEPLong'][0]
    
    # Extract latitude
    latitude = f['ADEPLat'][0]
    
    # Compile results into a dictionary
    result = {
        'time': time,
        'day': day,
        'adep': adep,
        'distance': distance,
        'longitude': longitude,
        'latitude': latitude
    }
    
    return result


def load_chunks(file_prefix, chunk_size=5000):
    """Load chunks of a dictionary and combine them into a single dictionary."""
    combined_data = {}
    i = 0

    while True:
        # Construct the filename for each chunk
        chunk_file = os.path.join(file_prefix, f"efd_{i}.pkl")
        
        # Check if the chunk file exists
        if not os.path.exists(chunk_file):
            break  # No more chunks to load, exit the loop

        print(f"Loading chunk {i} from {chunk_file}...")
        
        # Load the chunk and update the combined dictionary
        with open(chunk_file, 'rb') as f:
            chunk_data = pickle.load(f)
            combined_data.update(chunk_data)
        
        print(f"Chunk {i} loaded and combined.")
        i += 1

    print(f"All {i} chunks have been loaded and combined successfully.")
    return combined_data



def save_chunks(data, file_prefix, chunk_size=5000):
    """Save a large dictionary in smal
    ler chunks to avoid memory issues."""
    # Convert the data items to an iterator
    data_iter = iter(data.items())
    
    # Save chunks using itertools.islice
    for i, chunk in enumerate(iter(lambda: dict(itertools.islice(data_iter, chunk_size)), {})):
        print(f'{i=}')
        file_name = os.path.join(file_prefix, f"efd_{i}.pkl")
        
        print(f"Saving chunk {i} as {file_name}...")
        with open(file_name, 'wb') as f:
            pickle.dump(chunk, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Chunk {i+1} saved as {file_name}")

    print(f"All {i+1} chunks have been saved successfully.")



def process_flight_manager(file_path, start_date, end_date, taf_reports=None, reg_reports=None):
    all_flights = {}
    sort = defaultdict(list)
    usecount, totalcount = 0, 0

    with open(file_path, 'rb') as f:
        # Only load necessary data from file
        flight_manager = pickle.load(f)
    
    for flight in flight_manager.flights.values():
        # Perform only necessary processing
        if not flight.Adep in airports_top50 or not flight.EOBT or not flight.get_first(flight.EOBT):
            continue
    
        totalcount += 1
        
        # Early exit for flights outside the date range
        if not (start_date <= flight.get_first(flight.EOBT) < end_date):
            continue
        
        if flight.delay and (-60 <= flight.delay <= 120):
            if taf_reports:
                flight.update_weather_data(1, taf_reports)
            if reg_reports:
                flight.add_regulation(reg_reports)
            usecount += 1
            sort[flight.reg].append(flight)
    
    new_flights, updated = {}, 0
    sorted_flights = {}
    for plane, flights in sort.items():
        sorted_flights = sorted(flights, key=lambda x: x.timestamp[-1])

        for i, flight in enumerate(sorted_flights):
            if flight.Ades != 'EHAM':
                continue
            new_flights[(flight.Adep, flight.Ades, flight.filed['EOBT'])] = flight

            if i > 0 and sorted_flights[i - 1].Ades == flight.Adep:
                flight.prev = sorted_flights[i - 1]

                updated += 1
    if sorted_flights:
        del sorted_flights
        gc.collect()
    return new_flights, totalcount, usecount


def get_efd_rf(archive, start_date, end_date, taf_reports=None, reg_reports=None, reload=False, temppath=r"C:\Users\iLabs_6\Documents\Tex\allwithCBAS\tempefd"):
    # print(f'tafreports = {taf_reports}')

    all_flights = {}
    sort = defaultdict(list)
    newall = {}
    total_useful, total_processed = 0, 0
    temp_pickle_files = []
    files = os.listdir(archive)
    if any('efd' in f for f in files) and not reload:
        return load_chunks(archive)

    for i, file in enumerate(tqdm(files)):
        if not 'chunk' in file:
            continue
        file_path = os.path.join(archive, file)
        
        file_new, totalcount, usecount = process_flight_manager(file_path, start_date, end_date, taf_reports, reg_reports)
        temp_file = os.path.join(temppath, f'efd_{i}.pkl')
        with open(temp_file, 'wb') as f:
            pickle.dump(file_new, f)
        temp_pickle_files.append(temp_file)  

        total_processed += totalcount
        total_useful += usecount
        del file_new 
        gc.collect()  

    newall = {}
    for temp_file_path in tqdm(temp_pickle_files):
        with open(temp_file_path, 'rb') as f:
            file_new = pickle.load(f)
            newall.update(file_new)  
        # os.remove(temp_file_path)
    print(f"Total processed: {total_processed}, Useful: {total_useful}")
    save_chunks(newall, archive)
    return newall



def merge(df, extended_df):

    # Convert 'FiledOBT' in both datasets to datetime format to ensure they are comparable
    df['FiledOBT'] = pd.to_datetime(df['FiledOBT'])
    extended_df['FiledOBT_efd'] = pd.to_datetime(extended_df['FiledOBT'])

    # Identify overlapping columns (except for 'ADEP', 'ADES', and 'FiledOBT')
    common_columns = set(df.columns).intersection(set(extended_df.columns)) - {'ADEP', 'ADES', 'FiledOBT'}
    print(f"Overlapping columns: {common_columns}")

    # Perform the merge, using suffixes to avoid losing information from df_top50_flights
    merged_df = df.merge(
        extended_df, 
        on=['ADEP', 'ADES'], 
        suffixes=('_df', '_extended'),  # Apply suffixes for overlapping columns
        how='left'  # Use 'left' join to preserve all information from df_top50_flights
    )

    # Filter to only keep rows where 'FiledOBT' from extended_df is within 1 hour of 'FiledOBT' from df
    one_hour_margin = pd.Timedelta(hours=1)
    merged_df = merged_df[
        (merged_df['FiledOBT_extended'] >= merged_df['FiledOBT_df'] - one_hour_margin) &
        (merged_df['FiledOBT_extended'] <= merged_df['FiledOBT_df'] + one_hour_margin)
    ]
    merged_df['FiledOBT'] = merged_df['FiledOBT_df']

    # If 'FiledOBT_extended' is no longer needed, drop it
    merged_df = merged_df.drop(columns=['FiledOBT_extended', 'FiledOBT_efd', 'FiledOBT', 'id', 'FiledOBT_df'])

    # View the merged dataframe
    # print(merged_df.head())

    # Check the shape of the merged dataframe
    print(f"Shape of the merged dataframe: {merged_df.shape}")

    # Check columns in the merged dataframe
    print(f"Columns in merged_df: {merged_df.columns.tolist()}")
    return merged_df
