import pandas as pd
import re
from datetime import datetime
import os
import py7zr
from tqdm import tqdm
import tempfile
import gc
from constants import *

# Define column names based on Eurocontrol SO6 specification
columns = ['segment_identifier', 'flight_origin', 'flight_destination', 'aircraft_type', 
           'time_begin', 'time_end', 'fl_begin', 'fl_end', 'status', 'callsign', 
           'date_begin', 'date_end', 'lat_begin', 'lon_begin', 'lat_end', 'lon_end',
           'flight_identifier', 'sequence', 'length', 'parity']

# Function to convert the custom date format in SO6 files
def convert_so6_time(df, date_col, time_col):
    combined_str = df[date_col] + df[time_col]
    return pd.to_datetime(combined_str, format='%y%m%d%H%M%S', errors='coerce')

# Function to optimize data types for memory efficiency
def optimize_data_types(df):
    # Convert object types (strings) to categories if there are few unique values
    for col in ['flight_origin', 'flight_destination', 'aircraft_type', 'callsign', 'status']:
        df[col] = df[col].astype('category')

    # Convert numeric columns to appropriate types
    df['lat_begin'] = pd.to_numeric(df['lat_begin'], errors='coerce', downcast='float')
    df['lon_begin'] = pd.to_numeric(df['lon_begin'], errors='coerce', downcast='float')
    df['lat_end'] = pd.to_numeric(df['lat_end'], errors='coerce', downcast='float')
    df['lon_end'] = pd.to_numeric(df['lon_end'], errors='coerce', downcast='float')
    df['fl_begin'] = pd.to_numeric(df['fl_begin'], errors='coerce', downcast='integer')
    df['fl_end'] = pd.to_numeric(df['fl_end'], errors='coerce', downcast='integer')
    df['length'] = pd.to_numeric(df['length'], errors='coerce', downcast='float')

    return df

# Function to read the SO6 file and parse its content
def read_so6_file(file_path):
    data = []
    
    # Read the SO6 file line by line
    with open(file_path, 'r') as f:
        for line in f:
            # Use regex to split the line based on spaces (multiple spaces treated as a single delimiter)
            split_line = re.split(r'\s+', line.strip())
            
            # Ensure the line has the correct number of columns
            if len(split_line) == len(columns):
                data.append(split_line)

    # Create a pandas DataFrame from the parsed data
    df = pd.DataFrame(data, columns=columns)

    # Convert date and time columns to proper datetime format using vectorized operations
    df['datetime_begin'] = convert_so6_time(df, 'date_begin', 'time_begin')
    df['datetime_end'] = convert_so6_time(df, 'date_end', 'time_end')

    # Convert latitude and longitude columns from degrees/minutes to decimal degrees
    coord_cols = ['lat_begin', 'lon_begin', 'lat_end', 'lon_end']
    df[coord_cols] = df[coord_cols].apply(pd.to_numeric, errors='coerce') / 60

    # Optimize data types for memory efficiency
    df = optimize_data_types(df)

    # Drop rows with invalid datetime values
    df = df.dropna(subset=['datetime_begin', 'datetime_end'])

    return df

# Function to process each .7z file
def process_7z_file(archive_path):
    aggregated_list = []

    # Use a temporary directory to extract the files
    with tempfile.TemporaryDirectory() as temp_dir:
        with py7zr.SevenZipFile(archive_path, mode='r') as archive:
            archive.extractall(path=temp_dir)

            # Process each extracted file
            for extracted_file in archive.getnames():
                extracted_file_path = os.path.join(temp_dir, extracted_file)

                # Read the SO6 file and process it
                so6_df = read_so6_file(extracted_file_path)

                # Aggregate the dataframe as per flight
                aggregated_df = so6_df.groupby('flight_identifier').agg({
                    'callsign': 'first',  # Take the first callsign
                    'flight_origin': 'first',  # Take the first origin
                    'flight_destination': 'first',  # Take the first destination
                    'aircraft_type': 'first',  # Take the first aircraft type
                    'datetime_begin': 'min',  # Get the first timestamp (minimum)
                    'datetime_end': 'max',  # Get the last timestamp (maximum)
                }).reset_index()

                # Append the aggregated DataFrame to the list
                aggregated_list.append(aggregated_df)
                del aggregated_df
                gc.collect()

    # Return concatenated DataFrame for the current .7z archive
    if aggregated_list:
        return pd.concat(aggregated_list, ignore_index=True)
    return pd.DataFrame()


def process_ddr(archive): 
    try:
        final_aggregated_df = pd.read_csv(archive)
    except: 
        # Initialize an empty list for aggregation
        final_aggregated_list = []

        # Loop through the folder containing .7z files
        archive_folder = r"C:\Users\iLabs_6\Documents\Tex\T-DDR"
        archive_folder = r"C:\Users\iLabs_6\Documents\Tex\realtimetest\realtimetestDDR"
        flist = [f for f in os.listdir(archive_folder) if f.endswith('.7z')]

        for file in tqdm(flist, desc="Processing .7z files"):
            archive_path = os.path.join(archive_folder, file)

            # Process the .7z file and get the aggregated DataFrame
            aggregated_df = process_7z_file(archive_path)

            # Append the aggregated DataFrame to the final list
            if not aggregated_df.empty:
                final_aggregated_list.append(aggregated_df)

        # Concatenate all final aggregated DataFrames at once
        if final_aggregated_list:
            final_aggregated_df = pd.concat(final_aggregated_list, ignore_index=True)
        else:
            final_aggregated_df = pd.DataFrame()

        # Print the final aggregated dataframe to verify the output
        print(final_aggregated_df.info(memory_usage='deep'))
        print(final_aggregated_df.head())

        # Optionally save the combined aggregated dataframe to a CSV
        final_aggregated_df.to_csv(archive, index=False)


    final_aggregated_df.rename(columns={'datetime_begin': 'FiledOBT'}, inplace=True)
    final_aggregated_df.rename(columns={'datetime_end': 'FiledAT'}, inplace=True)
    final_aggregated_df.rename(columns={'flight_origin': 'ADEP'}, inplace=True)
    final_aggregated_df.rename(columns={'flight_destination': 'ADES'}, inplace=True)
    capacity_df = pd.DataFrame()


    for ap in airports_top50:

        calcdf = capacity_calc(final_aggregated_df, ap, airport_capacity=airport_dict[ap]['capacity'])
        capacity_df =  pd.concat([capacity_df, calcdf])
    return capacity_df


def capacity_calc(P: pd.DataFrame, airport: str = "EGLL", airport_capacity: int = 88):
    """Calculating the airport's capacity used per 15 minutes, based on the number of arrivals and departures.

    Args:
        P (pd.DataFrame): Dataframe with all flight data.
        airport (str, optional): Airport for which capacity will be calculated. Defaults to "EGLL".
        airport_capacity (int, optional): Capacity of the airport per hour (maximum take-offs/landings). Defaults to 88.

    Returns:
        pd.DataFrame: Dataframe with columns 'Time' and 'capacity' for the specified airport.
    """
    print(f"Calculating capacity for airport: {airport}")
    print(f'aip cap = {airport_capacity}')
    # Convert FiledOBT and FiledAT to datetime
    dform = "%Y-%m-%d %H:%M:%S"
    P = P.assign(FiledOBT=lambda x: pd.to_datetime(x.FiledOBT, format=dform))
    P = P.assign(FiledAT=lambda x: pd.to_datetime(x.FiledAT, format=dform))

    # Filter for flights either departing from or arriving at the specified airport
    dep = P.query("ADEP == @airport")
    des = P.query("ADES == @airport")

    # Generate a full range of 15-minute intervals for the entire month (assuming P covers the full month)
    start_time = min(P['FiledOBT'].min(), P['FiledAT'].min()).replace(minute=0, second=0, microsecond=0)
    end_time = max(P['FiledOBT'].max(), P['FiledAT'].max()).replace(minute=45, second=59, microsecond=999999)
    full_time_range = pd.date_range(start=start_time, end=end_time, freq='15T')

    # Assign times in 15-minute intervals to both departures and arrivals
    dep = dep.assign(Time=lambda x: x.FiledOBT.dt.floor('15T'))
    des = des.assign(Time=lambda x: x.FiledAT.dt.floor('15T'))

    # Combine departure and arrival data
    dep['Type'] = 'Departure'
    des['Type'] = 'Arrival'
    new_df = pd.concat([dep, des], axis=0)

    # Create a DataFrame for the full time range
    all_times_df = pd.DataFrame(full_time_range, columns=['Time'])

    # Group the flights by each 15-minute time interval to count the number of flights
    flight_counts = new_df.groupby('Time').size().reindex(full_time_range, fill_value=0)

    # Merge the flight counts into the full time range DataFrame
    all_times_df = all_times_df.merge(flight_counts.rename('flight_count'), left_on='Time', right_index=True, how='left').fillna(0)

    # Calculate capacity used at each time interval
    all_times_df['capacity'] = all_times_df['flight_count'] / airport_capacity / 4  # Divided by 4 for 15-min intervals
    all_times_df['airport'] = airport

    return all_times_df[['Time', 'airport', 'capacity']].sort_values('Time')


