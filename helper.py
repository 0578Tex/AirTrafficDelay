import numpy as np

def extract_flight_details(f, period=1440):
    """
    Extract various flight details and return them as a dictionary.
    
    :param f: DataFrame containing flight details
    :param period: Time period for mapping angles (default is 1440 minutes for a day)
    :return: Dictionary with extracted flight details
    """
    # Compute time
    sin_component = f['sin_ETOT']
    cos_component = f['cos_ETOT']
    angle = np.arctan2(sin_component, cos_component)
    angle = np.where(angle < 0, angle + 2 * np.pi, angle)  # Adjust to [0, 2*pi)
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
    
    # Combine results into a dictionary
    result = {
        'time': time,
        'day': day,
        'adep': adep,
        'distance': distance,
        'longitude': longitude,
        'latitude': latitude
    }
    
    return result