import numpy as np

def circular_to_time(sin_component, cos_component, period=1440):
    """
    Convert circular (sine, cosine) representation back to time in minutes.
    
    :param sin_component: Sine component of the circular representation.
    :param cos_component: Cosine component of the circular representation.
    :param period: The number of minutes in a full cycle (default is 1440 minutes = 24 hours).
    :return: Time in minutes corresponding to the given sine and cosine components.
    """
    # Calculate the angle (theta) using arctangent, ensuring the correct quadrant
    theta = np.arctan2(sin_component, cos_component)
    
    # arctan2 returns values in the range [-pi, pi]. Map them to [0, 2*pi)
    theta = np.mod(theta, 2 * np.pi)
    
    # Convert the angle back to time in minutes
    time_in_minutes = (theta * period) / (2 * np.pi)
    
    return time_in_minutes

def extract_flight_details(f, period=1440):
    """
    Extract various flight details and return them as a dictionary.
    
    :param f: DataFrame containing flight details
    :param period: Time period for mapping angles (default is 1440 minutes for a day)
    :return: Dictionary with extracted flight details
    """

    ETOT = circular_to_time(f['sin_ETOT'][0], f['cos_ETOT'][0]) 
    
    ETA =  circular_to_time(f['sin_ETA'][0], f['cos_ETA'][0]) 
    # Extract day
    daycols = f[[col for col in f.columns if 'day' in col]]
    day = daycols.max().idxmax()
    
    # Extract ADEP
    adepdf = f[[col for col in f.columns if ("ADEP_" in col and col != "ADEP_capacity")]]
    adep = adepdf.max().idxmax()

    #ac type
    acf = f[[col for col in f.columns if ("actype" in col)]]
    actype = acf.max().idxmax()

    ft = f[[col for col in f.columns if ("flighttype" in col)]]
    flttype = ft.max().idxmax()

    flighttime = f['eflighttime_Tmin_-300'][0]
    # Extract distance
    distance = f['distance'][0]
    
    # Extract longitude
    longitude = f['ADEPLong'][0]
    
    # Extract latitude
    latitude = f['ADEPLat'][0]
    
    # Combine results into a dictionary
    result = {
        'ETOT': ETOT,
        'ETA': ETA,
        'day': day,
        'adep': adep,
        'distance': distance,
        'longitude': longitude,
        'latitude': latitude,
        'flighttime': flighttime,
        'actype': actype,
        'flighttype': flttype
    }
    
    return result