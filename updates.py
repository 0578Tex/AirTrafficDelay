from constants import *
from datetime import datetime, timedelta
import numpy as np
from tqdm import tqdm
import pandas as pd

def get_minimum_turnaround_time(type):
    if type in public_airliners_turnaround_times.keys():
        return timedelta(minutes=public_airliners_turnaround_times[type])
    return timedelta(minutes = 45)


def get_knock_on_delay(flight, current_time):
    """
    Calculates the knock-on delay from the previous flight using the same aircraft.
    """
    if flight.prev and not isinstance(flight.prev, float):
        prev_flight = flight.prev
        prev_flight.get_states()
        prev_flight.get_latest(current_time)
        
        # Get the minimum turnaround time
        minimum_turnaround = get_minimum_turnaround_time(flight.aircrafttype) #timedelta(minutes=45)
        # aircrafyt type specificeren

        # Get the latest known arrival time of the previous flight
        # Prefer actual arrival time, then estimated arrival time
        prev_ata = prev_flight.latest.get('ATA')
        prev_eta = prev_flight.latest.get('ETA')
        
        # Determine the best available arrival time
        arrival_time = prev_ata or prev_eta
        
        if arrival_time is None:
            # If we have no arrival time, check if the previous flight has departed
            prev_atot = prev_flight.actual.get('ATOT') or prev_flight.latest.get('ETOT')
            if prev_atot is None:
                # Previous flight hasn't departed yet; estimate duration
                estimated_duration = prev_flight.filed.get('ETO') - prev_flight.filed.get('EOBT')
                if estimated_duration is None:
                    # As a fallback, use average flight duration or zero
                    estimated_duration = timedelta(hours=1)  # Default to 1 hour
                arrival_time = prev_atot + estimated_duration
            else:
                # Previous flight hasn't even departed yet; cannot calculate knock-on delay
                return 0
            
        
        # Calculate the expected aircraft ready time
        expected_ready_time = arrival_time + minimum_turnaround
        
        # Get the scheduled departure time of the current flight
        scheduled_departure = flight.filed.get('EOBT')
        
        if scheduled_departure is None:
            return 0  # Cannot calculate without scheduled departure time
        
        # Calculate knock-on delay
        knock_on_delay_seconds = (expected_ready_time - scheduled_departure).total_seconds()
        knock_on_delay_minutes = max(0, knock_on_delay_seconds / 60)  # Ensure non-negative
        
        return knock_on_delay_minutes
    else: 
        # No previous flight; no knock-on delay
        return 0
    
def encode_visibility(taf):
    sorted_taf = sorted(taf, key=lambda x: x[0], reverse=True)
    ddistance, tdistance = [], []
    for t in sorted_taf:
        timestamp, mess = t
        if isinstance(mess, dict):
            if mess["visibility"]:
                distance = mess["visibility"].distance
                if '10km' in distance:
                    ddistance.append(10000)
                elif 'm' in distance:
                    ddistance.append(int(distance.replace('m', '')))
        else:
            if mess and mess.visibility and mess.visibility.distance:
                if '10km' in mess.visibility.distance:
                    tdistance.append(10000)
                else:
                    tdistance.append(int(mess.visibility.distance.replace('m', '')))
    if not ddistance:
        ddistance = [10000]
    if not tdistance:
        tdistance = [10000]
    return min(min(ddistance), min(tdistance)), min(min(ddistance), min(tdistance))

def encode_wind(taf):
    sorted_taf = sorted(taf, key=lambda x: x[0], reverse=True)
    sdict, ddict, gdict, strend, dtrend, gtrend = [], [], [], [], [], []
    for t in sorted_taf:
        timestamp, mess = t
        if mess is None:
            continue
        if isinstance(mess, dict):
            if mess['wind']:
                sdict.append(mess['wind'].speed)
                ddict.append(mess['wind'].degrees)
                gdict.append(mess['wind'].gust)
        elif mess.wind:
            strend.append(mess.wind.speed)
            dtrend.append(mess.wind.degrees)
            gtrend.append(mess.wind.gust)
                
    res = []
    for l in sdict, gdict, strend, gtrend:
        l = [x for x in l if x is not None]
        res.append(max(l) if l else None)
    res.insert(1, ddict[-1] if ddict else None)
    res.insert(4, dtrend[-1] if dtrend else None)
    return res

# Extract the dynamic (time-step specific) features for a given time step
def extract_features(flight, current_time, dt, prev):
    # flight.get_states()
    flight.get_latest(current_time)
    fEOBT = flight.filed['EOBT']
    lTSAT = flight.latest['TSAT'] if  flight.latest['TSAT'] else  flight.latest['EOBT']
    lTOBT = flight.latest['TOBT'] if  flight.latest['TOBT'] else  flight.latest['EOBT']
    # eobt_min = lEOBT.hour * 60 + lEOBT.minute if lEOBT else 0
    # eta_min = flight.latest['ETA'].hour * 60 + flight.latest['ETA'].minute if flight.latest['ETA'] else 0
    # updates = len([t for t in flight.timestamp if t > current_time])
    atfmdelay = int(flight.latest['atfmdelay']) /60 if flight.latest['atfmdelay'] else 0
    regulations = len(flight.regulations)
    cobt = flight.latest['COBT']
    cobt_delay = (cobt - fEOBT).total_seconds() /60 if fEOBT and cobt else 0
    eflighttime = (flight.latest['ETA'] - fEOBT).total_seconds()/60  if fEOBT and flight.latest['ETA'] else 0
    ko = get_knock_on_delay(flight, current_time)

    ddistance, tdistance = encode_visibility(flight.AdepTAF)
    sdict, ddict, gdict, strend, dtrend, gtrend = encode_wind(flight.AdepTAF)
    TSATdelay = (fEOBT - lTSAT).total_seconds() /60 if fEOBT and lTSAT else 0
    TOBTdelay = (fEOBT - lTOBT).total_seconds() /60 if fEOBT and lTOBT else 0

    dt = np.round(dt,1)

    f_etodep = flight.filed['ETO_DEP']
    l_etodep = flight.latest['ETO_DEP']
    etodepdelay = (l_etodep- f_etodep).total_seconds() /60 if f_etodep and l_etodep else 0

    cdmstatus = flight.latest['CDMStatus']
    # atfm_diff = 0
    # if prev:
    #     atfm_diff = flight.latest['atfmdelay'] - prev['atfmdelay']
    timetocbas = flight.actual['cbasentry'] - current_time

    offblock = 0 if flight.latest['CDMStatus'] == 'ACTUALLOFFBLOCK' else 1
    dt = int(dt)
    if flight.latest['ETO_DEP']:
        t_to_atot = (flight.latest['ETO_DEP']- current_time).total_seconds()/60
    else:
        t_to_atot = 300

    return {
        # f'ko_{dt}': ko,
        f't_to_atot_Tmin_{dt}': t_to_atot,
        f'atfmdelay_Tmin_{dt}': atfmdelay,
        f'regulations_Tmin_{dt}': regulations,
        f'cobt_delay_Tmin_{dt}': cobt_delay,
        f'eflighttime_Tmin_{dt}': eflighttime,
        f'visibility_Tmin_{dt}': tdistance,
        f'ko_Tmin_{dt}': ko,
        f'TSATdelay_Tmin_{dt}': TSATdelay,
        f'TOBTdelay_Tmin_{dt}': TOBTdelay,
        f'etodepdelay_Tmin_{dt}': etodepdelay,
        f'cdmstatus_Tmin_{dt}': cdmstatus,
        f'timetoCBAS_Tmin_{dt}': timetocbas,
        f'offblock_Tmin_{dt}': offblock,
        f'wspeed_Tmin_{dt}': max([sdict, strend]) if sdict is not None and strend is not None else (sdict or strend or 0),
        f'wdirec_Tmin_{dt}': max([ddict, dtrend]) if ddict is not None and dtrend is not None else (ddict or dtrend or 0),
        f'wguts_Tmin_{dt}': max([gdict, gtrend]) if gdict is not None and gtrend is not None else (gdict or gtrend or 0),


        f'fltstate_SI_Tmin_{dt}': 1 if flight.latest['fltstate'] == 'SI' else 0,
        f'fltstate_FI_Tmin_{dt}':  1 if flight.latest['fltstate'] == 'FI' else 0,
        f'fltstate_other_Tmin_{dt}':  1 if flight.latest['fltstate'] not in ['FI', 'SI'] else 0,

        f'modeltyp_ACT_Tmin_{dt}': 1 if flight.latest['modeltype'] =='ACT' else 0,
        f'modeltyp_CAL_Tmin_{dt}': 1 if flight.latest['modeltype'] =='CAL' else 0,
        f'modeltyp_EST_Tmin_{dt}': 1 if flight.latest['modeltype'] =='EST' else 0,
    }

def extend_with_horizon_updates(flight, start_time, dt=timedelta(minutes=30), cap=None, end_time=None, mode='None'):
    # Fixed features are extracted once
    # flight.get_states()
    fixed_features = flight.filed
    # fixed_features['ADEP'] = flight.ADEP
    del fixed_features['COBT']
    del fixed_features['CTA']
    del fixed_features['ETO_DEP']
    del fixed_features['ETO_LANDING']
    reobt = round_to_nearest_15_minutes(fixed_features['EOBT'])
    fixed_features['cap_DEP'] = match_capacity(reobt, fixed_features['ADEP'], cap = cap)
    reta = round_to_nearest_15_minutes(fixed_features['ETA'])
    fixed_features['cap_DES'] = match_capacity(reta, fixed_features['ADES'], cap=cap)
    
    fixed_features['delay'] = flight.delay

    fixed_features['actype'] = flight.aircrafttype
    fixed_features['flighttype'] = flight.flighttype
    # del fixed_features['modeltype']

    dynamic_features_over_time = fixed_features  # Collect dynamic features for each time step

    current_time = start_time
    # print(flight)
    prev = None
    while current_time <= end_time:
        flight.get_latest(current_time)
        if mode == 'atot':
            dt_minutes = (current_time - flight.actual['ETO_DEP']).total_seconds() / 60

        if mode == 'etot':
            dt_minutes = (current_time - flight.filed['ETO_DEP']).total_seconds() / 60


        timestep_features = extract_features(flight, current_time, dt_minutes, prev)
        prev = timestep_features
        dynamic_features_over_time.update(timestep_features)  # Append dynamic features
        current_time += dt

    return  dynamic_features_over_time

def round_to_nearest_15_minutes(dt: datetime) -> datetime:
    discard = timedelta(minutes=dt.minute % 15, seconds=dt.second, microseconds=dt.microsecond)
    dt_rounded = dt - discard
    if discard >= timedelta(minutes=7.5):
        dt_rounded += timedelta(minutes=15)
    return dt_rounded

def match_capacity(rounded_dt, ADEP, cap=None) -> float:

    # Filter the capacity dataframe for the matching ADEP, ADES, and rounded time
    match = cap[(cap['airport'] == ADEP) & 
                        (cap['Time'] == rounded_dt)]
    
    # If a match is found, return the capacity value, otherwise return 0 (or some default value)
    if not match.empty:
        return match['capacity'].values[0]  # Extract the capacity value
    else:
        return 0.0  # Default capacity if no match found


# Prepare the dataset with updates (one row per flight, with updates)
def prepare_dataset_with_updates(ef, cap=None, horizon_hours=5, dt_minutes=1, taf_ar=None, save_ar=None, mode=None):
    all_features = []
    dt = timedelta(minutes=dt_minutes)      
    
    # Loop through each flight in the dictionary
    for key, flight in tqdm(ef.items()):

        adep, ades, eobt = key  # Unpack flight key

   
        flight.update_taf(taf_ar)
        
        flight.get_states()
        if not flight.filed['ETO_DEP'] or not flight.filed['ETA']:
            continue
        if not flight.actual['ETO_DEP'] :
            continue
        # start_time = eobt - timedelta(hours=horizon_hours)
        if mode == 'atot':
            # print(f'{mode=}')
            start_time = flight.actual['ETO_DEP'] - timedelta(hours=horizon_hours)

        if mode == 'etot':
            start_time = flight.filed['ETO_DEP'] - timedelta(hours=horizon_hours)

        # Get features for each flight (fixed + dynamic updates)
        flight_features = extend_with_horizon_updates(flight, start_time, dt, cap=cap, end_time=flight.actual['ETO_DEP'], mode=mode)

        all_features.append(flight_features) # Each flight's features are a single row
        # print(all_features)
        # break
    
    # Convert to DataFrame
    extended_df = pd.DataFrame(all_features)
    if save_ar:
        extended_df.to_csv(save_ar, index=False)

    return extended_df