import pandas as pd
import py7zr
# import chardet
import airportsdata
airports = airportsdata.load()
import math
import time
# import chardet
import os
import datetime
import pickle
from flights import Flight, Flights
from datetime import datetime, timedelta
import tqdm as tqdm


def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0  # Earth radius in kilometers
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = (math.sin(dlat / 2))**2 + math.cos(lat1) * math.cos(lat2) * (math.sin(dlon / 2))**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c
    return distance


def unzipper(path, destination_dir, new_filenames=None):
    """
    Extracts the contents of a 7z archive to a specified directory and renames the extracted files.

    :param path: Path to the 7z file.
    :param destination_dir: Directory where the contents should be extracted.
    :param new_filenames: Optional dictionary mapping original filenames to new filenames.
    """
    print(f'zippath = {path}')
    
    # Ensure the destination directory exists
    os.makedirs(destination_dir, exist_ok=True)
    
    with py7zr.SevenZipFile(path, mode='r') as archive:
        all_files = archive.getnames()  # Get the list of file names in the archive
        archive.extractall(path=destination_dir)  # Extract all files to the destination directory

    if new_filenames:
        for old_name, new_name in new_filenames.items():
            old_path = os.path.join(destination_dir, old_name)
            new_path = os.path.join(destination_dir, new_name)
            if os.path.exists(old_path):
                os.rename(old_path, new_path)
                print(f'Renamed {old_path} to {new_path}')
            else:
                print(f'File {old_path} does not exist and cannot be renamed.')


def unpacker(archive_path):
    res = []
    mainfolders = os.listdir(archive_path)
    print(mainfolders)
    output = r'C:\Users\LVNL_iLAB1\Documents\Students\Tex Ruskamp\Unpacked'


    print(archive_path)
    # mainfolders.remove('.DS_Store')
    for folder in mainfolders:
        zipfiles = os.listdir(f'{archive_path}/{folder}')
        for file in zipfiles:

            zippath = f'{archive_path}/{folder}/{file}'
            print(zippath)
            unzipper(zippath, output)
    return res


def dateconvert(date):
    if date == '00' or date == ['00'] or date == '':
        return None
    if isinstance(date, list):
        return [datetime.strptime(x, '%y%m%d%H%M%S') if x != '00' else None for x in date]
    try:
        return datetime.strptime(date, '%y%m%d%H%M%S')
    except:
        return None

def efd(archive_path, flight_manager, chunk):
    
    mess = 0
    we = 0
    fpid = []
    file_list = []
    print(archive_path, 'archive path')
    for folder in os.listdir(archive_path):
        folder_path = os.path.join(archive_path, folder)
        if not any(c in folder for c in chunk):
            continue
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):

                file_path = os.path.join(folder_path, file)
                file_list.append(file_path)
        # print(file)
    
    skipline = ['PRF ']  # lines to not read
    print(file_list)

    for filename in tqdm.tqdm(file_list):
        with open(f'{filename}', 'r', encoding='iso-8859-1') as file:
            messages = file.read().split('\n\n')
            # print(f'mess = {messages}')

        for message in messages:
            mess += 1
            lines = message.split('\n')
            data = {}
            points = {}
            skipline2 = []
            if "EHAM" in message:
                for line in lines:
                    # print(line)
                    if line[1:5] in skipline + skipline2:
                        continue
                    parts = line.split(' ', 1)  # Split each line into key and value
                    if len(parts) == 2:
                        key, value = parts
                        if key in data.keys():
                            data[key] += value
                        else:
                            data[key] = value.strip()  # Remove leading/trailing spaces
                    
                    if 'ADID' in line:
                        adep = data.get('-ADEP', '')
                        adid = data.get('', '').split('-')
                        for i in range(len(adid)):
                            if adep in adid[i] and 'ADID' in adid[i]:
                                if len(adid) > i+1:
                                    eto_dep = adid[i+1].strip().split(' ')[-1]
                                    takeoff = dateconvert(eto_dep)

                    if 'EHAACBAS' in line:
                        cbasentry = line.split(' ')[5]

                    if 'REGUL' not in line and len(parts) > 1 and parts[0] == '' and '-ASP' not in parts[1]:
                        point = parts[1][5:].split(' ')
                        # print('point: ', point)
                        if len(point) > 3 and point[0] == '-PTID':
                            try:
                                alt = int(point[3][1:])

                            except:
                                continue

                            if alt <= 300:
                                point_time = dateconvert(point[5])
                                ptid = point[1]
                                if point_time not in ['', None]:
                                    points[ptid] = [point_time - takeoff, alt]
                            if alt >= 300 and len(point[3][1:]) < 5:
                                skipline2.append('-PT ')
                                skipline2.append( '-VEC')
                        elif point[0] =='-RELDIST':

                            try:
                                alt = int(point[3][1:])

                            except:
                                continue

                            if alt > 50 and alt <= 320:
                                point_time = dateconvert(point[5])
                                ptid = point[1]
                                if point_time not in ['', None]:
                                    points[ptid] = [point_time - takeoff, alt]
                            if alt >= 300 and len(point[3][1:]) < 5:
                                skipline2.append('-PT ')
                                skipline2.append( '-VEC')
                
                    # Extract relevant fields
                timestamp = data.get('-TIMESTAMP', '')
                ifplid = data.get('-IFPLID', '')
                eobt = data.get('-EOBD', '') + data.get('-EOBT', '') +'00'
                eta = data.get('-EDA', '') + data.get('-ETA', '') +'00'
                cdmstatus = data.get('-CDMSTATUS', '')
                taxitime = data.get('-TAXITIME', '')
                adep = data.get('-ADEP', '')
                ades = data.get('-ADES', '')
                arcid = data.get('-ARCID', '')
                event = data.get('-EVENT', '')  # Add Event
                fltstate = data.get('-FLTSTATE', '')  # Add Flight State
                depaptype = data.get('-DEPAPTYPE', 'STANDARD')  # Add Departure AP Type
                reg = data.get('-REG', '')  # Add Registration
                dep_status = data.get('-DEPSTATUS')
                aobt = data.get('-AOBD', '') + data.get('-AOBT', '') +'00'
                ata = data.get('-ADA', '') + data.get('-ATA', '') +'00'
                adid = data.get('', '').split('-')
                modeltype = data.get('-MODELTYP')
                atfmdelay = data.get('-ATFMDELAY')
                cta = data.get('-CDA', '') + data.get('-CTA', '') + '00'
                cobt = data.get('-COBD', '') + data.get('-COBT', '') + '00'
                flighttype = data.get('-FLTTYP', '')
                aircrafttype = data.get('-ARCTYP', '')
                operator = data.get('-AOOPR')
                tsat = data.get('-EOBD', '') + data.get('-TSAT', '')
                tobt = data.get('-EOBD', '') + data.get('-TOBT', '')

                
                # try:
                if event == 'CPR':


                    potATOT = data.get('', '').split(' ')[5]
                else:
                    potATOT = None

                try:
                    dist = haversine(airports[adep]['lat'], airports[adep]['lon'], airports[ades]['lat'], airports[ades]['lon'])
                except:
                    dist = 9999999
                # print(adep, ades, " one of these airports is not known")


                try:
                    dist = haversine(airports[adep]['lat'], airports[adep]['lon'], airports[ades]['lat'], airports[ades]['lon'])
                except:
                    dist = 9999999
                # print(adep, ades, " one of these airports is not known")

                    # print(adep, ades, " one of these airports is not known")

                #only store the departure points if the distance is shorter than 2500 km


                eto_dep = ''
                eto_landing = ''
                sid = ''
                star = ''

                # retreive sid, star, eto over runway both landing and taking off
                for i in range(len(adid)):
                    if adep in adid[i] and 'ADID' in adid[i]:
                        if len(adid) > i+1:
                            eto_dep = adid[i+1].strip().split(' ')[-1]
                        else:
                            print(adid)
                        if len(adid) > i + 2:
                            sid = adid[i+2]
                    elif ades in adid[i] and 'ADID' in adid[i]:
                        eto_landing = adid[i+1].strip().split(' ')[-1]
                        if len(adid) > i + 2:
                            star = adid[i+2]
                            # star is not here usually, often before in the route points, figure out a way of implementing
                    
                
                if fltstate in ['FI','FS','SI'] and dateconvert(eto_landing) and dateconvert(eto_dep) and abs(dateconvert(eto_landing) - dateconvert(eto_dep)) <timedelta(hours=2):
                    filed_departure = points
                    initial_filed_departure = points
                    flown_departure = None
                elif event == 'CPR' and dateconvert(eto_landing) and dateconvert(eto_dep) and abs(dateconvert(eto_landing) - dateconvert(eto_dep)) <timedelta(hours=3):
                    flown_departure = points
                    filed_departure = None
                    initial_filed_departure = None
                else:
                    flown_departure = None
                    filed_departure = None
                    initial_filed_departure = None

                #if fligth plan already known: append to entry

                # Create a dictionary for each IFPLID
                ifplid_dict = {
                    # 'ifplid': ifplid,

                    'timestamp': timestamp,
                    'EOBT': eobt,
                    'AOBT': aobt,
                    'ETA': eta,
                    'ATA': ata,
                    'ETO_DEP': eto_dep,
                    'ETO_LANDING': eto_landing,
                    'ATOT': potATOT,
                    'COBT': cobt,
                    'CTA': cta,
                    'CDMStatus': cdmstatus,
                    'taxitime': taxitime,
                    'depaptype': depaptype,
                    'Adep': adep,
                    'Ades': ades,
                    'Arcid': arcid,
                    'event': event,
                    'fltstate': fltstate,
                    'reg': reg,
                    'dep_status': dep_status,
                    'dist': dist,
                    #newvalues
                    'atfmdelay': atfmdelay,
                    'modeltype': modeltype,
                    'star': star,
                    'sid': sid,
                    'flighttype': flighttype,
                    'aircrafttype': aircrafttype,
                    'filed_departure': filed_departure,
                    'initial_filed_departure':initial_filed_departure,
                    'flown_departure': flown_departure,
                    'operator': operator,
                    'cbasentry': cbasentry,
                    'TOBT': tobt,
                    'TSAT': tsat
                }
                # if ifplid_dict['Adep'] == 'LFPG' and ifplid_dict['Ades'] == 'EHAM':
                try:

                    # if ifplid_dict['flighttype']!= 'N':
                    flight = flight_manager.get_or_create(ifplid, **ifplid_dict)
                    flight.update_data(**ifplid_dict)
                except ValueError:
                    we+= 1
                    continue
                # res.append(flight)
                    # df = pd.concat([df, pd.DataFrame([ifplid_dict]).set_index('IFPLID')], sort=False)
                fpid.append(ifplid)
        #     break
        # break
    
    # df.insert(1, 'MESSAGES',df['Timestamp'].str.len())
    print(f'messages read = {mess}')
    print(f'messages with error = {we}')

    print(f'unique flightplan ids = {len(set(fpid))}, {len(fpid)}')
    return flight_manager