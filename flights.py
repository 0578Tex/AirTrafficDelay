"""
The Flight classes
"""
from datetime import datetime, timedelta


def round_to_nearest_hour(dt):
    # Adds 30 minutes to 'dt' then truncates the minutes and seconds
    if dt.minute < 30:
        return dt.replace(minute=0, second=0, microsecond=0)
    # If the minutes are 30 or more, add one hour and then truncate to the hour
    else:
        return (dt + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)


class Flights:
    def __init__(self):
        self.flights = {}
    
    def get_or_create(self, ifplid, **kwargs):

        if ifplid not in self.flights:
            self.flights[ifplid] = Flight(ifplid, **kwargs)

        return self.flights[ifplid]

    def __str__(self):
        return "\n".join(str(flight) for flight in self.flights.values())


class Flight:

    def __init__(self, ifplid, timestamp, EOBT, AOBT, ETA, ATA, ETO_DEP, ETO_LANDING, COBT, CTA, ATOT, 
                 CDMStatus, taxitime, depaptype, Adep, Ades, Arcid, event, fltstate, reg, dep_status, 
                 dist, atfmdelay, modeltype, star, sid, flighttype, aircrafttype,filed_departure,initial_filed_departure, flown_departure, operator, cbasentry, TSAT, TOBT):
        self.ifplid = ifplid
        self.operator = operator
        self.timestamp = []
        self.EOBT = {}
        self.AOBT = {}
        self.ETA = {}
        self.ATA = {}
        self.ETO_DEP = {}
        self.ETO_LANDING = {}
        self.COBT = {}
        self.CTA = {}
        self.ATOT = {}
        self.CDMStatus = {}
        self.taxitime = {}
        self.depaptype = depaptype
        self.Adep = Adep
        self.AdepMETAR = {}
        self.AdepTAF = {}
        self.AdepMETAR = {}
        self.AdepTAF = {}
        self.Ades = Ades
        self.AdesMETAR = {}
        self.AdesTAF = {}
        self.AdesMETAR = {}
        self.AdesTAF = {}
        self.Arcid = Arcid
        self.event = {}
        self.fltstate = {}
        self.regulations = {}
        self.prev = {}
        self.dep_status = {}
        self.event = {}
        self.fltstate = {}
        self.regulations = {}
        self.prev = {}
        self.dep_status = {}
        self.dist = dist
        self.atfmdelay = {}
        self.modeltype = {}
        self.cbasentry = {}
        self.TSAT = {}
        self.TOBT = {}
        self.reg = reg
        self.star = star
        self.sid = sid
        self.flighttype = flighttype
        self.aircrafttype = aircrafttype
        self.filed_departure = filed_departure
        self.initial_filed_departure = initial_filed_departure
        self.flown_departure = flown_departure
        self.get_filed()


    def get_first(self, attribute):
        """
        Helper method to return the first non-None item if the attribute is a dictionary.
        If the attribute is not a dictionary, return the attribute itself.
        """

        if isinstance(attribute, dict):
            # Sort the dictionary by keys and iterate over values
            sorted_attribute = dict(sorted(attribute.items()))
            for item in sorted_attribute.values():
                if item is not None:
                    return item
            return None  # All items are None
        else:
            return attribute  # Return the non-dict attribute directly

    def get_last(self, attribute):
        """
        Helper method to return the last non-None item if the attribute is a dictionary.
        If the attribute is not a dictionary, return the attribute itself.
        """
        if isinstance(attribute, dict):
            # Sort the dictionary in reverse order and iterate over values
            sorted_attribute = dict(sorted(attribute.items(), reverse=True))
            
            for item in sorted_attribute.values():
                if item is not None:
                    return item
            return None  # All items are None
        else:
            return attribute  # Return the non-dict attribute directly


    def get_states(self):
        self.get_filed()
        self.get_actual()


    def get_filed(self):
        """
        Gathers the planned (filed) information about the flight, typically available before the flight departs.
        """
        self.filed = {
            'distance': self.get_first(self.dist),
            'ETOT': self.get_first(self.ETOT),

            'ADEP': self.get_first(self.Adep),
            'ADES': self.get_first(self.Ades),
            'EOBT': self.get_first(self.EOBT),            # Estimated Off Block Time
            'ETA': self.get_first(self.ETA),              # Estimated Time of Arrival
            'ETO_DEP': self.get_first(self.ETO_DEP),      # Estimated Take Off Time (Departure)
            'ETO_LANDING': self.get_first(self.ETO_LANDING),  # Estimated Time of Landing
            'COBT': self.get_first(self.COBT),            # Calculated Off Block Time
            'CTA': self.get_first(self.CTA),              # Calculated Time of Arrival
            'CDMStatus': self.get_first(self.CDMStatus),  # Status from Collaborative Decision Making
            'taxitime': self.get_first(self.taxitime),    # Estimated taxi time
            'event': self.get_first(self.event),          # Flight event (e.g., takeoff, landing)
            'regulations': self.get_first(self.regulations),  # Any regulations affecting the flight
            'dep_status': self.get_first(self.dep_status),  # Departure status
            'atfmdelay': self.get_first(self.atfmdelay),  # Air Traffic Flow Management delay
            'modeltype': self.get_first(self.modeltype),  # Model type of the aircraft
            'star': self.get_first(self.star),            # Standard Terminal Arrival Route (STAR)
            'sid': self.get_first(self.sid),              # Standard Instrument Departure (SID)
            'cbasentry': self.get_first(self.cbasentry),
            'TSAT': self.get_first(self.TSAT),
            'TOBT': self.get_first(self.TOBT),
            'fltstate': self.get_first(self.fltstate)
        }
        return self.filed
    

    def get_actual(self):
        """
        Gathers the actual flight information, typically available after the flight has departed or arrived.
        """
        self.actual = {
            'AOBT': self.get_last(self.AOBT),             # Actual Off Block Time
            'ATOT': self.get_last(self.ATOT),             # Actual Off Block Time
            'ATA': self.get_last(self.ATA),               # Actual Time of Arrival
            'ETO_DEP': self.get_last(self.ETO_DEP),       # Actual Estimated Take Off Time (after departure)
            'ETO_LANDING': self.get_last(self.ETO_LANDING), # Actual Estimated Landing Time
            'COBT': self.get_last(self.COBT),             # Calculated Off Block Time (actual)
            'CTA': self.get_last(self.CTA),               # Calculated Time of Arrival (actual)
            'CDMStatus': self.get_last(self.CDMStatus),   # Status from Collaborative Decision Making (actual)
            'taxitime': self.get_last(self.taxitime),     # Actual taxi time
            'event': self.get_last(self.event),           # Actual flight event (e.g., takeoff, landing)
            'regulations': self.get_last(self.regulations), # Regulations affecting the flight (actual)
            'dep_status': self.get_last(self.dep_status), # Departure status (actual)
            'atfmdelay': self.get_last(self.atfmdelay),   # Air Traffic Flow Management delay (actual)
            'modeltype': self.get_last(self.modeltype),   # Aircraft model type
            'star': self.get_last(self.star),             # Standard Terminal Arrival Route (actual)
            'sid': self.get_last(self.sid),               # Standard Instrument Departure (actual)
            'cbasentry': self.get_last(self.cbasentry),
            'TSAT': self.get_last(self.TSAT),
            'TOBT': self.get_last(self.TOBT),
            'fltstate': self.get_last(self.fltstate)
        }
        return self.actual
    
    def get_latest(self, timestamp):
        """
        Retrieves the latest available values for the flight at or before the given timestamp.
        """

        def find_latest(data_dict, timestamp):
            if isinstance(data_dict, dict) and data_dict:
                valid_times = [time for time in data_dict.keys() if time <= timestamp]
                if valid_times:
                    latest_time = max(valid_times)
                    return data_dict[latest_time]
            return None

        self.latest = {
            'EOBT': find_latest(self.EOBT, timestamp),
            'AOBT': find_latest(self.AOBT, timestamp),
            'ETA': find_latest(self.ETA, timestamp),
            'ATA': find_latest(self.ATA, timestamp),
            'ETO_DEP': find_latest(self.ETO_DEP, timestamp),
            'ETO_LANDING': find_latest(self.ETO_LANDING, timestamp),
            'COBT': find_latest(self.COBT, timestamp),
            'CTA': find_latest(self.CTA, timestamp),
            'CDMStatus': find_latest(self.CDMStatus, timestamp),
            'taxitime': find_latest(self.taxitime, timestamp),
            'event': find_latest(self.event, timestamp),
            'regulations': find_latest(self.regulations, timestamp),
            'dep_status': find_latest(self.dep_status, timestamp),
            'atfmdelay': find_latest(self.atfmdelay, timestamp),
            'modeltype': find_latest(self.modeltype, timestamp),
            'star': find_latest(self.star, timestamp),
            'sid': find_latest(self.sid, timestamp),
            'cbasentry': find_latest(self.cbasentry, timestamp),
            'TSAT': find_latest(self.TSAT, timestamp),
            'TOBT': find_latest(self.TOBT, timestamp),
            'fltstate': find_latest(self.fltstate, timestamp),
            'atfmdelay': find_latest(self.atfmdelay, timestamp)
        }
        
        return self.latest

    @classmethod
    def get_or_create(self, ifplid, **kwargs):
        if ifplid not in self.flights:
            self.flights[ifplid] = Flight(ifplid, **kwargs)
        else:
            # Ensure existing flight is updated with new data
            self.flights[ifplid].update_data(**kwargs)
        return self.flights[ifplid]


    def update_data(self, **kwargs):
        t = self.dateconvert(kwargs['timestamp'])  # Convert timestamp
        
        for key, value in kwargs.items():
            if hasattr(self, key) and isinstance(getattr(self, key), dict):  # Check if the field is a dict
                # Convert the value if it's a time-related field

                if key in ['EOBT', 'AOBT', 'ETA', 'ATA', 'ETO_DEP', 'ETO_LANDING', 'COBT', 'CTA', 'cbasentry', 'TSAT', 'TOBT']:
                    value = self.dateconvert(value) if value else None

                if key == 'ATOT':  # Special handling for ATOT
                    value = self.dateconvert(value) if value else None

                if value == None:
                    continue

                # Insert into dictionary using timestamp 't'
                if t not in getattr(self, key):  # Ensure no duplicate timestamps
                    getattr(self, key)[t] = value
                elif getattr(self, key)[t] != value:  # Update value if it's different
                    getattr(self, key)[t] = value

            elif key == 'timestamp':  # Handle timestamp (still a list)
                if t not in getattr(self, key):  # Ensure no duplicate timestamps
                    getattr(self, key).append(t)

            elif key in ['flown_departure', 'filed_departure', 'initial_filed_departure']:
                if value not in ['', None] and getattr(self, key) != value:
                    setattr(self, key, value)

            elif hasattr(self, key):  # Handle non-dict fields
                if getattr(self, key) != value:
                    setattr(self, key, value)


            else:
                print(f"Warning: {key} is not a valid data field")

    @property
    def ETOT(self):
        if self.taxitime and self.EOBT:
            return {t: self.EOBT[t] +timedelta(minutes = int(self.get_latest(t)['taxitime']) if self.get_latest(t)['taxitime'] is not None else 0) 
                    for t in self.EOBT if self.get_latest(t)['taxitime']}
        return {}

    
    @property
    def delay(self):
        if self.ATOT and self.ETOT:
            self.get_states()

            if self.filed['ETO_DEP'] and self.actual['ETO_DEP']:
                return (self.actual['ETO_DEP'] - self.filed['ETO_DEP']).total_seconds()/60
        return None

    @property
    def flight_time(self):
        if self.ETOT and self.ETA:
            self.get_states()

            if self.filed['ETA'] and self.filed['ETOT']:
                return (self.filed['ETOT'] - self.filed['ETO_DEP']).total_seconds()/60
        return None

    def merge(self, other_flight):
        if self.timestamp[0] > other_flight.timestamp[-1]:
            # Self is later than the other flight, so add self to the end
            second = self
            self = other_flight
        elif self.timestamp[-1] < other_flight.timestamp[0]:
            # Self is earlier than the other flight
            second = other_flight
        else:
            second = other_flight

        for key, value in second.__dict__.items():
            if hasattr(self, key) and isinstance(getattr(self, key), list):
                setattr(self, key, list(set(getattr(self, key) + value)))  # Merge and remove duplicates
            elif hasattr(self, key) and isinstance(getattr(self, key), dict):
                # Merge dictionaries and prioritize newer data
                for k, v in value.items():
                    if k not in getattr(self, key) or getattr(self, key)[k] < v:
                        getattr(self, key)[k] = v
            elif hasattr(self, key):
                if getattr(self, key) != value:
                    setattr(self, key, value)
            else:
                print(f"Warning: {key} is not a valid data field")


    def merge_flights(self, efd_flights, update_weather=False, metar_reports=[], taf_reports=[]):
        atotflights = []
        arcidlist = {}

        # Pre-process to map Arcid to flight indices and prepare initial state
        for flightnr, flight in efd_flights.flights.items():
            if flight and flight.timestamp:
                if flight.Arcid not in arcidlist:
                    arcidlist[flight.Arcid] = [flightnr, flight.timestamp[0], flight.timestamp[-1], flight.fltstate]
                else:
                    other_flight_info = arcidlist[flight.Arcid]
                    # Assuming there's logic to decide if a merge is needed
                    if self.should_merge(other_flight_info, flight):
                        efd_flights.flights[other_flight_info[0]].merge(flight)
                        efd_flights.flights[flightnr] = None  # Mark for removal or further processing
                        # Update the arcidlist with the new merged state
                        arcidlist[flight.Arcid] = [other_flight_info[0], efd_flights.flights[other_flight_info[0]].timestamp[0], efd_flights.flights[other_flight_info[0]].timestamp[-1], efd_flights.flights[other_flight_info[0]].fltstate]
                        print(f"Merged flight {other_flight_info[0]} and {flightnr}")
                    else:
                        # Update timestamp range without merging
                        start_time = min(other_flight_info[1], flight.timestamp[0])
                        end_time = max(other_flight_info[2], flight.timestamp[-1])
                        arcidlist[flight.Arcid] = [flightnr, start_time, end_time, flight.fltstate]

        # Optional: Update weather data more efficiently
        if update_weather:
            for flight in efd_flights.flights.values():
                if flight:
                    flight.update_weather_data(metar_reports, taf_reports)
                    # flight.add
        return efd_flights

    def should_merge(self,other_flight_info, flight):
        # Simplified logic to decide if flights should merge based on your conditions
        return abs(flight.timestamp[0] - other_flight_info[2]) < timedelta(hours=4) and other_flight_info[3] != 'TE'

    def merge(self, other_flight):
        if self.timestamp[0] > other_flight.timestamp[-1]:
        #self is later than other flight, add self to end
            second = self
            self = other_flight
        #check if second is now actually self
        elif self.timestamp[-1] < other_flight.timestamp[0]:
            #self is earlier than other flight

            second = other_flight
        else:
            # self and other flight are not before or after each other, stop merge.
            # return
            second = other_flight


        for key, value in second.__dict__.items():

            if hasattr(self, key) and isinstance(getattr(self, key), list):
                setattr(self, key, getattr(self, key)+ value)

            elif hasattr(self, key):
                # Set the value only if it's different
                if getattr(self, key) != value:
                    setattr(self, key, value)
            else:
                print(f"Warning: {key} is not a valid data field")

        del self

    def update_weather_data(self, metar_reports, taf_reports):
        """Updates the METAR and TAF data for the flight based on ETOT and ATA/ETA."""
        # self.update_metar(metar_reports)
        # self.update_metar(metar_reports)
        self.update_taf(taf_reports)

    
    def add_regulation(self, regulations):
        if self.Adep in regulations.keys():
            rlist = regulations[self.Adep]
            if not self.EOBT:
                return
            for regu in rlist:
                if regu.starttime <= self.filed['EOBT'] <= regu.endtime:
                    self.regulations.append(regu)
                    # print(f'reg added = {regu}')


    def update_metar(self, metar_reports):
        """Fetch and store METAR data for departure and arrival airports."""
        # Departure METAR
        if self.ETOT and self.ETOT[-1]:
            departure_time = round_to_nearest_hour(self.ETOT[-1]) if self.ETOT else None
            if departure_time and self.Adep in metar_reports and departure_time in metar_reports[self.Adep]:
                self.AdepMETAR = metar_reports[self.Adep][departure_time]

        # Arrival METAR
        eta_last = self.ETA[-1] if self.ETA else None
        ata_last = self.ATA[-1] if self.ATA else None
        cta_last = self.CTA[-1] if self.CTA else None

        # Determine the atime based on the conditions
        atime = eta_last if eta_last is not None else ata_last if ata_last is not None else cta_last

        if atime:
            arrival_time = round_to_nearest_hour(atime)
            if arrival_time and self.Ades in metar_reports and arrival_time in metar_reports[self.Ades]:
                self.AdesMETAR = metar_reports[self.Ades][arrival_time]

    def update_taf(self, taf_reports):
        """Fetch and store TAF data for departure and arrival airports."""
        # Departure TAF
        if self.ETO_DEP and self.get_last(self.ETO_DEP):
            # print(f'taffferttt = {self.ETO_DEP}')
            departure_time = round_to_nearest_hour(self.get_last(self.ETO_DEP))
            # print(f'deptime ={departure_time}')
            # print(f'taferporst = {taf_reports[self.Adep]}')
            if departure_time and self.Adep in taf_reports and departure_time in taf_reports[self.Adep]:
                self.AdepTAF = taf_reports[self.Adep][departure_time]

        # Arrival TAF
        eta_last = self.get_last(self.ETA)
        ata_last = self.get_last(self.ATA)
        cta_last = self.get_last(self.CTA)
        eta_last = self.get_last(self.ETA)
        ata_last = self.get_last(self.ATA)
        cta_last = self.get_last(self.CTA)

        # Determine the atime based on the conditions
        atime = eta_last if eta_last is not None else ata_last if ata_last is not None else cta_last
        
        if atime:
            arrival_time = round_to_nearest_hour(atime) 
            if arrival_time and self.Ades in taf_reports and arrival_time in taf_reports[self.Ades]:
                self.AdesTAF = taf_reports[self.Ades][arrival_time]


    def dateconvert(self, date):
        if date == '00' or date == ['00']:
            return None
        if isinstance(date, list):
            return [datetime.strptime(x, '%y%m%d%H%M%S') if x != '00' else None for x in date]
        return datetime.strptime(date, '%y%m%d%H%M%S')


    def __str__(self):

        return (f"Flight {self.ifplid}:\n"
                f"  From {self.Adep} to {self.Ades}\n"
                f"  ArcID: {self.Arcid}, Reg: {self.reg}\n"
                f"  ETOT: {len(self.ETOT)} {self.ETOT} \n"
                f"  ATOT: {len(self.ATOT)} {self.ATOT}"
                f"  delay: {self.delay} \n"

                f"  Departure Type: {self.depaptype}, Flight Type: {self.flighttype}, Aircraft: {self.aircrafttype}\n"
                f"  Departure Times - EOBT: {len(self.EOBT if self.EOBT else 0)if self.EOBT else 0} {self.EOBT}, AOBT: {len(self.AOBT)} {self.AOBT}, ETA: {self.ETA}, ATA: {self.ATA}\n"
                f"  Operational Times - ETO DEP: {self.ETO_DEP}, ETO Landing: {self.ETO_LANDING}, COBT: {self.COBT}, CTA: {self.CTA}\n"
                f"  Status - CDM: {self.CDMStatus}, Flight State: {self.fltstate}\n"
                f"  Distance: {self.dist}, ATFM Delay: {self.atfmdelay}, Taxitime: {self.taxitime}\n"
                f"  Navigation - STAR: {self.star}, SID: {self.sid}\n"
                f"  Model Type: {len(self.modeltype) if self.modeltype else 0} {self.modeltype}, Event: {self.event}\n"
                f"  Timestamps: {len(self.timestamp) if self.timestamp else 0} {self.timestamp}\n"
                )
    

