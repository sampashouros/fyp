import pandas as pd
import requests
from datetime import datetime, timedelta

#generate a list of timestamps for the past week
def generate_times(start_date, days=29, times_per_day=24):
    timestamps = []
    for i in range(days):
        day = (start_date + timedelta(days=i))
        for hour in [0, 8, 16]:  # Midnight, 8 AM, 4 PM
            timestamps.append(day.replace(hour=hour, minute=0, second=0, microsecond=0))
    return timestamps
#initialise variables
start_date = datetime.now() - timedelta(days=29)
datetime_list = generate_times(start_date)

locations = [
    {"name": "Brooklyn", "latitude": 40.678177, "longitude": -73.944160},
    {"name": "Bronx", "latitude": 40.840347, "longitude": -73.876969},
    {"name": "Manhattan", "latitude": 40.787534, "longitude": -73.961126},
    {"name": "Staten Island", "latitude": 40.599252, "longitude": -74.114240},
    {"name": "Queens", "latitude": 40.725119, "longitude": -73.788628},
]

api_key = 'AIzaSyBCCAHGjX6o2Oh2JxPssW00fDoDaDRzZkE'
url = 'https://airquality.googleapis.com/v1/history:lookup?key={}'.format(api_key)

#prep an empty list for collecting data
data_for_df = []

#iterate through each location
for location in locations:
    #iterate through each datetime to fetch data for the current location
    for dt in datetime_list:
        formatted_timestamp = dt.strftime('%Y-%m-%dT%H:%M:%SZ')
        payload = {
            "dateTime": formatted_timestamp,
            "location": {
                "latitude": location['latitude'],
                "longitude": location['longitude']
            },
            "universalAqi": True,
            "extraComputations": [
                "POLLUTANT_CONCENTRATION",
                "DOMINANT_POLLUTANT_CONCENTRATION"
            ]
        }
        
        response = requests.post(url, json=payload)
        
        if response.status_code == 200:
            response_data = response.json()
            hour_info = response_data.get('hoursInfo', [{}])[0]
            
            #extract AQI and dominant pollutant details
            indexes = hour_info.get('indexes', [{}])[0]
            aqi = indexes.get('aqi', 'N/A')
            category = indexes.get('category', 'N/A')
            dominant_pollutant = indexes.get('dominantPollutant', 'N/A')
            
            #initialise row data with common details
            row_data = {
                'DateTime': dt.strftime('%Y-%m-%d %H:%M:%S'),
                'Location': location['name'],
                'Latitude': location['latitude'],
                'Longitude': location['longitude'],
                'AQI': aqi,
                'Category': category,
                'Dominant Pollutant': dominant_pollutant
            }
            
            #extract and add specific pollutant concentrations
            pollutants = hour_info.get('pollutants', [])
            for pollutant in pollutants:
                code = pollutant.get('code')
                concentration = pollutant.get('concentration', {}).get('value', 'N/A')
                units = pollutant.get('concentration', {}).get('units', '')
                row_data[f"{code} ({units})"] = concentration
            
            data_for_df.append(row_data)
        else:
            print(f"Error fetching data for {formatted_timestamp} at {location['name']}: {response.status_code} - {response.text}")

#create a datafarme from the collected data and save as csv
df = pd.DataFrame(data_for_df)

print(df)
df.to_csv(r"C:\Users\spash\Documents\AQI DATA\AQI DATA.csv")
