import pandas as pd
import numpy as np
import datetime

def generate_timeseries_data(start_date, num_points, mean, std, trend=0):
    """Generiert Zeitreihendaten."""
    dates = [start_date + datetime.timedelta(minutes=i) for i in range(num_points)]
    values = np.random.normal(mean, std, num_points) + np.arange(num_points) * trend
    df = pd.DataFrame({'timestamp': dates, 'value': values})
    df['timestamp'] = df['timestamp'].astype(str) # Als String speichern
    return df

# Parameter für Client 1
start_date1 = datetime.datetime(2024, 7, 26, 0, 0, 0)
num_points1 = 100
mean1 = 10
std1 = 0.5

# Parameter für Client 2
start_date2 = datetime.datetime(2024, 7, 26, 0, 0, 0)
num_points2 = 100
mean2 = 6
std2 = 1.5
trend2 = 0.03

# Daten generieren
df1 = generate_timeseries_data(start_date1, num_points1, mean1, std1)
df2 = generate_timeseries_data(start_date2, num_points2, mean2, std2, trend2)

# In CSV-Dateien speichern
df1.to_csv('data/client_1.csv', index=False)
df2.to_csv('data/client_2.csv', index=False)

print("Beispieldaten wurden erstellt.")
