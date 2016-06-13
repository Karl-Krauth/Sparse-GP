import pandas as pd
import numpy as np

# Read in flight data and keep months of Jan-April.
flights = pd.read_csv('2008.csv')
flights = flights[(flights['Month'] >= 1) & (flights['Month'] <= 4)]

# Read in plane data and drop invalid values/change headers.
planes = pd.read_csv('plane-data.csv')[['tailnum', 'year']].replace('None', np.nan).replace('0000', np.nan).dropna()
planes.columns = ['TailNum', 'plane_year']

# Merge the two files and output the result.
res = pd.merge(flights, planes, on='TailNum')
res.to_csv('temp.csv')
