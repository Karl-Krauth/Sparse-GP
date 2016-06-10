import pandas as pd
import numpy as np


res1 = pd.read_csv('result.csv')
res2 = pd.read_csv('plane-data.csv')[['TailNum', 'plane_year']].replace('None', np.nan).dropna()
res = pd.merge(res1, res2, on='TailNum')
res.to_csv('temp.csv')
