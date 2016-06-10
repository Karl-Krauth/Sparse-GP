import numpy as np
import pandas as pd

# 1. Load the dataset
data = pd.read_csv('temp.csv')
data.fillna(value=np.nan)
data_filtered = data[['DayOfWeek', 'AirTime', 'Distance', 'plane_year', 'DepTime', 'ArrTime', 'DayofMonth', 'Month', 'ArrDelay']].dropna()
actual_delays = data_filtered['ArrDelay']>0
data_filtered = data_filtered[actual_delays]
X = data_filtered[['DayOfWeek', 'AirTime', 'Distance', 'plane_year', 'DepTime', 'ArrTime', 'DayofMonth', 'Month']].values
Y = data_filtered['ArrDelay'].values[:, None]
print X.shape, Y.shape

# Transform plane deployment year in plane age
X[:, 3] = 2008 - X[:, 3]

# Transform timestamps to hour values.
X[:, 4] = np.floor_divide(X[:, 4], 100) + np.mod(X[:, 4], 100) / 60.0
X[:, 5] = np.floor_divide(X[:, 5], 100) + np.mod(X[:, 5], 100) / 60.0

# Split the data into test and training set.
N, D = X.shape
num_test = 100000# N*0.10
num_train = 700000
N_shuffled = np.random.permutation(N)
test = N_shuffled[:num_test]
train = N_shuffled[num_test:num_test + num_train]
X_train, Y_train = X[train], Y[train]
X_test, Y_test = X[test], Y[test]

# Output test and training data.
np.savetxt("train.csv", np.concatenate([X_train, Y_train], 1), delimiter=",")
np.savetxt("test.csv", np.concatenate([X_test, Y_test], 1), delimiter=",")
