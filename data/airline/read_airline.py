import numpy as np
import pandas as pd

# 1. Load the dataset
data = pd.read_csv('temp.csv')
data.fillna(value=np.nan)
data_filtered = data[['DayOfWeek', 'AirTime', 'Distance', 'plane_year', 'DepTime', 'ArrTime', 'DayofMonth', 'Month', 'ArrDelay']].dropna()
actual_delays = data_filtered['ArrDelay']>0
data_filtered = data_filtered[actual_delays]
X = data_filtered[['DayOfWeek', 'AirTime', 'Distance', 'plane_year', 'DepTime', 'ArrTime', 'DayofMonth', 'Month']].values
# X = data_filtered[['Distance']].values
Y = data_filtered['ArrDelay'].values[:, None]
print X.shape, Y.shape

# Transform plane deployment year in plane age
X[:, 3] = 2008 - X[:, 3]
X[:, 4] = np.floor_divide(X[:, 4], 100) + np.mod(X[:, 4], 100) / 60.0
X[:, 5] = np.floor_divide(X[:, 5], 100) + np.mod(X[:, 5], 100) / 60.0

# 2. Holdout sets
N, D = X.shape
num_test = 100000# N*0.10
num_train = 700000
N_shuffled = np.random.permutation(N)
test = N_shuffled[:num_test]
train = N_shuffled[num_test:num_test + num_train]
X_train, Y_train = X[train], Y[train]
X_test, Y_test = X[test], Y[test]

# print data_filtered.columns.values
# print "train", X_train.mean(0)
# print "test", X_test.mean(0)
# print Y_train.mean(), Y_test.mean()

np.savetxt("train.csv", np.concatenate([X_train, Y_train], 1), delimiter=",")
np.savetxt("test.csv", np.concatenate([X_test, Y_test], 1), delimiter=",")

# Xmean = X_train.mean(0)
# Xstd = X_train.std(0)
# X_train = (X_train-Xmean)/Xstd

# Ymean = Y_train.mean()
# Ystd = Y_train.std()
# Y_train = (Y_train-Ymean)/Ystd
