import pandas as pd
from pandas import Series, DataFrame

s = pd.read_csv('C:\\Users\\Raptor\\OneDrive\\Desktop\\python\\pmat les\\pr01\\dataset\\taxi-passenger-count.csv', header=None).squeeze()
# print(s.value_counts(normalize=True)[[1,6]])
# print(s.quantile([0.25, 0.5, 0.75]))
# print(s.value_counts(normalize=True)[[3,4,5,6]].sum())