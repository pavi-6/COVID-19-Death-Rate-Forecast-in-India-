from statsmodels.tsa.arima_model import ARIMA
import pandas as pd
import numpy as np
import matplotlib.pyplot as pyplt
%matplotlib inline
from datetime import datetime
from pandas.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_pacf

df = pd.read_csv('Covid_final.csv',header=0, parse_dates=[0],index_col=0,squeeze=True)
df = df[['Deaths']]
print(df.isnull().sum())

print(df.isnull().sum())
print(df.head())
print(df.size)
df.plot()
pyplt.show()

X = df.values
X.size
train = X[0:77]
test = X[77:]

from statsmodels.tsa.ar_model import AR
from sklearn.metrics import mean_squared_error
model_ar = AR(train)
model_ar_fit = model_ar.fit()

predictions = model_ar_fit.predict(start=77,end=100)
pyplt.plot(test)
pyplt.plot(predictions,color='red')
