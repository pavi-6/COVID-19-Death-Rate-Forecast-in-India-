from statsmodels.tsa.arima_model import ARIMA
import pandas as pd
import numpy as np
import matplotlib.pyplot as pyplt
from datetime import datetime
from pandas.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_pacf
#from keras.callbacks import History

#history = History()

'''def parser(x):
	return datetime.strptime('190'+x, '%Y-%m')
 
series = read_csv('shampoo-sales.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)'''


df = pd.read_csv('Corona_19.csv',header=0, parse_dates=[0],index_col=0,squeeze=True)
print(df.isnull().sum())
df.dropna(axis=0,inplace=True)
print(df.isnull().sum())
print(df.head())

df.plot()
pyplt.show()

autocorrelation_plot(df)
pyplt.show()

#Running the example, we can see that there is a positive correlation with the first 35-to-37 lags that is perhaps significant for the first 10 lags.
#A good starting point for the AR parameter of the model may be .

plot_pacf(df,lags=30)
pyplt.show()

#d = 2

#Stationary
cases_diff = df.diff(periods=1)
print(cases_diff)


# fit model
model = ARIMA(df, order=(7,0,1))
model_fit = model.fit()
print(model_fit.summary())

from pandas import DataFrame
residuals = DataFrame(model_fit.resid)
residuals.plot()
pyplt.show()

residuals.plot(kind='kde')
pyplt.show()

print(residuals.describe())

