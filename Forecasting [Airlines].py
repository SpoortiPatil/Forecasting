# Import necesssary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# Importing the data
airline = pd.read_excel("Airlines+Data.xlsx")
airline.head()
airline.shape
airline.describe()
airline.info()

# Checking NA values in the dataset
airline.isna().sum()
#There are no NA values in the dataset

#Converting 'month' column as index
airline.set_index('Month', inplace=True)
airline.head()

#Visualizing the data
airline.plot()

#Testing for stationarity
from statsmodels.tsa.stattools import adfuller

# Defining a function for ADF test
def adfuller_test(sales):
    result = adfuller(sales)
    labels = ['ADF Test statistic','p-values','#Lags Used','Number of observations used'] 
    for value, label in zip(result, labels):
        print(label+' : '+str(value) )
    if result[1] <= 0.05:
        print("The data is stationary")
    else:
        print("The data is not stationary")
adfuller_test(airline['Passengers'])
#Since the data is not stationary, we have to difference it to make it stationary

airline_diff = airline.diff(periods=1)
airline_diff.head()

#Again testing using dickey fuller test
adfuller_test(airline_diff['Passengers'].dropna())
airline_diff['Passengers'].plot()

# Another order of differentiation
airline_diff2 = airline_diff.diff(periods=1)
airline_diff2.head()

#Again testing using dickey fuller test
adfuller_test(airline_diff2['Passengers'].dropna())
airline_diff2['Passengers'].plot()
#The data has become stationary after 2nd order of differentiation

from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
#Plotting auto correlation and partial correlation plot
plot_acf(airline_diff2['Passengers'].dropna(),lags=20)
#The 'q' value for ARIMA model can be taken as 0 referring the above plot

plot_pacf(airline_diff2['Passengers'].dropna(),lags=20)
#The 'p' value for ARIMA model can be taken as 0 referring the above plot

#Model building
from statsmodels.tsa.arima_model import ARIMA
model=ARIMA(airline['Passengers'],order=(0,2,0))
model_fit=model.fit()
model_fit.aic
model_fit.summary()

# Checking other values for p,d,q for which the AIC should be the lowest
p=0
d=2
q=0
pdq=[]
aic=[]
for q in range(9):
    try:
        model=ARIMA(airline.Passengers, order=(p,d,q)).fit(disp=0)
        x= model.aic
        x1=p,d,q
        aic.append(x)
        pdq.append(x1)
    except:
        pass

keys=pdq
values=aic
d=dict(zip(keys,values))
print(d)
#For order of (0,2,8) AIC value is least at 832.16, hence building model at this order

model=ARIMA(airline['Passengers'],order=(0,2,8))
model_fit=model.fit()
model_fit.aic

#Predicting the values for this model
airline.size
airline['forecast']=model_fit.predict(start=84,end=97,dynamic=True)
airline[['Passengers','forecast']].plot(figsize=(12,8))

#Using the SARIMAX to predict the values for seasonal data
import statsmodels.api as sm
model=sm.tsa.statespace.SARIMAX(airline['Passengers'],order=(0, 2, 8),seasonal_order=(0,2,8,12))
results=model.fit()
airline['forecast']=results.predict(start=84,end=97,dynamic=True)
airline[['Passengers','forecast']].plot(figsize=(12,8))
#Forecasted value is very similar to the acutal values, hence the model is good

#Forecasting the future values
from pandas.tseries.offsets import DateOffset
future_dates=[airline.index[-1]+ DateOffset(months=x)for x in range(0,24)]
len(future_dates)
future_dates_df=pd.DataFrame(index=future_dates[1:],columns=airline.columns)
future_dates_df
future_df=pd.concat([airline,future_dates_df], axis=0)
future_df.shape
future_df['forecast'] = results.predict(start = 94, end = 119, dynamic= True)  
future_df[['Passengers', 'forecast']].plot(figsize=(12, 8))
#The forecasted values looks good as it follows both trend and seasonality of original data¶

# Forecasted values of 2 future years
future_df['forecast'].iloc[96:]