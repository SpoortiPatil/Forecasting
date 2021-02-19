# Import necesssary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# Importing the data
plastic = pd.read_csv("D:\Data science\Assignments docs\Forecasting\PlasticSales.csv")
plastic.head()
plastic.shape
plastic.describe()
plastic.info()

# Checking NA values in the dataset
plastic.isna().sum()
#There are no NA values in the dataset

from pandas import datetime

#Converting the 'month' from object type to 'datetime' format
month = []
for date in plastic["Month"]:
    values = datetime.strptime(date,"%d-%m-%Y")
    month.append(values)
plastic["month"] = month
plastic.head()
plastic.info()

# Dropping original object type month
plastic = plastic.drop(columns=["Month"],axis=1)
plastic.head()

#Converting 'month' column as index
plastic.set_index('month', inplace=True)
plastic.head()

#Visualizing the data
plastic.plot()

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
adfuller_test(plastic['Sales'])

plastic_diff = plastic.diff(periods=1)
plastic_diff.head()

#Again testing using dickey fuller test
adfuller_test(plastic_diff['Sales'].dropna())
plastic_diff['Sales'].plot()

#The data has become stationary after 2nd order of differentiation
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf

#Plotting auto correlation and partial correlation plot
plot_acf(plastic_diff2['Sales'].dropna(),lags=20)
#The 'q' value for ARIMA model can be taken as 0 referring the above plot

plot_pacf(plastic_diff2['Sales'].dropna(),lags=20)
#The 'p' value for ARIMA model can be taken as 0 referring the above plot

#Model building
from statsmodels.tsa.arima_model import ARIMA
model=ARIMA(plastic['Sales'],order=(0,2,0))
model_fit=model.fit()
model_fit.aic
model_fit.summary()

# Checking other values for p,d,q for which the AIC should be the lowest
p=0
d=2
q=0
pdq=[]
aic=[]

import warnings
warnings.filterwarnings('ignore')
for q in range(9):
    try:
        model=ARIMA(plastic.Sales, order=(p,d,q)).fit(disp=0)
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
{(0, 2, 0): 705.8421198982761, (0, 2, 1): 707.8328185216869, (0, 2, 2): 709.3790845787385, (0, 2, 3): 701.6428435105047, (0, 2, 4): 702.2574937851018, (0, 2, 5): 702.1421869852454, (0, 2, 6): 703.9649031759229}
For order of (0,2,3) AIC value is least at 701.64, hence building model at this order

model=ARIMA(plastic['Sales'],order=(0,2,3))
model_fit=model.fit()
model_fit.aic

#Predicting the values for this model
plastic.shape
plastic['forecast']=model_fit.predict(start=48,end=61,dynamic=True)
plastic[['Sales','forecast']].plot(figsize=(12,8))

#Using the SARIMAX to predict the values for seasonal data
import statsmodels.api as sm
model=sm.tsa.statespace.SARIMAX(plastic['Sales'],order=(0, 2, 3),seasonal_order=(0,2,3,12))
results=model.fit()

plastic['forecast']=results.predict(start=48,end=61,dynamic=True)
plastic[['Sales','forecast']].plot(figsize=(12,8))
#Forecasted value is very similar to the acutal values, hence the model is good
#Forecasting the future values

from pandas.tseries.offsets import DateOffset
future_dates=[plastic.index[-1]+ DateOffset(months=x)for x in range(0,24)]
len(future_dates)
future_dates_df=pd.DataFrame(index=future_dates[1:],columns=plastic.columns)
future_dates_df

future_df=pd.concat([plastic,future_dates_df], axis=0)
future_df.shape
future_df['forecast'] = results.predict(start = 59, end = 83, dynamic= True)  
future_df[['Sales', 'forecast']].plot(figsize=(12, 8))

#The forecasted values looks good as it follows both trend and seasonality of original data
# Forecasted values of 2 future years
future_df['forecast'].iloc[59:]