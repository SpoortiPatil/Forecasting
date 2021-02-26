# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# Import the dataset
coke = pd.read_excel("D:\Data science\Assignments docs\Forecasting\CocaCola_Sales.xlsx")
coke.head()
coke.shape
coke.describe()
coke.info()

#Converting the 'Quarter' format to datetime format
index = []
for i in range(42):
    p = i
    index.append(i)
coke['index'] = index
coke.head()

#Converting 'index' column as index
coke.set_index('index', inplace=True)
coke.head()

#Visualizing the data
coke.plot()

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
adfuller_test(coke['Sales'])
#Since the data is not stationary, we have to difference it to make it stationary

coke_diff = coke["Sales"].diff(periods=1)
coke_diff.head()

# Again testing using dickey fuller test
adfuller_test(coke_diff.dropna())
coke_diff.plot()
#The p-value is still higher and its not stationary after first order differentiation, hence going for 2nd order differentiatio

# Another order of differentiation
coke_diff2 = coke_diff.diff(periods=1)
coke_diff2.head()

# Again testing using dickey fuller test
adfuller_test(coke_diff2.dropna())
coke_diff2.plot()
#The data has become stationary after 2nd order of differentiation

from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
#Plotting auto correlation and partial correlation plot
plot_acf(coke_diff2.dropna(),lags=20)
#The 'q' value for ARIMA model can be taken as 0 referring the above plot

plot_pacf(coke_diff2.dropna(),lags=18)
#The 'p' value for ARIMA model can be taken as 0 referring the above plot

#Model building
from statsmodels.tsa.arima_model import ARIMA
model=ARIMA(coke['Sales'],order=(0,2,0))
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
for q in range(4):
    try:
        model=ARIMA(coke.Sales, order=(p,d,q)).fit(disp=0)
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
{(0, 2, 0): 633.3976676318601, (0, 2, 1): 608.9435808960338, (0, 2, 2): 593.1928392438945, (0, 2, 3): 618.5524422768146}

#For order of (0,2,2) AIC value is least at 593.19, hence building model at this order
model=ARIMA(coke['Sales'],order=(0,2,2))
model_fit=model.fit()
model_fit.aic

#Predicting the values for this model
coke.shape
coke.tail()
coke['forecast']=model_fit.predict(start=33,end=41,dynamic=True)
coke[['Sales','forecast']].plot(figsize=(12,8))

#Using the SARIMAX to predict the values for seasonal data
import statsmodels.api as sm
model=sm.tsa.statespace.SARIMAX(coke['Sales'],order=(0, 2, 2),seasonal_order=(0,2,2,4))
results=model.fit()
coke['forecast']=results.predict(start=33,end=41,dynamic=True)
coke[['Sales','forecast']].plot(figsize=(12,8))
#Forecasted value is very similar to the acutal values, hence the model is good

#Forecasting the future values
future_dates=[x for x in range(42,50)]
len(future_dates)

future_dates_df=pd.DataFrame(index=future_dates[1:],columns=coke.columns)
future_df=pd.concat([coke,future_dates_df], axis=0)
future_df.shape

future_df['forecast'] = results.predict(start = 41, end = 49, dynamic= True)  
future_df[['Sales', 'forecast']].plot(figsize=(12, 8))

#The forecasted values looks good as it follows both trend and seasonality of original data

# Forecasted values of 2 future years
future_df['forecast'].iloc[41:]