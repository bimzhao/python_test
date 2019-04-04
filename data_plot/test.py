import pandas as pd
import numpy as np
import datetime

#to plot within notebook
import matplotlib.pyplot as plt
#%matplotlib inline

#setting figure size
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 20,10

#for normalizing data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

#read the file
df = pd.read_csv('NSE-TATAGLOBAL11.csv')

#print the head
df.head()

#setting index as date
df['Date'] = pd.to_datetime(df.Date,format='%Y-%m-%d')
df.index = df['Date']

#plot
fig = plt.figure(figsize=(16,8))
ax = fig.add_subplot(111) 
ax.plot(df['Date'], df['Close'], label='Close Price history')
ax.set_xlim(datetime.date(2013,6,1), datetime.date(2018,12,1))
plt.show()
