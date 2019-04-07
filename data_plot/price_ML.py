import pandas as pd
import numpy as np
import datetime

from pandas import Timestamp

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
#fig = plt.figure(figsize=(16,8))
#ax = fig.add_subplot(111) 
#ax.plot(df['Date'], df['Close'], label='Close Price history')
#ax.set_xlim(datetime.date(2013,6,1), datetime.date(2018,12,1))
###############################################################

#creating dataframe with date and the target variable
data = df.sort_index(ascending=True, axis=0)
new_data = pd.DataFrame(index=range(0,len(df)),columns=['Date', 'Close'])

for i in range(0,len(data)):
     new_data['Date'][i] = data['Date'][i]
     new_data['Close'][i] = data['Close'][i]

#splitting into train and validation
train = new_data[:987]
valid = new_data[987:]

print(new_data.shape, train.shape, valid.shape)
#((1235, 2), (987, 2), (248, 2))

print(train['Date'].min(), train['Date'].max(), valid['Date'].min(), valid['Date'].max())

#(Timestamp('2013-10-08 00:00:00'),
#Timestamp('2017-10-06 00:00:00'),
#Timestamp('2017-10-09 00:00:00'),
#Timestamp('2018-10-08 00:00:00'))

#make predictions
preds = []
for i in range(0,248):
    a = train['Close'][len(train)-248+i:].sum() + sum(preds)
    b = a/248
    preds.append(b)

#calculate rmse
rms=np.sqrt(np.mean(np.power((np.array(valid['Close'])-preds),2)))
print(rms)

pd.options.mode.chained_assignment = None  # default='warn'

#plot
valid['Predictions'] = 0
valid['Predictions'] = preds

fig = plt.figure(figsize=(16,8))
ax = fig.add_subplot(111)
ax.plot(train['Date'], train['Close']) #need add train[...]
ax.plot(valid['Date'], valid[['Close', 'Predictions']]) #same above
ax.set_xlim(datetime.date(2013,6,1), datetime.date(2018,12,1))

plt.show()
