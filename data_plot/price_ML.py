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
#print(valid)

fig = plt.figure(figsize=(16,8))
ax = fig.add_subplot(111)
ax.plot(train['Date'], train['Close']) #need add train[...]
ax.plot(valid['Date'], valid[['Close', 'Predictions']]) #same above
ax.set_xlim(datetime.date(2013,6,1), datetime.date(2018,12,1))

#plt.show()
################################################################################################

#setting index as date values
df['Date'] = pd.to_datetime(df.Date,format='%Y-%m-%d')
df.index = df['Date']

#sorting
data = df.sort_index(ascending=True, axis=0)
#print(data)
#print(df)

#creating a separate dataset
new_data = pd.DataFrame(index=range(0,len(df)),columns=['Date', 'Close'])

for i in range(0,len(data)):
    new_data['Date'][i] = data['Date'][i]
    new_data['Close'][i] = data['Close'][i]

print(new_data)

#create features
from fastai.structured import  add_datepart
add_datepart(new_data, 'Date')
new_data.drop('Elapsed', axis=1, inplace=True)  #elapsed will be the time stamp
print(new_data)

new_data['mon_fri'] = 0
for i in range(0,len(new_data)):
    if (new_data['Dayofweek'][i] == 0 or new_data['Dayofweek'][i] == 4):
        new_data['mon_fri'][i] = 1
    else:
        new_data['mon_fri'][i] = 0

#split into train and validation
train = new_data[:987]
valid = new_data[987:]

x_train = train.drop('Close', axis=1)
y_train = train['Close']
x_valid = valid.drop('Close', axis=1)
y_valid = valid['Close']

#implement linear regression
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train,y_train)


#make predictions and find the rmse
preds = model.predict(x_valid)
rms=np.sqrt(np.mean(np.power((np.array(y_valid)-np.array(preds)),2)))
print(rms)

#plot
valid['Predictions'] = 0
valid['Predictions'] = preds

valid.index = new_data[987:].index
train.index = new_data[:987].index
print(valid[['Year', 'Month','Day']])

fig = plt.figure(figsize=(16,8))
ax = fig.add_subplot(111)
ax.plot(train['Close'])
ax.plot(valid[['Close', 'Predictions']])
#ax.set_xlim(datetime.date(2013,6,1), datetime.date(2018,12,1))

plt.show()

