#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('ln -sf /opt/bin/nvidia-smi /usr/bin/nvidia-smi')
get_ipython().system('pip install gputil')
get_ipython().system('pip install psutil')
get_ipython().system('pip install humanize')
get_ipython().system('pip install torch')
get_ipython().system('pip install folium')


import psutil
import humanize
import os
import GPUtil as GPU
GPUs = GPU.getGPUs()


# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
from sklearn import linear_model
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import BaggingRegressor
import torch
import torch.nn as nn
import folium
from folium import FeatureGroup, LayerControl, Map, Marker
from folium.plugins import HeatMap
from folium.plugins import TimestampedGeoJson
from folium.plugins import MarkerCluster


# In[5]:


df = pd.read_csv('/Users/bindushreeha/downloads/ny/train.csv')
df.head()
df.isnull().sum()
df.dropna(how = 'any', inplace = True)
duplicate_df = df[df.duplicated()]
print('Duplicate Rows: ', duplicate_df.shape)


# In[ ]:


#FARE AMOUNT ANALYSIS
df['fare_amount'].describe()
neg_fare = np.array(np.where(df['fare_amount']<0))
neg_fare

#output
#count    1.199990e+06
#mean     1.134186e+01
#std      9.877427e+00
#min     -4.500000e+01
#25%      6.000000e+00
#50%      8.500000e+00
#75%      1.250000e+01
#max      1.273310e+03
#Name: fare_amount, dtype: float64


# In[ ]:


df.drop(df.index[neg_fare], inplace = True)
df['fare_amount'].describe()


# In[ ]:


#PASSENGER COUNT ANALYSIS
df['passenger_count'].describe()
#Generally a new york taxi can accomodate 4 people. But the above describe method shows that there a more than 4 people in a taxi.
#If we there are more than 4 people in a taxi, that record will be dropped.
taxi_carry = np.array(np.where(df['passenger_count']>4))
taxi_carry
df.drop(df.index[taxi_carry], inplace = True)


# In[ ]:


#LATITUDE AND LONGITUDE ANALYSIS
df['pickup_latitude'].describe()
df['dropoff_latitude'].describe()


# In[ ]:


out_lat_pickup = np.array(np.where((df['pickup_latitude']< -90) | (df['pickup_latitude'] > 90)))
out_lat_drop = np.array(np.where((df['dropoff_latitude']< -90) | (df['dropoff_latitude'] > 90)))
lat_out = np.column_stack((out_lat_pickup,out_lat_drop))
df.drop(df.index[lat_out], inplace = True)
df['pickup_longitude'].describe()
df['dropoff_longitude'].describe()


# In[ ]:


out_long_pickup = np.array(np.where((df['pickup_longitude']< -180) | (df['pickup_longitude'] > 180)))
out_long_drop = np.array(np.where((df['dropoff_longitude']< -180) | (df['dropoff_longitude'] > 180)))
out_long = np.column_stack((out_long_pickup,out_long_drop))
#After finding all such outlier values, we can drop them from the dataset.
df.drop(df.index[out_long], inplace = True)


# In[ ]:


#NEW YORK LATITUDE AND LONGITUDE
#Considering the city of New York, we have to keep in mind the longitude and latitude boundaries of the city. After some domain research, i found out that,
#Longitude Boundary - (-74.03, -73.75)
#Latitude Boundary - (40.63, 40.85)
#Hence, we need to remove values outside this boundary.
boundary = {'min_lng':-74.263242,
              'min_lat':40.573143,
              'max_lng':-72.986532, 
              'max_lat':41.709555}

outside_nyc = np.array(np.where(~((df.pickup_longitude >= boundary['min_lng'] ) & (df.pickup_longitude <= boundary['max_lng']) &
            (df.pickup_latitude >= boundary['min_lat']) & (df.pickup_latitude <= boundary['max_lat']) &
            (df.dropoff_longitude >= boundary['min_lng']) & (df.dropoff_longitude <= boundary['max_lng']) &
            (df.dropoff_latitude >=boundary['min_lat']) & (df.dropoff_latitude <= boundary['max_lat']))))

df.drop(df.index[outside_nyc], inplace = True)

df.info()


# In[ ]:


#EXPLORATORY DATA ANALYSIS
feat_corr = df.corr()
plt.figure(figsize = (12,8))
sns.heatmap(feat_corr, annot = True, cmap = 'coolwarm')
b, t = plt.ylim() 
b += 0.5 
t -= 0.5 
plt.ylim(b, t)
plt.show()


# In[ ]:


df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
df['hour'] = df['pickup_datetime'].dt.hour
df['day_of_week'] = df['pickup_datetime'].dt.day
df['month'] = df['pickup_datetime'].dt.month
df['year'] = df['pickup_datetime'].dt.year


# In[ ]:


#We can gain more insight on the fare amount by calculating the distance between the pickup and dropoff locations based on the latitude and longitude values.
#Haversine Distance Formula.
def haversine_distance(df, lat1, long1, lat2, long2):
    """
    Calculates the haversine distance between 2 sets of GPS coordinates in df
    """
    r = 6371  # average radius of Earth in kilometers
       
    phi1 = np.radians(df[lat1])
    phi2 = np.radians(df[lat2])
    
    delta_phi = np.radians(df[lat2]-df[lat1])
    delta_lambda = np.radians(df[long2]-df[long1])
     
    a = np.sin(delta_phi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    d = (r * c) # in kilometers

    return d

df['distance'] = haversine_distance(df,'pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude')
df.head()



# In[ ]:


#1. Highest Pickup and Dropoff Locations within NYC

long_border = (-74.03, -73.75)
lat_border = (40.63, 40.85)

df.plot(x = 'pickup_longitude', y = 'pickup_latitude', kind = 'scatter',s=.02, alpha=.6,title = 'Pickups',figsize = (12,8),color = 'green')
plt.xlim(long_border)
plt.ylim(lat_border)


# In[ ]:


long_border = (-74.03, -73.75)
lat_border = (40.63, 40.85)

df.plot(x = 'dropoff_longitude', y = 'dropoff_latitude', kind = 'scatter',s=.02, alpha=.6,title = 'Dropoffs',figsize = (12,8),color = 'red')
plt.xlim(long_border)
plt.ylim(lat_border)


# In[ ]:


#Airport trips are quite expensive as they are far from the main city.

#I will be trying to check the fare amount from the two main airports in New York

# 1. John F Kennedy International Airport
# 2. La Guardia Airport
JFK = {'min_long':-73.8352,
     'min_lat':40.6195,
     'max_long':-73.7401, 
     'max_lat':40.6659}

jfk_center = [40.6437,-73.7900]

jfk_pickup = df.loc[(df['pickup_longitude']>= JFK['min_long']) & (df['pickup_longitude'] <= JFK['max_long'])]

jfk_pickup = df.loc[(df['pickup_latitude']>= JFK['min_lat']) & (df['pickup_latitude'] <= JFK['max_lat'])]

jfk_pickup.shape[0]

jfk_dropoff = df.loc[(df['dropoff_longitude']>= JFK['min_long']) & (df['dropoff_longitude'] <= JFK['max_long'])]

jfk_dropoff = df.loc[(df['dropoff_latitude']>= JFK['min_lat']) & (df['dropoff_latitude'] <= JFK['max_lat'])]

jfk_dropoff.shape[0]


# In[ ]:


m=folium.Map(location =jfk_center,zoom_start = 10,)
folium.Marker(location=jfk_center, popup='JFK Airport',icon=folium.Icon(color='black')).add_to(m)

mc = MarkerCluster().add_to(m)
#Add markers in blue for each pickup location and line between JFK and Pickup location over time. The thickness of line indicates the fare_amount

for index,row in jfk_pickup.iterrows():
    folium.CircleMarker([row['dropoff_latitude'],row['dropoff_longitude']],radius = 3).add_to(m)


# In[ ]:


plt.figure(figsize = (12,8))
sns.kdeplot(np.log(jfk_pickup['fare_amount'].values),label='JFK Pickups')
sns.kdeplot(np.log(jfk_dropoff['fare_amount'].values),label='JFK Dropoff')
sns.kdeplot(np.log(df['fare_amount'].values),label='All Trips in Train data')
plt.title("Fare Amount Distribution")


# In[ ]:


#Pickup fare amount is pretty high compared to dropoff fare amount.
#We also observe that fare amount to and from the airport are high compared to overall trip fare amounts.
#The city of new york is mainly divided into 5 boroughs. Namely-
# 1. Manhatten
# 2. Queens
# 3. Brooklyn
# 4. Bronx
# 5. Staten Island
nyc_boroughs = {
        'manhattan':{
        'min_lng':-74.0479,
        'min_lat':40.6829,
        'max_lng':-73.9067,
        'max_lat':40.8820
    },
    
    'queens':{
        'min_lng':-73.9630,
        'min_lat':40.5431,
        'max_lng':-73.7004,
        'max_lat':40.8007

    },

    'brooklyn':{
        'min_lng':-74.0421,
        'min_lat':40.5707,
        'max_lng':-73.8334,
        'max_lat':40.7395

    },

    'bronx':{
        'min_lng':-73.9339,
        'min_lat':40.7855,
        'max_lng':-73.7654,
        'max_lat':40.9176

    },

    'staten_island':{
        'min_lng':-74.2558,
        'min_lat':40.4960,
        'max_lng':-74.0522,
        'max_lat':40.6490

    }
    
}


# In[ ]:


def borough(lat,long):
    locs=nyc_boroughs.keys()
    for loc in locs:
        if lat>=nyc_boroughs[loc]['min_lat'] and lat<=nyc_boroughs[loc]['max_lat'] and long>=nyc_boroughs[loc]['min_lng'] and long<=nyc_boroughs[loc]['max_lng']:
            return loc
    return 'others'


# In[ ]:


df['pickup_borough'] = df.apply(lambda row : borough(row['pickup_latitude'],row['pickup_longitude']), axis = 1)
df['dropoff_borough'] = df.apply(lambda row : borough(row['dropoff_latitude'],row['dropoff_longitude']), axis = 1)
df.head()


# In[ ]:


plt.figure(figsize = (12,8))
sns.countplot(y = df['pickup_borough'])


# In[ ]:


plt.figure(figsize = (12,8))
sns.countplot(y = df['dropoff_borough'])


# In[ ]:


#Here, we have visualized the distribution of pickup boroughs and dropoff boroughs.
#Not a suprise, Manhattan has the highest number of pickups and dropoffs followed by Queens.
#Finally, we will check the price distribution for each of the boroughs
plt.figure(figsize = (16,10))

i = 1
for key in nyc_boroughs.keys():
    plt.subplot(3,2,i)
    sns.kdeplot(np.log(df.loc[df['pickup_borough']==key,'fare_amount'].values),label='Pickup '+ key)
    sns.kdeplot(np.log(df.loc[df['dropoff_borough']==key,'fare_amount'].values),label='Dropoff'+ key).set_title("Fare Amount (log scale) for "+key)
    i=i+1
plt.title('Distribution of Fare Amount for Each Borough')
plt.tight_layout()


# In[ ]:


#Machine Learning
X = df.drop(['key','fare_amount','pickup_datetime'], axis = 1)

y = df['fare_amount']
X.head()
y.head()
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.25)


# In[ ]:


#LINEAR REGRESSION
linReg = linear_model.LinearRegression()
linReg.fit(X_train, y_train)
linPred = linReg.predict(X_test)

linError = mean_squared_error(y_test,linPred)
lin_r2 = r2_score(y_test,linPred)

print(f'Mean Squared Error for Linear Regression: {linError}')
print('\n')
print(f'R2 Score for Linear Regression: {lin_r2}')

#output
#Mean Squared Error for Linear Regression: 70.00993834859254
#R2 Score for Linear Regression: 0.2561362957098128


# In[ ]:


#RIDGE REGRESSION
ridReg = linear_model.Ridge(alpha=.5)
ridReg.fit(X_train, y_train)
ridPred = ridReg.predict(X_test)

ridError = mean_squared_error(y_test,ridPred)
rid_r2 = r2_score(y_test,ridPred)

print(f'Mean Squared Error for Ridge Regression: {ridError}')
print('\n')
print(f'R2 Score for Ridge Regression: {rid_r2}')

#output
#Mean Squared Error for Ridge Regression: 70.00966764094561
#R2 Score for Ridge Regression: 0.25613917200991776


# In[ ]:


#LASSO REGRESSION
lasso_reg = linear_model.Lasso(alpha=0.1)
lasso_reg.fit(X_train, y_train)
lasso_pred = lasso_reg.predict(X_test)

lasso_error = mean_squared_error(y_test, lasso_pred)
lasso_r2 = r2_score(y_test, lasso_pred)

print(f'Mean Squared Error for Lasso Regression: {lasso_error}')
print('\n')
print(f'R2 Score for Lasso Regression: {lasso_r2}')

#output
#Mean Squared Error for Lasso Regression: 87.17457011678741
#R2 Score for Lasso Regression: 0.07376009497254776


# In[ ]:


#K-NEAREST NEIGHBORS
knn = KNeighborsRegressor(n_neighbors=4)
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)

knn_error = mean_squared_error(y_test, knn_pred)
knn_r2 = r2_score(y_test, knn_pred)

print(f'Mean Squared Error for K-Nearest Neighbor: {knn_error}')
print('\n')
print(f'R2 Score for K-Nearest Neighbor: {knn_r2}')

#output
#Mean Squared Error for K-Nearest Neighbor: 20.41052551429924
#R2 Score for K-Nearest Neighbor: 0.7831358022345518


# In[ ]:


#DECISION TREE
cart = DecisionTreeRegressor()
cart.fit(X_train, y_train)
cart_pred = cart.predict(X_test)

cart_error = mean_squared_error(y_test, cart_pred)
cart_r2 = r2_score(y_test, cart_pred)

print(f'Mean Squared Error for Decision Tree: {cart_error}')
print('\n')
print(f'R2 Score for Decision Tree: {cart_r2}')

#output
#Mean Squared Error for Decision Tree: 33.488145551338526
#R2 Score for Decision Tree: 0.6441845745443593


# In[ ]:


#RANDOM FOREST
rand_regr = RandomForestRegressor(n_estimators=100)
rand_regr.fit(X_train, y_train)
rand_pred = rand_regr.predict(X_test)

rand_error = mean_squared_error(y_test, rand_pred)
rand_r2 = r2_score(y_test, rand_pred)

print(f'Mean Squared Error for Random Forest: {rand_error}')
print('\n')
print(f'R2 Score for Random Forest: {rand_r2}')

#output
#Mean Squared Error for Random Forest: 18.086985069537818
#R2 Score for Random Forest: 0.8078236885986604


# In[ ]:


#ADABOOST
ada_regr = AdaBoostRegressor(random_state=0, n_estimators=100)
ada_regr.fit(X_train, y_train)
ada_pred = ada_regr.predict(X_test)

ada_error = mean_squared_error(y_test, ada_pred)
ada_r2 = r2_score(y_test,ada_pred)

print(f'Mean Squared Error for AdaBoost: {ada_error}')
print('\n')
print(f'R2 Score for AdaBoost: {ada_r2}')

#output
#Mean Squared Error for AdaBoost: 71.75043103887319
#R2 Score for AdaBoost: 0.2376433592721936


# In[ ]:


#GRADIENT BOOSTING
grad_reg = GradientBoostingRegressor(n_estimators=100, learning_rate=0.01,max_depth=1, random_state=0, loss='ls')
grad_reg.fit(X_train, y_train)
grad_pred = grad_reg.predict(X_test)

grad_error = mean_squared_error(y_test, grad_pred)
grad_r2 = r2_score(y_test,grad_pred)

print(f'Mean Squared Error for AdaBoost: {grad_error}')
print('\n')
print(f'R2 Score for AdaBoost: {grad_r2}')

#output
#Mean Squared Error for AdaBoost: 71.50312296674711
#R2 Score for AdaBoost: 0.24027103618452827


# In[ ]:


#BAGGING REGRESSOR
bag_reg= BaggingRegressor(n_estimators = 100)
bag_reg.fit(X_train, y_train)
bag_pred = bag_reg.predict(X_test)

bag_error = mean_squared_error(y_test, bag_pred)
bag_r2 = r2_score(y_test, bag_pred)

print(f'Mean Squared Error for Bagging Regressor: {bag_error}')
print('\n')
print(f'R2 Score for Bagging Regressor: {bag_r2}')

#output
#Mean Squared Error for Bagging Regressor: 18.158586105676275
#R2 Score for Bagging Regressor: 0.8070629192960541


# In[ ]:


#ARTIFICIAL NEURAL NETWORK
cont_cols = ['pickup_longitude',
       'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude',
       'passenger_count']
conts_data = np.stack([df[col].values for col in cont_cols], 1)

X = torch.tensor(conts_data, dtype = torch.float)

y = torch.tensor(df['fare_amount'].values, dtype=torch.float).reshape(-1,1)

class TabularModel(nn.Module):

    def __init__(self, n_cont, out_sz, layers, p=0.5):
        super().__init__()
        self.emb_drop = nn.Dropout(p)
        self.bn_cont = nn.BatchNorm1d(n_cont)
        
        layerlist = []
        
        for i in layers:
            layerlist.append(nn.Linear(n_cont,i)) 
            layerlist.append(nn.ReLU(inplace=True))
            layerlist.append(nn.BatchNorm1d(i))
            layerlist.append(nn.Dropout(p))
            n_cont = i
        layerlist.append(nn.Linear(layers[-1],out_sz))
            
        self.layers = nn.Sequential(*layerlist)
    
    def forward(self, x_cont):
        x_cont = self.emb_drop(x_cont)
        
        x_cont = self.bn_cont(x_cont)
        x_cont = self.layers(x_cont)
        return x_cont
    
model = TabularModel(X.shape[1], 1, [200,100], p=0.4)
model


# In[ ]:


criterion = nn.SmoothL1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)

batch_size = 1066815
test_size = int(batch_size * .2)

X_train = X[:batch_size-test_size]
X_test = X[batch_size-test_size:batch_size]
y_train = y[:batch_size-test_size]
y_test = y[batch_size-test_size:batch_size]

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[ ]:


import time
start_time = time.time()

epochs = 100
losses = []

for i in range(epochs):
    i+=1
    y_pred = model(X_train)
    loss = criterion(y_pred, y_train) # RMSE
    losses.append(loss)
    
    
    if i%25 == 1:
        print(f'epoch: {i:3}  loss: {loss.item():10.8f}')

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(f'epoch: {i:3}  loss: {loss.item():10.8f}') 
print(f'\nDuration: {time.time() - start_time:.0f} seconds') 


# In[ ]:


plt.plot(range(epochs), losses)
plt.ylabel('RMSE Loss')
plt.xlabel('epoch');


# In[ ]:


with torch.no_grad():
    y_val = model(X_test)
    loss = torch.sqrt(criterion(y_val, y_test))
print(f'RMSE: {loss:.8f}')
#RMSE: 2.22084880

print(f'{"PREDICTED":>12} {"ACTUAL":>8} {"DIFF":>8}')
for i in range(50):
    diff = np.abs(y_val[i].item()-y_test[i].item())
    print(f'{i+1:2}. {y_val[i].item():8.4f} {y_test[i].item():8.4f} {diff:8.4f}')


# In[ ]:




