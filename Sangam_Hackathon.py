import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib
from sklearn.model_selection import KFold,cross_val_score
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout
from sklearn.linear_model import LinearRegression


#Preprocessing Training data
train_df=pd.read_csv('Train.csv')
d_weather_type = {'Clouds':0,
                  'Clear':1,
                  'Rain':2,
                  'Drizzle':3, 
                  'Mist':4,
                  'Haze':5, 
                  'Fog':6,
       'Thunderstorm':7,
       'Snow':8, 
       'Smoke':9
      }

#Converting is_holiday column to categorical
train_df['is_holiday']=train_df['is_holiday'].apply(lambda x: 1 if x!='None' else 0)

#deleting the rows with duplicated date_time column
train_df = train_df[train_df['date_time'].duplicated()==False]


#converting date_time column to datetime format
train_df['date_time']=pd.to_datetime(train_df['date_time'])
train_df = train_df.set_index(train_df['date_time'])

train_df=train_df.drop(train_df['date_time']['2013-05-12 02:00:00'],axis=0)

train_df['weather_type']=train_df['weather_type'].map(d_weather_type)

#Converting holidays to that particular whole day(24 hours) 
for i in tqdm(range(len(train_df))):
    if train_df['is_holiday'][i]==1:
        p = train_df['date_time'].dt.date[i]
        s = p.strftime('%Y')+'-'+p.strftime('%m')+'-'+p.strftime('%d')
        train_df.loc[train_df['date_time'][s],'is_holiday']=1

d1 = {
    5:1,
    6:1,
    1:0,
    2:0,
    3:0,
    4:0,
    0:0
}

#creating new features
train_df['is_weekend']=train_df.date_time.dt.dayofweek
train_df['is_weekend']=train_df['is_weekend'].map(d1)

train_df['hour']=train_df['date_time'].dt.hour

#Getting only the required features in another dataframe for training
train_dataframe=pd.DataFrame()
train_dataframe['is_holiday'] = train_df['is_holiday']
train_dataframe['is_weekend'] = train_df['is_weekend']
train_dataframe['visibility_in_miles'] = train_df['visibility_in_miles']
train_dataframe['temperature'] = train_df['temperature']
train_dataframe['clouds_all'] = train_df['clouds_all']
train_dataframe['weather_type'] = train_df['weather_type']
train_dataframe['hour'] = train_df['hour']
train_dataframe['rain_p_h']=train_df['rain_p_h']
#train_dataframe['snow_p_h']=train_df['snow_p_h']
train_dataframe['traffic_volume'] = train_df['traffic_volume']

train_dataframe.to_csv('training_dataframe.csv')


x_train=train_dataframe.iloc[:,:-1].values
y_train=train_dataframe.iloc[:,-1].values




label_encoder_w = LabelEncoder()
x_train[:,5] = label_encoder_w.fit_transform(x_train[:,5])
onehotencoder_w = OneHotEncoder(categorical_features=[5])
x_train = onehotencoder_w.fit_transform(x_train).toarray()

x_train=x_train[:,1:]
#converting hours column into categorical variable
label_encoder_h = LabelEncoder()
x_train[:,-2] = label_encoder_h.fit_transform(x_train[:,-2])
onehotencoder_h = OneHotEncoder(categorical_features=[-2])
x_train = onehotencoder_h.fit_transform(x_train).toarray()

#Avoiding the dummy variable trap
x_train=x_train[:,1:]


 
#Applying Data preprocessing same as above with the test set
test_df=pd.read_csv('Test.csv')
test_df['is_holiday']=test_df['is_holiday'].apply(lambda x: 1 if x!='None' else 0)
test_df['date_time']=pd.to_datetime(test_df['date_time'])

test_df = test_df.set_index(test_df['date_time'])

test_df['weather_type']=test_df['weather_type'].map(d_weather_type)


for i in tqdm(range(len(test_df))):
    if test_df['is_holiday'][i]==1:
        p = test_df['date_time'].dt.date[i]
        s = p.strftime('%Y')+'-'+p.strftime('%m')+'-'+p.strftime('%d')
        test_df.loc[test_df['date_time'][s],'is_holiday']=1

test_df['is_weekend']=test_df.date_time.dt.dayofweek
test_df['is_weekend']=test_df['is_weekend'].map(d1)
test_df['hour']=test_df['date_time'].dt.hour

test_dataframe=pd.DataFrame()
test_dataframe['is_holiday'] = test_df['is_holiday']
test_dataframe['is_weekend'] = test_df['is_weekend']
test_dataframe['visibility_in_miles'] = test_df['visibility_in_miles']
test_dataframe['temperature'] = test_df['temperature']
test_dataframe['clouds_all'] = test_df['clouds_all']
test_dataframe['weather_type'] = test_df['weather_type']
test_dataframe['hour'] = test_df['hour']
test_dataframe['rain_p_h']=test_df['rain_p_h']

test_dataframe.to_csv('testing_dataframe.csv')

x_test=test_dataframe.iloc[:,:].values

label_encoder_test_w = LabelEncoder()
x_test[:,5] = label_encoder_test_w.fit_transform(x_test[:,5])
onehotencoder_test_w = OneHotEncoder(categorical_features=[5])
x_test = onehotencoder_test_w.fit_transform(x_test).toarray()
x_test=x_test[:,1:]

label_encoder_test_h = LabelEncoder()
x_test[:,-2] = label_encoder_test_h.fit_transform(x_test[:,-2])
onehotencoder_test_h = OneHotEncoder(categorical_features=[-2])
x_test = onehotencoder_test_h.fit_transform(x_test).toarray()
x_test=x_test[:,1:]


#feature_scaling
scalar = MinMaxScaler(feature_range=(0,1))
scaled_x= scalar.fit_transform(x_train)
scaled_x_test = scalar.transform(x_test)


random_model = RandomForestRegressor(n_estimators=200)
random_model.fit(x_train,y_train)
cross_val_score(estimator=random_model,X=x_train, y=y_train,cv=5).mean()


random_pred=random_model.predict(x_test)
random_pred=list(random_pred)
random_pred_int = [float(round(i)) for i in random_pred]


submission=pd.DataFrame()
submission['date_time']=test_df['date_time']
submission['traffic_volume']=random_pred_int
submission.to_csv('random_submission_later.csv',index=False)
joblib.dump(random_model,'acc_99.95716.pkl')
cross_val_score(estimator=random_model,X=x_train, y=y_train,cv=5).mean()





ANN_model2 = Sequential()
# The Input Layer :
ANN_model2.add(Dense(128, kernel_initializer='normal',input_dim = scaled_x.shape[1], activation='relu'))
# The Hidden Layers :
ANN_model2.add(Dropout(0.2))
ANN_model2.add(Dense(256, kernel_initializer='normal',activation='relu'))

ANN_model2.add(Dense(256, kernel_initializer='normal',activation='relu'))
ANN_model2.add(Dropout(0.2))

ANN_model2.add(Dense(256, kernel_initializer='normal',activation='relu'))
ANN_model2.add(Dropout(0.2))

# The Output Layer :
ANN_model2.add(Dense(1, kernel_initializer='normal',activation='linear'))
# Compile the network :

ANN_model2.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])

ANN_model2.fit(scaled_x, y_train, epochs=200, batch_size=32, validation_split = 0.1)

ann_pred2 = ANN_model2.predict(scaled_x_test)
ann_best =[ int(ann_pred2[i]) for i in range(ann_pred2.shape[0])]

submission=pd.DataFrame()
submission['date_time']=test_df['date_time']
submission['traffic_volume']=ann_best
submission.to_csv('ann_relu_submission.csv',index=False)

joblib.dump(ANN_model2,'acc_99.95899.pkl')



x_train_prev = train_dataframe.iloc[:,[0,1,3,4,6,7]].values
x_test_prev =  test_dataframe.iloc[:,[0,1,3,4,6,7]].values

onehotencoder_h_prev = OneHotEncoder(categorical_features=[-2])
x_train_prev = onehotencoder_h_prev.fit_transform(x_train_prev).toarray()

x_train_prev=x_train_prev[:,1:]

onehotencoder_h_prev_test = OneHotEncoder(categorical_features=[-2])
x_test_prev = onehotencoder_h_prev_test.fit_transform(x_test_prev).toarray()
x_test_prev=x_test_prev[:,1:]

scaled_x_train_prev=scalar.fit_transform(x_train_prev)
scaled_x_test_prev=scalar.transform(x_test_prev)




#accuracy = 99.96043
from keras.layers import LeakyReLU
ANN_leaky_relu = Sequential()
# The Input Layer :
ANN_leaky_relu.add(Dense(128, kernel_initializer='normal',input_dim = scaled_x_train_prev.shape[1]))
ANN_leaky_relu.add(LeakyReLU(alpha=0.05))
# The Hidden Layers :
ANN_leaky_relu.add(Dropout(0.2))
ANN_leaky_relu.add(Dense(256, kernel_initializer='normal'))
ANN_leaky_relu.add(LeakyReLU(alpha=0.05))

ANN_leaky_relu.add(Dense(256, kernel_initializer='normal'))
ANN_leaky_relu.add(LeakyReLU(alpha=0.05))
ANN_leaky_relu.add(Dropout(0.2))

ANN_leaky_relu.add(Dense(256, kernel_initializer='normal'))
ANN_leaky_relu.add(LeakyReLU(alpha=0.05))
ANN_leaky_relu.add(Dropout(0.2))

# The Output Layer :
ANN_leaky_relu.add(Dense(1, kernel_initializer='normal',activation='linear'))
# Compile the network :
ANN_leaky_relu.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
ANN_leaky_relu.fit(scaled_x_train_prev, y_train, epochs=40, batch_size=32, validation_split = 0.1)

ann_leaky_pred = ANN_leaky_relu.predict(scaled_x_test_prev)
ann_leaky_pred_list =[ float(ann_leaky_pred[i]) for i in range(ann_leaky_pred.shape[0])]

joblib.dump(ANN_leaky_relu,'acc_99.96043.pkl')

submission=pd.DataFrame()
submission['date_time']=test_df['date_time']
submission['traffic_volume']=ann_leaky_pred_list
submission.to_csv('ann_leaky_relu_submission.csv',index=False)








final = []
for i in tqdm(range(len(x_test))):
    final.append((ann_best[i]+random_pred_int[i]+ann_leaky_pred_list[i])/3)
    
submission=pd.DataFrame()
submission['date_time']=test_df['date_time']
submission['traffic_volume']=final
#accuracy: 99.96292
submission.to_csv('final.csv',index=False)



