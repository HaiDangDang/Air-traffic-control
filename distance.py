import lightgbm as lgb
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
from sensors_use_train import sensor_use
from sklearn.multioutput import MultiOutputRegressor
import tqdm
from sklearn.metrics import mean_squared_error
from math import sin, cos, sqrt, atan2, radians
import matplotlib.pyplot as plt
def convert_longLati_km(longs, latis, geo, center_long, center_lati, height):
    R = 6373.0
    longs = np.radians(longs)
    latis = np.radians(latis)
    center_long = np.radians(center_long)
    center_lati = np.radians(center_lati)
    dlon = center_long - longs
    dlat = center_lati - latis
    a = np.sin(dlat / 2) ** 2 + np.cos(latis) * np.cos(center_lati) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c
    diff_height = (geo - height)/1000
    distance_height = np.sqrt(distance ** 2 + diff_height ** 2)
    return distance, distance_height

def RMSE_2D(X,X_pred):

    def vectorize_radians(x):
        return np.vectorize(radians)(x)

    X = vectorize_radians(X)
    X_pred = vectorize_radians(X_pred)

    R = 6373000

    lat1 = X[:,0]
    lat2 = X_pred[:,0]
    dlat = X[:,0]-X_pred[:,0]
    dlon = X[:,1]-X_pred[:,1]

    a = np.power(np.sin(dlat / 2), 2) + np.cos(lat1) * np.cos(lat2) * np.power(np.sin(dlon / 2), 2)
    c = 2 * np.arctan2(np.sqrt(a),np.sqrt(1-a))

    vector_errors = R * c

    sum_errors = np.sum(np.power(vector_errors, 2))
    return np.sqrt(sum_errors/len(vector_errors))

sensors = pd.read_csv("train/sensors.csv")

train_data = []
for sensor in sensor_use:
    sensor_data = pd.read_pickle(f"train/sensors/sensor_{sensor}.pkl")
    sensor_serial = sensors[sensors["serial"] == sensor]
    distance, distance_height = convert_longLati_km(sensor_data["longitude"].values,
                                                    sensor_data["latitude"].values,
                                                    sensor_data["geoAltitude"].values,
                                                    sensor_serial["longitude"].values,
                                                    sensor_serial["latitude"].values,
                                                    sensor_serial["height"].values)
    delta_time = sensor_data["timestamp"].values * 10 ** -9 - sensor_data["timeAtServer"].values
    sensor_data["delta_time"] = delta_time
    sensor_data["delta_time_mean"] = np.mean(delta_time)
    sensor_data["delta_time_std"] = np.std(delta_time)

    signalstrength_std = np.std(sensor_data["signalstrength"])
    signalstrength_mean = np.mean(sensor_data["signalstrength"])
    sensor_data["mean_signalstrength"] = signalstrength_mean
    sensor_data["std_signalstrength"] = signalstrength_std
    sensor_data["68_signalstrength"] = signalstrength_mean + signalstrength_std
    sensor_data["95_signalstrength"] = signalstrength_mean + 2*signalstrength_std
    sensor_data["98_signalstrength"] = signalstrength_mean + 3*signalstrength_std



    sensor_data["distance"] = distance
    distance_std = np.std(distance)
    distnace_mean = np.mean(distance)
    sensor_data["mean_distance"] = distnace_mean
    sensor_data["std_distance"] = distance_std
    sensor_data["68_distance"] = distnace_mean + distance_std
    sensor_data["95_distance"] = distnace_mean + 2*distance_std
    sensor_data["98_distance"] = distnace_mean + 3*distance_std

    sensor_data["distance_height"] = distance_height

    type = 0
    if sensor_serial["type"].values[0] == "GRX1090":
        type = 1
    sensor_data["type"] = type
    sensor_data["height"] = sensor_serial["height"].values[0]


    size_random = int(len(sensor_data) * 0.24)
    random_index = np.random.randint(0, len(sensor_data) - 2,size_random)
    sensor_data = sensor_data.iloc[random_index,:]
    train_data.append(sensor_data)
    print(len(train_data))
train_data.keys()
# sensor_use = list(sensors_data.keys())
#
#
# idx = []
# for i,s in sensors.iterrows():
#     if s["serial"] in sensor_use:
#        idx.append(i)
#
# sensors_data[463.0]
# sensors = sensors.loc[idx,:]
# train_data = []
# for key in sensor_use:
#     value = sensors_data[key]
#     sensor = sensors[sensors["serial"] == key]
#     distance, distance_height = convert_longLati_km(value["longitude"].values,
#                                                     value["latitude"].values,
#                                                     value["geoAltitude"].values,
#                                                     sensor["longitude"].values,
#                                                     sensor["latitude"].values,
#                                                     sensor["height"].values)
#     sensors_data[key]["distance"] = distance
#     sensors_data[key]["distance_height"] = distance_height
#     signal_error = sensors_data[key]["signalstrength"]/distance
#     sensors_data[key]["mean"] = np.mean(signal_error)
#     sensors_data[key]["std"] = np.std(signal_error)
#     sensors_data[key]["median"] = np.median(signal_error)
#
#     type = 0
#     if sensor["type"].values[0] == "GRX1090":
#         type = 1
#         print(type)
#     sensors_data[key]["type"] = type
#     sensors_data[key]["height"] = sensor["height"].values[0]
#     train_data.append(sensors_data[key])
train_data.shape
train_data = pd.concat(train_data)
train_data.keys()
features = ['baroAltitude', 'signalstrength', 'delta_time', 'delta_time_mean',
       'delta_time_std', 'mean_signalstrength', 'std_signalstrength',
       '68_signalstrength', '95_signalstrength', '98_signalstrength',
                      'mean_distance', 'std_distance', '68_distance',
       '95_distance', '98_distance', 'type', 'height']
train_X = train_data[features].values
train_y = train_data.loc[:,"distance"].values
train_X[9]
x_train, x_test, y_train, y_test = train_test_split(train_X,train_y,test_size = 0.1, random_state = 0)
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

lgb_params1 = {}
lgb_params1['application'] = 'regression'
lgb_params1['boosting'] = 'gbdt'
lgb_params1['learning_rate'] = 0.3
lgb_params1['num_leaves'] = 50
lgb_params1['min_sum_hessian_in_leaf'] = 2e-2
lgb_params1['min_gain_to_split'] = 0
lgb_params1['bagging_fraction'] = 0.9
lgb_params1['feature_fraction'] = 0.8
lgb_params1['metric'] = 'rmse'
lgb_params1['num_threads'] = 14
# lgb_params1['n_estimators'] = 1000
# lgb_params1['colsample_bytree'] = 0.8
lgb_params1['max_depth'] = 8
#lgb_params1['min_child_weight'] = 300

x_val, x_test, y_val, y_test = train_test_split(x_test,y_test,test_size = 0.5, random_state = 0)


d_valid = lgb.Dataset(x_val, y_val)
d_train = lgb.Dataset(x_train, y_train)

watchlist = [d_train, d_valid]
lgb_model_ = lgb.train(lgb_params1, train_set=d_train, num_boost_round=100,
                          valid_sets=watchlist, verbose_eval=100, early_stopping_rounds = 100,feature_name=features)
lgb_model_.save_model('model.txt')

lgb.plot_importance(lgb_model_)
plt.show()
y_pred = lgb_model_.predict(x_test)
x = np.abs(y_test - y_pred)
np.mean(x)

print(y_pred[:20])
print(y_test[:20])

mean_squared_error(y_test,y_pred)
import sys

icao = int(3, 16)

