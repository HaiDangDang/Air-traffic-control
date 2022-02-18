import kalman
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import lightgbm as lgb
from mpl_toolkits.basemap import Basemap
from collections import Counter
from math import sin, cos, sqrt, atan2, radians
from mlat import geodesy
from sensors_use_train import sensor_use, found_same_train_test_diff_height, sensor_use_test

from mlat import geodesy, constants

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error

import joblib


#train = pd.read_pickle("train/train_master_fix_index.pkl")
train = pd.read_pickle("test/test.pkl")
sensors = pd.read_csv("train/sensors.csv")
sensors_test = pd.read_csv("test/sensors.csv")
sensors = pd.read_csv("test/sensors.csv")
x = np.invert(pd.isna(train["latitude"]))
train = train.loc[pd.isna(train["latitude"]),:]
sensor_same = []
for j,sens in sensors_test.iterrows():
    if sens["serial"] in sensor_use_test:
        for i, sens_2 in sensors.iterrows():
            if sens["latitude"] == sens_2["latitude"]:
                if sens["longitude"] == sens_2["longitude"]:
                    sensor_same.append([sens["serial"], sens_2["serial"]])
                    break
sensor_same = np.array(sensor_same)
sensor_same.shape

#train = pd.read_pickle("tmp/tmp.pkl")
# tdoa_df = pd.DataFrame(columns=["serial_1", "time_diff_1", "signal_strength_1", "x_1", "y_1", "z_1", " type_1", "d_aircraft_1",
#                                 "serial_2", "time_diff_2", "signal_strength_2", "x_2", "y_2", "z_2", " type_2", "d_aircraft_2",
#                                 "TDOA", "distance_diff", "distance_signal_real_diff", "distance_signal_diff",
#                                  'x', 'y', 'z','baroAltitude'])

columns=["serial_1", "time_diff_1", "signal_strength_1", "x_1", "y_1", "z_1", "height_1", " type_1", "d_aircraft_1",
        "serial_2", "time_diff_2", "signal_strength_2", "x_2", "y_2", "z_2", "height_2", " type_2", "d_aircraft_2",
        "TDOA", "distance_diff", "distance_signal_real_diff", "distance_signal_diff",
         'x', 'y', 'z','baroAltitude', "id"]
tdoa_df = []

sensors_dict = {}
sensors_dict[474]
for s in sensor_use_test:
    sensor = sensors[sensors["serial"] == s]

    # if s in sensor_same[:, 0]:
    #     s = sensor_same[sensor_same[:,0] == s, 1][0]
    # else:
    #     s = s + 1000
    llh = sensor[["latitude", "longitude", "height"]].values[0]
    x, y, z = geodesy.llh2ecef(llh)
    type = 0
    if sensor["type"].values[0] == "GRX1090":
        type = 1

    sensors_dict[s] = {"xyz": [x, y, z],
                       "type": type,
                       "height": sensor["height"].values[0],
                       "lati_long": [sensor["latitude"].values[0],sensor["longitude"].values[0]]}


def pasr_data(s_1, s_2, measure_1, measure_2, aircraft, id):
    dict_s1 = sensors_dict[s_1]
    dict_s2 = sensors_dict[s_2]

    timestamp_1 = measure_1[1] * 10**-9
    timestamp_2 = measure_2[1] * 10**-9

    signal_strength_1 = measure_1[2]
    signal_strength_2 = measure_2[2]

    time_diff_1 = timestamp_1 - aircraft["timeAtServer"]
    time_diff_2 = timestamp_2 - aircraft["timeAtServer"]

    TDOA = timestamp_1 - timestamp_2
    assert TDOA >= 0

    distance_signal_diff = TDOA * constants.Cair

    # d_aircraft_1 = geodesy.ecef_distance(aircraft["xyz"], dict_s1["xyz"])
    # d_aircraft_2 = geodesy.ecef_distance(aircraft["xyz"], dict_s2["xyz"])

    #distance_signal_real_diff = np.abs(d_aircraft_1 - d_aircraft_2)

    distance_diff = geodesy.ecef_distance(dict_s1["xyz"], dict_s2["xyz"])

    tdoa_df.append([s_1, time_diff_1, signal_strength_1, dict_s1["xyz"][0], dict_s1["xyz"][1], dict_s1["xyz"][2], dict_s1["height"],
                    dict_s1["type"], -1,
                    s_2, time_diff_2, signal_strength_2, dict_s2["xyz"][0], dict_s2["xyz"][1], dict_s2["xyz"][2], dict_s2["height"],
                    dict_s2["type"], -1,
                    TDOA, distance_diff, -1, distance_signal_diff,
                    aircraft["xyz"][0], aircraft["xyz"][1], aircraft["xyz"][2], aircraft["baroAltitude"], id])

#

for i, item in train.iterrows():
    measur = item["measurements"]

    if i % 10000 == 0:
        print(i)
    measur = item["measurements"]
    pairs = []
    llh = item[["latitude", "longitude", "geoAltitude"]].values
    x, y, z = geodesy.llh2ecef(llh)
    aircraft = {"xyz":  [x, y, z],
                "baroAltitude": item["baroAltitude"],
                "timeAtServer": item["timeAtServer"]}

    assert len(measur) >= 2
    for j in range(len(measur) - 1):
        for _j in range(j + 1, len(measur)):
            s_1 = measur[j][0]
            s_2 = measur[_j][0]
            if s_1 in sensor_same[:, 0]:
                s_1 = sensor_same[sensor_same[:, 0] == s_1, 1][0]
            else:
                s_1 = s_1 + 1000

            if s_2 in sensor_same[:, 0]:
                s_2 = sensor_same[sensor_same[:, 0] == s_2, 1][0]
            else:
                s_2 = s_2 + 1000

            if measur[j][1] > measur[_j][1]:
                pasr_data(s_1, s_2, measur[j], measur[_j], aircraft, int(i +14000000))
            else:
                pasr_data(s_2, s_1, measur[_j], measur[j], aircraft, int(i+14000000))
tdoa_df = pd.DataFrame(tdoa_df, columns=columns)
tdoa_df.shape
tdoa_df.to_pickle(f"train/TDOA/test.pkl")
tdoa_df = []
print()
print("SAVE FILE")
print()
#
#
#

x_train = []
x_test = []
y_train = []
y_test = []
sc = StandardScaler()


for i in np.arange(14) + 1:

    x = pd.read_pickle(f"train/TDOA/{i}.pkl")
    x = x[x["distance_signal_diff"] <= 1000000]
    idx = np.random.choice(np.arange(len(x)),  int(len(x)*0.55), replace=False)
    x = x.iloc[idx,:]
    x["distance_signal_real_diff"] = x["d_aircraft_1"] - x["d_aircraft_2"]
    x[["x", "y", "z",
        "x_1", "y_1", "z_1",
           "x_2", "y_2", "z_2"
           , "distance_diff", "distance_signal_diff", "distance_signal_real_diff", "d_aircraft_1", "d_aircraft_2"]] = \
        x[["x", "y", "z",
            "x_1", "y_1", "z_1",
               "x_2", "y_2", "z_2"
               , "distance_diff", "distance_signal_diff", "distance_signal_real_diff", "d_aircraft_1", "d_aircraft_2"]] / 1000
    llh = geodesy.ecef2llh([x['x'].values * 1000, x['y'].values * 1000, x['z'].values * 1000])
    y = llh[1]
    x = x.drop(
        ["serial_1", "serial_2", "x", "y", "z", "d_aircraft_1",  "d_aircraft_2", "distance_signal_diff", "id", "z_1", "z_2"], axis=1)
    features = x.keys()

    x = x.values
    x[0]
    x_train_t, x_test_t, y_train_t, y_test_t = train_test_split(x, y, test_size=0.1, random_state=10)

    if len(x_train) == 0:
        # x_train_t = sc.fit_transform(x_train_t)
        # x_test_t = sc.transform(x_test_t)
        x_train, x_test, y_train, y_test = x_train_t, x_test_t, y_train_t, y_test_t
    else:
        # x_train_t = sc.transform(x_train_t)
        # x_test_t = sc.transform(x_test_t)
        x_train = np.concatenate((x_train, x_train_t))
        x_test = np.concatenate((x_test, x_test_t))
        y_train = np.concatenate((y_train, y_train_t))
        y_test = np.concatenate((y_test, y_test_t))
    print(i)
print(x_train.shape)
a = x_train[:,12]
a[0]
np.min(a)
# x = train["distance_signal_real_diff"] /1000
# fig = plt.figure
# fig(num=None, figsize=(30, 6), dpi=80, facecolor='w', edgecolor='k')
# ax = plt.plot(x, c="r")
# plt.show()

# del train
# del y_data
# del x
lgb_params1 = {}
lgb_params1['application'] = 'regression'
lgb_params1['boosting'] = 'gbdt'
lgb_params1['learning_rate'] = 0.1
lgb_params1['num_leaves'] = 90
lgb_params1['min_data_in_leaf'] = 20

lgb_params1['min_sum_hessian_in_leaf'] = 2e-2
lgb_params1['min_gain_to_split'] = 0
lgb_params1['bagging_fraction'] = 0.9
lgb_params1['feature_fraction'] = 0.8
lgb_params1['bagging_freq'] = 153
lgb_params1['metric'] = 'rmse'
lgb_params1['num_threads'] = 8
# lgb_params1['n_estimators'] = 1000
# lgb_params1['colsample_bytree'] = 0.8
lgb_params1['max_depth'] = 9
#lgb_params1['min_child_weight'] = 300
lgb_params1['device'] = "gpu"
lgb_params1['gpu_platform_id'] = 0
lgb_params1['gpu_device_id'] = 0

# lgb_params1['max_bin'] = 255

#
#
# lgb_params1 = {}
# lgb_params1['application'] = 'regression'
# lgb_params1['boosting'] = 'gbdt'
# lgb_params1['learning_rate'] = 0.1
# lgb_params1['num_leaves'] = 98
# lgb_params1['max_depth'] = 8
#
# lgb_params1['min_sum_hessian_in_leaf'] = 2e-2
# lgb_params1['min_gain_to_split'] = 0
# lgb_params1['bagging_fraction'] = 0.9
# lgb_params1['bagging_freq'] = 153
# lgb_params1['num_iterations'] = 200
# lgb_params1['max_bin'] = 350
# # lgb_params1['lambda_l1'] = 0.3
# # lgb_params1['lambda_l2'] = 0.4
# #
# # lambda_l1
# lgb_params1['feature_fraction'] = 0.8
# lgb_params1['metric'] = 'rmse'
# #lgb_params1['num_threads'] = 14
# lgb_params1['device'] = "gpu"
# lgb_params1['gpu_platform_id'] = 0
# lgb_params1['gpu_device_id'] = 0
#


# lgb_params1['n_estimators'] = 1000
# lgb_params1['colsample_bytree'] = 0.8

x_val, x_test, y_val, y_test = train_test_split(x_test,y_test,test_size = 0.2, random_state = 30)


d_valid = lgb.Dataset(x_val, y_val)
d_train = lgb.Dataset(x_train, y_train)

watchlist = [d_train, d_valid]

lgb_model_ = lgb.train(lgb_params1, train_set=d_train, num_boost_round=4500,
                          valid_sets=watchlist, verbose_eval=100, early_stopping_rounds = 200,feature_name=list(features))
    y_pred = lgb_model_.predict(x_test[:10000])
y_pred = lgb_model_.predict(x_test[:10000])
lgb_model2 = lgb
x_test.shape
#lgb_model_.save_model("model/distance_1_gbdt.txt")
# lgb_model_.save_model("model/distance_diff.txt")
# #
a = (y_test - y_pred)
np.std(a)
print(np.sqrt(mean_squared_error(y_test[:10000],y_pred)))
# a = abs(y_test - y_pred)
# np.mean(a)
# #
# # sns.distplot(a)
# plt.plot(np.sort(a))
# plt.show()
# x = np.sort(a)
# plt.plot(np.sort(a))
#
# plt.show()
# y_test[:10]
# # np.std(x)
# # import pickle
# # pickle.dump(sc, open('model/distance_diff_SC.pkl','wb'))
# x["x"]

# a = x["d_aircraft_1"]
# np.std(a)
# b = x["time_diff_2"]
# c = x["TDOA"]
# np.max(c) * constants.Cair
# np.max(b)
lgb.plot_importance(lgb_model_)
plt.show()
#
# x.shape
# x[0:2]
# x_train_t.shape
# a, b = np.split(x, [int(.2*len(x))])
# b.shape
# sc = StandardScaler()
# data = [[300, 500], [10000, -9], [20000, -1000000]]
# sc.fit_transform(data)
# print(sc.fit(data))
# a = [[300,100,0,200,-600], [1000,2452,6000,-333]]
# b= np.array(a)
# c = sc.fit_transform(b)
gbm_lat = lgb.Booster(model_file='model/lgb_lati_v2.txt')
gbm_long = lgb.Booster(model_file='model/lgb_long_v2.txt')
gbm_tdoa = lgb.Booster(model_file='model/lgb_tdoa_v2.txt')
gbm_height = lgb.Booster(model_file='model/lgb_height_v2.txt')

# gbm_tdoa.save_model("model/lgb_tdoa_v3.pkl", num_iteration=22000, start_iteration=16000)

x = pd.read_pickle(f"train/TDOA/test.pkl")

# y = d.values
tmp = x.drop(
    ["x", "y", "z", "d_aircraft_1", "d_aircraft_2", "distance_signal_real_diff"], axis=1)
d_diff = gbm_tdoa.predict(tmp)
x["distance_signal_diff"] = d_diff

tmp = x.drop(
    ["x", "y", "z", "d_aircraft_1", "d_aircraft_2", "distance_signal_real_diff"], axis=1)
latis = gbm_lat.predict(tmp)

longs = gbm_long.predict(tmp)
heights = gbm_height.predict(tmp)
x['x'] = latis
x['y'] = longs
x['z'] = heights

test = pd.read_pickle("test/test.pkl")
test = test[pd.isna(test["latitude"])]
submit = pd.read_csv('test/submit_sample.csv')
latis_1 = []
longs_1 = []
heights_1 = []
for j, item in test.iterrows():
    a = x[x["id"] == j + 14000000]
    if len(a) == 0:
        print("FAIL")
        break
    latis_1.append(np.mean(a['x'].values))
    longs_1.append(np.mean(a['y'].values))
    heights_1.append(np.mean(a['z'].values))

submit["latitude"] = latis_1
submit["longitude"] = longs_1
submit["geoAltitude"] = heights_1
submit.to_csv("submit/v_1.csv", index=False)