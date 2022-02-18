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
from sensors_use_train import sensor_use

from mlat import geodesy, constants
import torch
from sklearn.metrics import mean_squared_error
import time
from sys import getsizeof
import joblib
import pickle
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor



x_train = []
x_test = []
y_train = []
y_test = []
x_val = []
y_val = []
# gbm_lat = lgb.Booster(model_file='model/lgb_lati_v2.txt')
# gbm_long = lgb.Booster(model_file='model/lgb_long.pkl')
# gbm_tdoa = lgb.Booster(model_file='model/lgb_tdoa_v3.pkl')
# gbm_tdoa.save_model("model/lgb_tdoa_v3.pkl", num_iteration=22000, start_iteration=16000)

for i in [1, 2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15]:
    x = pd.read_pickle(f"train/TDOA/{i}.pkl")

    x = x[x["distance_signal_diff"] <= 1000000]
    llh = geodesy.ecef2llh([x["x"].values, x["y"].values, x["z"].values])
    x["x"] = llh[0]
    x["y"] = llh[1]
    x["z"] = llh[2]

    y = x.loc[:, "z"].values
    d = x["d_aircraft_1"] - x["d_aircraft_2"]
    x["distance_signal_diff"] = d
    # y = d.values
    x = x.drop(
        ["x", "y", "z", "d_aircraft_1", "d_aircraft_2", "distance_signal_real_diff"], axis=1)
    features = x.keys()

    x = x.values
    if i == 15:
        x_train_t, x_test_t, y_train_t, y_test_t = train_test_split(x, y, test_size=0.0001, random_state=10)
    else:
        x_train_t, x_test_t, y_train_t, y_test_t = train_test_split(x, y, test_size=0.05, random_state=10)

    if len(x_train) == 0:
        x_train, x_val, y_train, y_val = x_train_t, x_test_t, y_train_t, y_test_t
    else:
        x_train = np.concatenate((x_train, x_train_t))
        x_val = np.concatenate((x_val, x_test_t))
        y_train = np.concatenate((y_train, y_train_t))
        y_val = np.concatenate((y_val, y_test_t))
    print(x_train.shape)
    print(x_val.shape)
lgb_params1 = {}
lgb_params1['application'] = 'regression'
lgb_params1['boosting'] = 'gbdt'
lgb_params1['learning_rate'] = 0.095
lgb_params1['num_leaves'] = 112
lgb_params1['min_data_in_leaf'] = 150

lgb_params1['min_sum_hessian_in_leaf'] = 2e-2
lgb_params1['min_gain_to_split'] = 0
lgb_params1['bagging_fraction'] = 0.7
lgb_params1['feature_fraction'] = 0.7
lgb_params1['bagging_freq'] = 120
lgb_params1['metric'] = 'rmse'
lgb_params1['num_threads'] = 15
# lgb_params1['n_estimators'] = 1000
# lgb_params1['colsample_bytree'] = 0.8
lgb_params1['max_depth'] = 10
#lgb_params1['min_child_weight'] = 300
lgb_params1['device'] = "gpu"
lgb_params1['gpu_platform_id'] = 0
lgb_params1['gpu_device_id'] = 0

d_valid = lgb.Dataset(x_val, y_val)
d_train = lgb.Dataset(x_train, y_train)
watchlist = [d_train, d_valid]

lgb_model_x = lgb.train(lgb_params1, train_set=d_train, num_boost_round=10000,
                          valid_sets=watchlist, verbose_eval=100,
                       early_stopping_rounds = 200)

lgb_model_x.save_model('model/lgb_height_v2.txt', num_iteration=lgb_model_x.best_iteration)
lgb_model_x_2 = lgb_model_x
lgb_model_x.save_model('model/lgb_lat.pkl', num_iteration=lgb_model_x.best_iteration)
lgb.plot_importance(lgb_model_x)
plt.show()
x_test.shape
y_pred = lgb_model_x.predict(x_test)
a = y_pred - y_test
b= y_pred/y_test

mean_squared_error(y_test/1000, y_pred/1000)

d_valid = lgb.Dataset(x_val, y_val[:,1])
d_train = lgb.Dataset(x_train, y_train[:,1])

watchlist = [d_train, d_valid]

lgb_model_y = lgb.train(lgb_params1, train_set=d_train, num_boost_round=500,
                          valid_sets=watchlist, verbose_eval=100, early_stopping_rounds = 200,
                        feature_name=list(features))
lgb_model_y.save_model('model/lgb_tdoa_v2.pkl', num_iteration=lgb_model_y.best_iteration)

gbm_lat = lgb.Booster(model_file='model/lgb_lat.pkl')
gbm_long = lgb.Booster(model_file='model/lgb_long.pkl')
gbm_tdoa = lgb.Booster(model_file='model/lgb_tdoa.pkl')

height_grip = 11582
max_delta = 0.5
max_delta_l = 0.15
max_delta_lo = 0.2
0.02/6/6
max_grip = 8
a = geodesy.llh2ecef([0, 0, 0])
b = geodesy.llh2ecef([0.0005555555555555556, 0, 0])
geodesy.ecef_distance(a, b)
sensor = pd.read_csv("train/sensors.csv")
drops = []
# train_v1 = pd.read_csv("train/training_1_category_1.csv")
# train_v1.iloc[0,:]
delta_erros = 0.01
range_tdoa = np.arange(0.85, 1.14, delta_erros)


for i in np.arange(1) +1:
    i = 1
    x = pd.read_pickle(f"train/TDOA/{i}.pkl")
    x = x[x["distance_signal_diff"] <= 1000000]

    llh = geodesy.ecef2llh([x["x"].values, x["y"].values, x["z"].values])
    tmp = x.drop(
        ["serial_1", "serial_2", "x", "y", "z", "d_aircraft_1", "d_aircraft_2", "distance_signal_real_diff"], axis=1)


    x.keys()
    x["lati_pred"] = gbm_lat.predict(tmp)
    x["long_pred"] = gbm_long.predict(tmp)
    x["diff_tdoa"] = gbm_tdoa.predict(tmp)

    x["latitude"] = llh[0]
    x["longitude"] = llh[1]
    x["geoAltitude"] = llh[2]
    x["distance_signal_real_diff"] = x["d_aircraft_1"] - x["d_aircraft_2"]

    llh1 = geodesy.ecef2llh([x["x_1"].values, x["y_1"].values, x["z_1"].values])
    llh2 = geodesy.ecef2llh([x["x_2"].values, x["y_2"].values, x["z_2"].values])
    x["lati_1"] = x["lati_pred"] - llh1[0]
    x["lati_2"] = x["lati_pred"] - llh2[0]
    x["long_1"] = x["long_pred"] - llh1[1]
    x["long_2"] = x["long_pred"] - llh1[1]

    # mean_squared_error(x["distance_signal_real_diff"]/1000, x["distance_signal_real_diff"]/1000)

    del tmp

    xyz_s = x[["x_1", "y_1", "z_1", "x_2", "y_2", "z_2"]].values
    xyz_s = np.expand_dims(xyz_s, -1)
    #xyz_s = np.tile(xyz_s, (1, max_grip))

    baro = x["baroAltitude"].values
    baro = np.expand_dims(baro, -1)
    #baro = np.tile(baro, (1, max_grip))

    x["distance_diff"] = x["distance_diff"]

    # x["latitude"] = (x["latitude"] - x["lati_pred"])/max_delta
    # x["longitude"] = (x["longitude"] - x["long_pred"])/max_delta
    x["distance_signal_diff"] = x["diff_tdoa"]


    train = x[["latitude", "longitude", "geoAltitude",
                  "time_diff_1", "signal_strength_1", " type_1", "height_1", "lati_1", "long_1",
                  "time_diff_2", "signal_strength_2", " type_2", "height_2", "lati_2", "long_2",
                  "distance_diff", "distance_signal_diff", "baroAltitude"]].values
    counter = 0
    c_time = time.time()
    lati_s = []
    longti_s = []
    for j in range(max_grip):
        a = x["lati_pred"] - max_delta_l + (max_delta_l * 2 *j/ (max_grip -1))
        a = np.expand_dims(a, -1)

        b = x["long_pred"] - max_delta_lo + (max_delta_lo * 2 * j/ (max_grip -1))
        b = np.expand_dims(b, -1)
        if len(lati_s) == 0:
            lati_s = a
            longti_s = b
        else:
            lati_s = np.concatenate((lati_s, a), axis=-1)
            longti_s = np.concatenate((longti_s, b), axis=-1)
        del a,b

    # lati_s = []
    # longti_s = []
    # for j, item in x.iterrows():
    #     counter += 1
    #     lati_grip = np.linspace(item["lati_pred"] - max_delta, item["lati_pred"] + max_delta, max_grip)
    #     long_grip = np.linspace(item["long_pred"] - max_delta, item["long_pred"] + max_delta, max_grip)
    #     break
    #     lati_s.append(lati_grip)
    #     longti_s.append(long_grip)
    #     if counter % 100000 == 0:
    #         print(time.time() - c_time)
    #         c_time = time.time()
    #
    # lati_s = np.array(lati_s)
    # longti_s = np.array(longti_s)
    #break
    time_diff = x["diff_tdoa"].values
    time_diff = np.expand_dims(time_diff, -1)
    #time_diff = np.tile(time_diff, (1, max_grip))

    del x
    counter = 0
    for j in range(max_grip):
        lati_small = lati_s[:, j]
        lati_small = np.expand_dims(lati_small,-1)
        #lati_small = np.tile(lati_small,(1,max_grip))

        x_grid, y_grid, z_grid = geodesy.llh2ecef([lati_small, longti_s, baro])

        d_1 = geodesy.ecef_distance([x_grid, y_grid, z_grid], [xyz_s[:,0,:], xyz_s[:,1,:], xyz_s[:,2,:]])
        d_2 = geodesy.ecef_distance([x_grid, y_grid, z_grid], [xyz_s[:,3,:], xyz_s[:,4,:], xyz_s[:,5,:]])
        diff_tdoa = d_1 - d_2
        diff_tdoa = diff_tdoa

        # for tdoa_per in range_tdoa:
        #     for idx in range(max_grip):
        #         if tdoa_per < 1:
        #             location = (diff_tdoa[:,idx] >= x[f"{tdoa_per}"].values) & \
        #                        (diff_tdoa[:,idx] <= tdoa_per + delta_erros/2)
        #         else:
        #             location = (diff_tdoa[:,idx] <= x[f"{tdoa_per}"].values) & \
        #                        (diff_tdoa[:,idx] >= tdoa_per - delta_erros/2)
        #         x.loc[location, f"{tdoa_per}"] = diff_tdoa[location,idx]
        #         x.loc[location, f"x_{tdoa_per}"] = lati_small[location,0]
        train = np.concatenate((train, diff_tdoa), axis=1)
        counter += 1
        print(counter)

    np.save(f"train/TDOA/layer_2/{333}.npy", train)


    # np.save(f"train/TDOA/layer_2/{i}.npy", train)
    # del train
    print("SAVE")

# #
# item = x.loc[0,:]
#
# lati_grip = torch.linspace(item["lati_pred"] -0.3, item["lati_pred"] +0.3, 16).cuda()
# long_grip = torch.linspace(item["long_pred"] -0.5, item["long_pred"] +0.5, 16).cuda()
#
# lati_grip, long_grip = torch.meshgrid([lati_grip, long_grip])
# x_grid, y_grid, z_grid = geodesy.llh2ecef_torch([lati_grip, long_grip, item["baroAltitude"]])
#
# s_1 = item[["x_1", "y_1", "z_1"]].values
# s_2 = item[["x_2", "y_2", "z_2"]].values
# d_1 = geodesy.ecef_distance_torch([x_grid, y_grid, z_grid], s_1)
# d_2 = geodesy.ecef_distance_torch([x_grid, y_grid, z_grid], s_2)
# d = d_1 - d_2
# location = (item["distance_signal_real_diff"] * 0.8 < d) & \
#            (d < item["distance_signal_real_diff"] *1.2)
# torch.sum(location)
# d = d/1000
# d[2]
# a = x["diff_tdoa"]/x["distance_signal_real_diff"]
# np.sum((abs(a) > 0.8) & (abs(a) < 1.2)) / len(a)
#
a = geodesy.llh2ecef([0, 0, 0])
b = geodesy.llh2ecef([0.0476 * 0.6, 0, 0])
geodesy.ecef_distance(a, b)

a = [[2,3],[3,3]]
a = np.array(a)
b = np.array([1,2])
a > b
a.shape
b.shape
a = np.array(x[f"{tdoa_per}"].values[:2])
a = np.expand_dims(a,axis=-1)
b = np.array(diff_tdoa[:2,:3])
b > a
a < b