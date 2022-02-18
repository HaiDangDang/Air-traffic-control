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
# gbm_lat = lgb.Booster(model_file='model/lgb_lat.pkl')
# gbm_long = lgb.Booster(model_file='model/lgb_long.pkl')
# gbm_tdoa = lgb.Booster(model_file='model/lgb_tdoa.pkl')
# #
# for i in np.arange(14) + 1:
#     x = pd.read_pickle(f"train/TDOA/{i}.pkl")
#     y = pd.read_pickle(f"train/TDOA/layer_2/{i}.pkl")
#
#     x["lati_pred"] = y["lati_pred"]
#     x["long_pred"] = y["long_pred"]
#     x["diff_tdoa"] = y["diff_tdoa"]
#
#     llh = geodesy.ecef2llh([x["x"].values, x["y"].values, x["z"].values])
#     x["latitude"] = llh[0]
#     x["longitude"] = llh[1]
#     x["geoAltitude"] = llh[2]
#     idx = x["serial_1"] > x["serial_2"]
#     tmp = x.loc[idx, ['serial_2', 'time_diff_2',
#                       'signal_strength_2', 'x_2', 'y_2', 'z_2', 'height_2', ' type_2',
#                       'd_aircraft_2']].values
#
#     x.loc[idx, ['serial_2', 'time_diff_2',
#                 'signal_strength_2', 'x_2', 'y_2', 'z_2', 'height_2', ' type_2',
#                 'd_aircraft_2']] = x.loc[idx, ['serial_1', 'time_diff_1',
#                                                'signal_strength_1', 'x_1', 'y_1', 'z_1',
#                                                'height_1', ' type_1', 'd_aircraft_1']].values
#
#     x.loc[idx, ['serial_1', 'time_diff_1', 'signal_strength_1', 'x_1', 'y_1', 'z_1',
#                 'height_1', ' type_1', 'd_aircraft_1']] = tmp
#
#     x["distance_signal_real_diff"] = x["d_aircraft_1"] - x["d_aircraft_2"]
#     x["TDOA"] = x["time_diff_1"] - x["time_diff_2"]
#     x["distance_signal_diff"] = x["TDOA"] * constants.Cair
#     x.to_pickle(f"train/TDOA/layer_2/{i}.pkl")


# gbm_lat = lgb.Booster(model_file='model/lgb_lat.pkl')
# gbm_long = lgb.Booster(model_file='model/lgb_long.pkl')
# gbm_tdoa = lgb.Booster(model_file='model/lgb_tdoa.pkl')


# train_v1 = pd.read_csv("train/training_1_category_1.csv")
# train_v1.iloc[0,:]
delta_error_tdoa = 0.2
delta_error_lati_l = 0.6
delta_error_lati_s = 0.4
delta_error_long_s = 0.6
delta_error_long_l = 0.7

pairs = pd.read_pickle(f"train/TDOA/layer_2/pairs.pkl")
sums = []
k_nn_num = 10
feature_knn = 4
keys_pair = ['serial_1', 'serial_2', 'longitude', 'latitude', 'geoAltitude', 'id',
       'distance_signal_real_diff']
pairs_s = np.load("train/TDOA/layer_2/pairs_s.npy")
store_p = {}
store_p_cuda = {}
for pair in pairs_s:
    pair_data = pairs[pairs["serial_1"] == pair[0]]
    pair_data = pair_data[pair_data["serial_2"] == pair[1]]
    store_p_cuda[f"{pair[0]}_{pair[1]}"] = torch.Tensor(pair_data.values).cuda()
    sums.append(len(pair_data))
del pairs
np.sum(sums)
c_time = time.time()
for i in np.arange(14) + 1:
    i =1
    x = pd.read_pickle(f"train/TDOA/layer_2/{i}.pkl")
    x["diff_tdoa"] = x["diff_tdoa"] * x["distance_signal_real_diff"] / abs(x["distance_signal_real_diff"])
    x.keys()
    main_d = []
    c_time = time.time()
    counter = 0
    check = False
    for pair in pairs_s:
        data = x[x["serial_1"] == pair[0]]
        data = data[data["serial_2"] == pair[1]]
        if len(data) == 0:
            continue
        pair_data = store_p_cuda[f"{pair[0]}_{pair[1]}"]
        data_1 = data[['latitude', 'longitude','geoAltitude',
                       'lati_pred', 'long_pred', 'diff_tdoa', 'baroAltitude','distance_diff',
                       'distance_signal_diff',
                       'time_diff_1', 'signal_strength_1', 'x_1', 'y_1', 'z_1',
                       'height_1', ' type_1',
                        'time_diff_2','signal_strength_2', 'x_2', 'y_2', 'z_2',
                       'height_2', ' type_2']]
        data = data[["lati_pred", "long_pred", "diff_tdoa", "id", "distance_signal_real_diff"]].values
        data_2 = []

        for j in range(len(data)):
            counter += 1
            item = data[j, :]
            counter_loop = 0
            move = 0
            while True:
                knn = pair_data[((pair_data[:, 3] >= item[0]  - delta_error_lati_l) &
                                (pair_data[:, 3] <= item[0] + delta_error_lati_l) &
                                (pair_data[:, 2] >= item[1] - delta_error_long_l)&
                                 (pair_data[:, 2] <= item[1] + delta_error_long_l) &
                                 (pair_data[:, -2] != item[3]))]
                bound_min = np.min([item[2] * (1 - 0.2), item[2] * (1 + 0.2)])
                bound_max = np.max([item[2] * (1 - 0.2), item[2] * (1 + 0.2)])

                knn = knn[((knn[:, -1] >= bound_min) &
                           (knn[:, -1] <= bound_max))]
                # knn = knn[((knn[:, -1] >= item[2] * (1 - 0.2)) &
                #             (knn[:, -1] <= item[2] * (1 + 0.2)))]

                min_len = np.min([len(knn), k_nn_num*feature_knn])

                new_lati = torch.mean(knn[:min_len][:,3]).cpu().numpy()
                new_long = torch.mean(knn[:min_len][:,2]).cpu().numpy()
                if (abs(item[0] - new_lati) <= 0.001) & (abs(item[1] -new_long <= 0.001)):
                    break

                item[0] = new_lati
                data_1.iloc[0, 3] = item[0]

                item[1] = new_long
                data_1.iloc[0, 4] = item[1]
                counter_loop += 1
                if counter_loop == 24:
                    break

            # data_2.append(tmp_store.flatten())
            if counter % 10000 == 0:
                print(counter)
                print(time.time() - c_time)
                print("____________")
                c_time = time.time()

            #data_2.append(tmp_store)
        main_d.append(data_1)
        main_d = pd.concat(main_d)
        mean_squared_error(main_d["longitude"],main_d["long_pred"])
        mean_squared_error(main_d["latitude"],main_d["lati_pred"])
        mean_squared_error(x["diff_tdoa"]/1000,x["distance_signal_real_diff"]/1000)


    main_d.shape
    len(main_d[0])
    len(main_d)
    main_d = np.array(main_d)
    len(main_d[:,0])
    main_d.shape
    len(main_d[0,0][0])
    np.save(f"train/TDOA/layer_2/main_d_2.npy", main_d)

    for j, item in x.iterrows():
        knn = pairs[pairs["serial_1"] == item["serial_1"]]
        # knn = knn[knn["serial_2"] == item["serial_2"]]
        # knn = knn[knn["id"] != item["id"]]
        # knn = knn[knn["latitude"] >= item["lati_pred"] - delta_error_lati]
        # knn = knn[knn["latitude"] <= item["lati_pred"] + delta_error_lati]
        # knn = knn[knn["longitude"] >= item["long_pred"] - delta_error_long]
        # knn = knn[knn["longitude"] <= item["long_pred"] + delta_error_long]
        # knn = knn[knn["distance_signal_real_diff"] >= item["long_pred"] * (1-delta_error_tdoa)]
        # knn = knn[knn["distance_signal_real_diff"] <= item["long_pred"] * (1+delta_error_tdoa)]
main_d = pd.read_pickle("train/main_d.pkl")
knn = 40
6174
item["longitude"]
item["latitude"]
item["distance_signal_real_diff"]
item["d_aircraft_1"] - item["d_aircraft_2"]


x.loc[6174, "diff_tdoa"]
delta = 0.6
a = geodesy.llh2ecef([0, 0, 0])
b = geodesy.llh2ecef([1.2/9000, 0, 0])
geodesy.ecef_distance(a, b)

for j, item in x.iterrows():
    break
    9000/6
    la_m = item["lati_pred"]  - delta
    la_max = item["lati_pred"]  + delta
    lo_m = item["long_pred"] - delta
    lo_max = item["long_pred"] + delta
    lati_grip_1 = torch.linspace(la_m, la_max,9000 ).cuda()
    long_grip_1 = torch.linspace(lo_m, lo_max, 9000).cuda()
    lati_grip_2 = torch.linspace(0, 9000, 7,dtype=int).cuda()
    long_grip_2 = torch.linspace(0, 9000, 7,dtype=int).cuda()
    lati_grip_3 = torch.linspace(la_m, la_max, 7).cuda()
    long_grip_3 = torch.linspace(lo_m, lo_max, 7).cuda()
    # lati_grip_1 = torch.linspace(lati -delta, lati + delta, 8000).cuda()
    # long_grip_1 = torch.linspace(long - delta, long + delta, 8000).cuda()

    lati_grip_1, long_grip_1 = torch.meshgrid([lati_grip_1, long_grip_1])
    lati_grip_3, long_grip_3 = torch.meshgrid([lati_grip_3, long_grip_3])

    x_grid_1, y_grid_1, z_grid_1 = geodesy.llh2ecef_torch([lati_grip_1, long_grip_1, item["geoAltitude"]])
    d_1 = geodesy.ecef_distance_torch([x_grid_1, y_grid_1, z_grid_1], item[['x_1', 'y_1', 'z_1']])
    d_2 = geodesy.ecef_distance_torch([x_grid_1, y_grid_1, z_grid_1], item[['x_2', 'y_2', 'z_2']])
    diff_tdoa = d_1 - d_2

    location = ((diff_tdoa >= item["distance_signal_real_diff"] * 0.9) & (diff_tdoa <= item["distance_signal_real_diff"] * 1.1))
    zeros = np.zeros((6,6))
    for idx_1 in range(6):
        for idx_2 in range(6):


            sum_S =torch.sum(location[lati_grip_2[idx_2]:lati_grip_2[idx_2 + 1], long_grip_2[idx_1]:long_grip_2[idx_1 + 1]])
            zeros[idx_2, idx_1] = sum_S.cpu().numpy()
        np.argmax(zeros)
    print(np.round(zeros,4))
    zeros/1000000
    lati_grip_3[1,:]
    # item["lati_pred"] = 50.1814
    item["lati_pred"] += delta/6
    item["long_pred"] -= delta/12

    delta = delta/2
    geodesy.ecef_distance(item[['x', 'y', 'z']], item[['x_1', 'y_1', 'z_1']])
    geodesy.ecef_distance(item[['x', 'y', 'z']], item[['x_2', 'y_2', 'z_2']])
    torch.min(d_1)
    13162 - 11508
    location = ((x_grid_1 >= item["latitude"] * 0.9) & (x_grid_1 <= item["latitude"] * 1.1))

    156 134
    torch.sum(location)


    diff_tdoa[location][diff_tdoa[location] < 1] -= 2
    sorted, indices = torch.sort(diff_tdoa[location])
    lati = lati_grip_1[location][indices][:knn]
    lati = torch.mean(lati).cpu().numpy()
    long = long_grip_1[location][indices][:knn]
    long = torch.mean(long).cpu().numpy()
    torch.sum(location)
    torch.sum(diff_tdoa <= item["diff_tdoa"] * 1.2)




13 z904


