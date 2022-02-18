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





# Create grid - map
k_nn = 100

grip_size = 12000

sensor = pd.read_csv("train/sensors.csv")

train = pd.read_pickle(f"train/TDOA/{1}.pkl")
max_min_ll = pd.read_pickle("train/max_min_LL.pkl")
max_min_ll.shape
all_flight = train["id"].values
all_flight = Counter(all_flight).keys()
#
all_flight_dict_store = []
all_find = []
#
# # gbm_lat = joblib.load('model/lgb_lat_v1.pkl')
# # gbm_long = joblib.load('model/lgb_long_v1.pkl')
# # gbm_TDOA = joblib.load('model/lgb_TDOA_v1.pkl')
# #
# # train["distance_signal_real_diff"] = train["d_aircraft_1"] - train["d_aircraft_2"]
# # train[["x_1", "y_1", "z_1",
# #    "x_2", "y_2", "z_2"
# #     , "distance_diff", "distance_signal_diff", "distance_signal_real_diff", "d_aircraft_1", "d_aircraft_2"]] = \
# #     train[["x_1", "y_1", "z_1",
# #        "x_2", "y_2", "z_2"
# #         , "distance_diff", "distance_signal_diff", "distance_signal_real_diff", "d_aircraft_1", "d_aircraft_2"]] / 1000
# # train = train.reset_index()
# # first_test = train.iloc[0, :]
# # fead_lat = first_test.drop(
# #         ["serial_1", "serial_2", "x", "y", "z", "d_aircraft_1",  "d_aircraft_2", "distance_signal_diff", "id", "z_1", "z_2","index"], axis=0)
# #
# # fead_lat =fead_lat.values
# # lat = gbm_lat.predict([fead_lat])
# #
# #
# # fead_long = first_test.drop(
# #         ["serial_1", "serial_2", "x", "y", "z", "d_aircraft_1",  "d_aircraft_2", "distance_signal_diff", "id", "z_1", "z_2","index"], axis=0)
# #
# # fead_long =fead_long.values
# # long = gbm_long.predict([fead_long])
# #
# #
# # fead_TDOA = first_test.drop(
# #         ["serial_1", "serial_2", "x", "y", "z", "d_aircraft_1",  "d_aircraft_2", "distance_signal_real_diff", "id", "z_1", "z_2","index"], axis=0)
# # fead_TDOA.keys()
# # fead_TDOA =fead_TDOA.values
# # tdoa = gbm_TDOA.predict([fead_TDOA])
# # first_test["distance_signal_real_diff"]
# #
# # c_delta = 1
# #
# # max_lat = lat[0] + c_delta
# # min_lat = lat[0] - c_delta
# # max_long =long[0] + c_delta
# # min_long = long[0] - c_delta
# #
# # height_grip = first_test["baroAltitude"]
# #
# # delta_lati = (max_lat - min_lat) / grip_size
# # delta_long = (max_long - min_long) / grip_size
# #
# # lati_grip = torch.arange(min_lat, max_lat, delta_lati).cuda()
# # long_grip = torch.arange(min_long, max_long, delta_long).cuda()
# #
# # lati_grip, long_grip = torch.meshgrid([lati_grip, long_grip])
# #
# # x_grid, y_grid, z_grid = geodesy.llh2ecef_torch([lati_grip, long_grip, height_grip])
# #
# # s_1 = first_test[["x_1", "y_1", "z_1"]].values * 1000
# # s_2 = first_test[["x_2", "y_2", "z_2"]].values * 1000
# # d_1 = geodesy.ecef_distance_torch([x_grid, y_grid, z_grid], s_1)
# # d_2 = geodesy.ecef_distance_torch([x_grid, y_grid, z_grid], s_2)
# #
# #
# # d = d_1 - d_2
# # TODA_dist = tdoa[0] *1000
# # local_location = (TODA_dist - 0.00065 <= d) & (d <= TODA_dist + 0.00065) & (d_1 < 1000000) & (d_2 < 1000000)
# # time_sum = d * 0
# # time_sum[local_location] = d_1[local_location] + d_2[local_location]
# #
# # sorted, indices = torch.sort(time_sum[local_location])
# # location_time = torch.stack((x_grid[local_location][indices], y_grid[local_location][indices],
# #                             z_grid[local_location]
# #                             ,time_sum[local_location][indices]),dim=1)
# #
# # location_time = location_time[:k_nn]
# # a = torch.mean(location_time[:, 0])
# # b = torch.mean(location_time[:,1])
# # c = torch.mean(location_time[:,2])
# #
# # np.sqrt(mean_squared_error(np.array([first_test["x"], first_test["y"], first_test["z"]]) /1000, np.array([a,b,c])/1000))
# #
# #
# # delta = 50
# # a = (x_grid[local_location] <= first_test["x"] + delta) &  (x_grid[local_location] >= first_test["x"] - delta)
# # b = (y_grid[local_location] <= first_test["y"] + delta) &  (y_grid[local_location] >= first_test["y"] - delta) & a
# # torch.sum(b)
# # delta = 0.02
# # a = (lati_grip[local_location] <= max_min_ll["latitude"] + delta) &  (lati_grip[local_location] >= max_min_ll["latitude"] - delta)
# # b = (long_grip[local_location] <= max_min_ll["longitude"] + delta) &  (long_grip[local_location] >= max_min_ll["longitude"] - delta) & a
# # torch.sum(b)
# c_time = time.time()
not_found = []
nn_S = []
counter = 0
c_time = time.time()
for id in all_flight:
    c_delta = 0.5
    id_fight = train[train["id"] == id]
    max_min = max_min_ll[max_min_ll["id"] == id]
    max_lat = max_min["latitude"].values[0] + c_delta
    min_lat = max_min["latitude"].values[0] - c_delta
    max_long = max_min["longitude"].values[0] + c_delta
    min_long = max_min["longitude"].values[0] - c_delta
    height_grip = id_fight["baroAltitude"].values[0]

    lati_grip = torch.linspace(min_lat, max_lat, grip_size)
    long_grip = torch.linspace(min_long, max_long, grip_size)

    lati_grip, long_grip = torch.meshgrid([lati_grip, long_grip])
    x_grid, y_grid, z_grid = geodesy.llh2ecef_torch([lati_grip, long_grip, height_grip])
    del lati_grip, long_grip
    xyz_stack = torch.cat((torch.unsqueeze(x_grid, -1), torch.unsqueeze(y_grid, -1), torch.unsqueeze(z_grid, -1)),
                          dim=-1)
    xyz_stack = xyz_stack.cuda()
    for i, item in id_fight.iterrows():
        s_1 = item[["x_1", "y_1", "z_1"]].values
        s_2 = item[["x_2", "y_2", "z_2"]].values
        d_1 = geodesy.ecef_distance_torch_stack(xyz_stack, s_1)
        d_2 = geodesy.ecef_distance_torch_stack(xyz_stack, s_2)
        d = d_1 - d_2
        TODA_dist = abs(item["d_aircraft_1"] - item["d_aircraft_2"])
        location = (TODA_dist-46 < abs(d)) & (abs(d)< TODA_dist + 46)
        time_sum = torch.sqrt((item["distance_signal_diff"] - d[location]) ** 2)
        location.shape
        sorted, indices = torch.sort(time_sum)
        location_time = torch.cat(
            (xyz_stack[location, :][indices, :]
             , torch.unsqueeze(sorted, -1)), dim=1)
        location_time = location_time[:k_nn, ].cpu().numpy()
        data = item.values
        nn_S.append([data, location_time.flatten()])
        counter += 1
        if counter % 100 == 0:
            print(counter)
            print(time.time() - c_time)
            c_time = time.time()