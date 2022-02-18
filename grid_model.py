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



# Create grid - map
k_nn = 100

max_long = 26
min_long = -13
max_lat = 62
min_lat = 32
grip_size = 24000
delta_lati = (max_lat - min_lat )/grip_size
delta_long = (max_long - min_long)/grip_size
lati_grip = torch.arange(min_lat, max_lat,delta_lati)
long_grip = torch.arange(min_long, max_long,delta_long)


lati_grip, long_grip=torch.meshgrid([lati_grip,long_grip])
lati_grip = lati_grip.cuda()
long_grip = long_grip.cuda()
height_grip = 11582


x_grid, y_grid, z_grid = geodesy.llh2ecef_torch([lati_grip, long_grip, height_grip])

del lati_grip, long_grip
torch.cuda.empty_cache()


sensors = pd.read_csv("train/sensors.csv")
train = pd.read_pickle(f"train/TDOA/{1}.pkl")

all_flight = train["id"].values
all_flight = Counter(all_flight).keys()

all_flight_dict_store = {}
all_find = []
for id in all_flight:
    location = None
    time_sum = None
    id_fight = train[train["id"] == id]
    for i, item in id_fight.iterrows():
        s_1 = item[["x_1", "y_1", "z_1"]].values
        s_2 = item[["x_2", "y_2", "z_2"]].values
        d_1 = geodesy.ecef_distance_torch([x_grid, y_grid, z_grid], s_1)
        d_2 = geodesy.ecef_distance_torch([x_grid, y_grid, z_grid], s_2)
        if time_sum is None:
            time_sum = d_1 * 0
        d = d_1 - d_2
        TODA_dist = item["d_aircraft_1"] - item["d_aircraft_2"]
        local_location = (TODA_dist-500*4 <= d) & (d<= TODA_dist + 500*4) & (d_1 < 1000000) & (d_2 < 1000000)
        if location is None:
            location = local_location
        else:
            location = local_location & location
        time_sum[location] = d_1[location] + d_2[location] + time_sum[location]

    # a_lati = lati_grip[location].cpu().numpy()
    # a_long = long_grip[location].cpu().numpy()

    # all_find.append([np.min(a_lati), np.mean(a_lati), np.std(a_lati), np.max(a_lati),
    #                  np.min(a_long), np.mean(a_long), np.std(a_long) ,np.max(a_long)])
    location_time = np.array([x_grid[location].cpu().numpy(), y_grid[location].cpu().numpy(), \
                                z_grid[location].cpu().numpy(), time_sum[location].cpu().numpy()])



    # x = long_grip[location]
    # x.mean()
    location_time= location_time[:,location_time[3,:].argsort()]
    location_time = location_time[:,:k_nn]
    # location_time[3,:20]
    # delta = 100
    # a = (location_time[0,:] <= item["x"] + delta) &  (location_time[0,:] >= item["x"] - delta)
    # b = (location_time[1,:] <= item["y"] + delta) &  (location_time[1,:] >= item["y"] - delta) & a
    # np.sum(b)
    # all_find.append([torch.sum(location),np.sum(b)])
    all_flight_dict_store[id] = location_time
    if id % 10 == 0:
        print(id)
    # all_find = np.array(all_find)
    # x = all_find[:,0]
    # x = all_find[:, -1] - all_find[:,-4]
    # np.min(x)
    # len(all_find)
    # if id == 10000:
    #     break
#
# a = x_grid <= 4114084
#
# mean_squared_error([564156.1875,4114397.7500,0],[563765.899786694, 41 14084.0394785465, 0])
#
# 4042234.0000, 4036351.5000
# 620913.5000, 625284.0625
# np.sqrt(mean_squared_error([32,-13.0000,0],[32.0000, -12.3500,0]))
5/16000
a = geodesy.llh2ecef([2, 0.,0])
b = geodesy.llh2ecef([0, 0.,0])
geodesy.ecef_distance(a,b)
44
geodesy.llh2ecef([53.625767, 14.411346, 6537])
item["x"]
140
# 311
# 277
# np.sqrt(311 **2 + 277**2)
from sys import getsizeof
max_long = 26
min_long = -13
max_lat = 62
min_lat = 32

max_long = 26
min_long = -13
max_lat = 34

min_lat = 32
np.linspace
lati_grip.shape
26 + 13
lati_grip = torch.arange(32, 62, 2)
long_grip = torch.arange(-13, 26, 39/15)
long_grip.shape
delta_lati = (max_lat - min_lat )/grip_size
delta_long = (max_long - min_long)/grip_size
lati_grip = torch.arange(min_lat, max_lat,delta_lati)
long_grip = torch.arange(min_long, max_long,delta_long)
torch.save(tensor, 'file.pt')

lati_grip, long_grip=torch.meshgrid([lati_grip,long_grip])
lati_grip = torch.sin(lati_grip) + torch.cos(lati_grip)
np.lindspace
# lati_grip = lati_grip.cuda()
# long_grip = long_grip.cuda()
from sys import getsizeof
getsizeof(lati_grip)
height_grip = 11582
