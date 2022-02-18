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

# Create grid - map
height_grip = 11582
max_delta = 3
max_grip = 8
a = geodesy.llh2ecef([0, 0, 0])
b = geodesy.llh2ecef([0, max_delta * 2/max_grip, 0])
delta_in_m = geodesy.ecef_distance(a, b)

sensor = pd.read_csv("train/sensors.csv")
stack_sensor_pair = []
len(stack_sensor_pair)

sensor_pair = np.load("train/sensor_pair.npy")
sensor_pair_done = [[],[]]
data = np.array([])
data_in_pd = []
train_master = []

for i in np.arange(14) +1:
    train_master = pd.read_pickle(f"train/TDOA/{i}.pkl")
    train_master = train_master[train_master["distance_signal_diff"] <= 1000000]
    #train_master = train_master.iloc[:int(len(train_master) * 0.2),:]
    counter = 0
    for s_sss in sensor_pair:
        s_1 = s_sss[0]
        s_2 = s_sss[1]
        counter += 1
        train = []

        idx_s = ((train_master["serial_1"] == s_1) & (train_master["serial_2"] == s_2) |
                 (train_master["serial_2"] == s_1) & (train_master["serial_1"] == s_2))
        train = train_master[idx_s]

        # x = train["serial_1"].values
        # y = train["serial_2"].values
        # dict_all_sensor = np.concatenate((x,y))
        # x = Counter(x)

        llh_1 = sensor[sensor["serial"] == s_1][["latitude", "longitude", "height"]].values[0]
        llh_2 = sensor[sensor["serial"] == s_2][["latitude", "longitude", "height"]].values[0]


        llh_fight = geodesy.ecef2llh([train["x"].values, train["y"].values, train["z"].values])

        train["latitude"] = llh_fight[0]
        train["longitude"] = llh_fight[1]
        train["geoAltitude"] = llh_fight[2]

        idx_s = (((train["latitude"] <= llh_1[0] + max_delta) & (train["latitude"] >= llh_1[0] - max_delta) &
                 (train["longitude"] <= llh_1[1] + max_delta) & (train["longitude"] >= llh_1[1] - max_delta)) |
                 ((train["latitude"] <= llh_2[0] + max_delta) & (train["latitude"] >= llh_2[0] - max_delta) &
                  (train["longitude"] <= llh_2[1] + max_delta) & (train["longitude"] >= llh_2[1] - max_delta)))
        if counter % 100 ==0:
            print(counter)

        if len(train) == 0:
            continue

        idx_ = train["serial_1"] == s_1
        train.loc[idx_, "x_1"] = llh_1[0]
        train.loc[idx_, "y_1"] = llh_1[1]
        train.loc[idx_, "z_1"] = llh_1[2]

        train.loc[idx_, "x_2"] = llh_2[0]
        train.loc[idx_, "y_2"] = llh_2[1]
        train.loc[idx_, "z_2"] = llh_2[2]

        idx_ = train["serial_1"] == s_2
        train.loc[idx_, "x_1"] = llh_2[0]
        train.loc[idx_, "y_1"] = llh_2[1]
        train.loc[idx_, "z_1"] = llh_2[2]

        train.loc[idx_, "x_2"] = llh_1[0]
        train.loc[idx_, "y_2"] = llh_1[1]
        train.loc[idx_, "z_2"] = llh_1[2]

        xyz_1 = geodesy.llh2ecef(llh_1)
        xyz_2 = geodesy.llh2ecef(llh_2)

        lati_grip_1 = torch.linspace(llh_1[0] - max_delta, llh_1[0] + max_delta, max_grip).cuda()
        long_grip_1 = torch.linspace(llh_1[1] - max_delta, llh_1[1] + max_delta, max_grip).cuda()

        x_grid_1, y_grid_1 = torch.meshgrid([lati_grip_1, long_grip_1])
        x_grid_1, y_grid_1, z_grid_1 = geodesy.llh2ecef_torch([x_grid_1, y_grid_1, height_grip])
        d_1 = geodesy.ecef_distance_torch([x_grid_1, y_grid_1, z_grid_1], xyz_1)
        d_2 = geodesy.ecef_distance_torch([x_grid_1, y_grid_1, z_grid_1], xyz_2)
        diff_tdoa_1 = d_1 - d_2

        lati_grip_2 = torch.linspace(llh_2[0] - max_delta, llh_2[0] + max_delta, max_grip).cuda()
        long_grip_2 = torch.linspace(llh_2[1] - max_delta, llh_2[1] + max_delta, max_grip).cuda()

        x_grid_1, y_grid_1 = torch.meshgrid([lati_grip_2, long_grip_2])
        x_grid_1, y_grid_1, z_grid_1 = geodesy.llh2ecef_torch([x_grid_1, y_grid_1, height_grip])
        d_1 = geodesy.ecef_distance_torch([x_grid_1, y_grid_1, z_grid_1], xyz_1)
        d_2 = geodesy.ecef_distance_torch([x_grid_1, y_grid_1, z_grid_1], xyz_2)
        diff_tdoa_2 = d_1 - d_2

        train["distance_diff"] = train["distance_diff"] / delta_in_m
        # d_1 positive , d_2 negative
        idx_ = train["d_aircraft_1"] < train["d_aircraft_2"]
        lati = train.loc[idx_, "latitude"]
        lati = (lati - (llh_1[0] - max_delta))/(max_delta*2)
        train.loc[idx_, "latitude"] = lati

        long = train.loc[idx_,"longitude"]
        long = (long - (llh_1[1] - max_delta))/(max_delta*2)
        train.loc[idx_, "longitude"] = long

        idx_ = train["d_aircraft_2"] <= train["d_aircraft_1"]
        lati = train.loc[idx_,"latitude"]
        lati = -(lati - (llh_2[0] - max_delta))/(max_delta*2)
        train.loc[idx_, "latitude"] = lati

        long = train.loc[idx_, "longitude"]
        long = -(long - (llh_2[1] - max_delta)) / (max_delta * 2)
        train.loc[idx_, "longitude"] = long

        diff_tdoa_1 = diff_tdoa_1.flatten().cpu().numpy()
        diff_tdoa_2 = diff_tdoa_2.flatten().cpu().numpy()
        assert np.sum(diff_tdoa_2 == np.inf) == 0
        assert np.sum(diff_tdoa_1 == np.inf) == 0
        diff_tdoa_2 = np.tile(diff_tdoa_2,(len(train),1))
        diff_tdoa_1 = np.tile(diff_tdoa_1,(len(train),1))


        tmp_store = train
        data_in_pd.append(tmp_store)
        train = train[["latitude", "longitude", "time_diff_1", "signal_strength_1", " type_1","time_diff_2", "signal_strength_2", " type_2", "TDOA",
                       "distance_diff", "distance_signal_diff", "baroAltitude"]].values
        train = np.concatenate((train, diff_tdoa_1, diff_tdoa_2), axis=-1)
        if len(data) == 0:
            data = train
        else:
            data = np.concatenate((data, train), axis=0)


    np.save(f"train/sen_pair/{i}.npy", data)
    data_in_pd = pd.concat(data_in_pd)
    data_in_pd.to_pickle(f"train/sen_pair/{i}_df.pkl")
    data = []
    data_in_pd = []
    print("SAVE")
