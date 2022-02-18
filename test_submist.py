import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import lightgbm as lgb
from mpl_toolkits.basemap import Basemap
from collections import Counter
from math import sin, cos, sqrt, atan2, radians
from mpl_toolkits import mplot3d
from mlat import geodesy, constants
import torch
from scipy.signal import savgol_filter

plane_speed = 800 * 1000 / 3600



test = pd.read_pickle("test/test.pkl")
test = test[pd.isna(test["latitude"])]
test.shape
sensor = pd.read_csv("test/sensors.csv")
xyz = geodesy.llh2ecef([sensor["latitude"].values, sensor["longitude"].values, sensor["height"].values])
sensor["x"] = xyz[0]
sensor["y"] = xyz[1]
sensor["z"] = xyz[2]
sensor = sensor.set_index("serial")
x = test["measurements"]
plane_id = test["aircraft"]
plane_id = Counter(plane_id)
plane_id = plane_id.keys()

train_data = pd.read_pickle(f"train/TDOA/test.pkl")
train_data = train_data.set_index("id")
train_data.shape


gbm_lat = lgb.Booster(model_file='model/lgb_lati_v2.txt')
gbm_long = lgb.Booster(model_file='model/lgb_long_v2.txt')
gbm_tdoa = lgb.Booster(model_file='model/lgb_tdoa_v2.txt')

total_fail = []
for id in plane_id:
    id = 172
    sens_erros = {}
    flight = test[test["aircraft"] == id]
    x = train_data.loc[flight["id"].values - 1 + 14000000, :]
    tmp = x.drop(
        ["x", "y", "z", "d_aircraft_1", "d_aircraft_2", "distance_signal_real_diff"], axis=1)
    tmp["id"] = tmp.index
    pred_tdoa = gbm_tdoa.predict(tmp)
    x["pred_tdoa"] = pred_tdoa

    tmp["distance_signal_diff"] = pred_tdoa
    pred_lati = gbm_lat.predict(tmp)
    pred_long = gbm_long.predict(tmp)
    x["pred_long"] = pred_long
    x["pred_lati"] = pred_lati
    w = savgol_filter(x["pred_lati"].values, 101, 2)
    x["pred_lati"] = w
    w = savgol_filter(x["pred_long"].values, 101, 2)
    x["pred_long"] = w

    x["distance_signal_real_diff"] = x["d_aircraft_1"] - x["d_aircraft_2"]
    # x = x[abs(x["time_diff_1"] - x["time_diff_2"]) >= 0.05]
    x["min_time"] = 0
    x["d_min"] = 0

    for j, item in flight.iterrows():
        min = 0
        s_min = 0
        for me in item["measurements"]:
            if me[1] <= min or min == 0:
                min = me[1]
                s_min = me[0]
        xyz = geodesy.llh2ecef([item["latitude"], item["longitude"],
                                item["geoAltitude"]])
        xyz_s = sensor.loc[s_min - 1, ['x', 'y', 'z']].values
        tdoa_real = geodesy.ecef_distance(xyz, xyz_s)

        x.loc[j, "d_min"] = tdoa_real
        x.loc[j, "min_time"] = min * 1e-9 - item["timeAtServer"]


flight.shape
first_point = flight.iloc[0:15, :]
first_point_data = x.loc[first_point["id"].values - 1 + 14000000, :]
first_point = first_point.reset_index()
total_time = first_point.loc[len(first_point) - 1, "timeAtServer"] - first_point.loc[0, "timeAtServer"]

# distance_travel = total_time * plane_speed
# lati_pred_center = first_point_data["pred_lati"]
# long_pred_center = first_point_data["pred_long"]
# lati_pred_center = np.mean(lati_pred_center)
# long_pred_center = np.mean(long_pred_center)

height = np.mean(first_point["baroAltitude"])
lati_grip = torch.linspace(46.8934 - 1, 46.8934 + 1, 5000).cuda()
long_grip = torch.linspace(7.1624 - 1, 7.1624 + 1, 5000).cuda()

lati_grip, long_grip = torch.meshgrid([lati_grip, long_grip])
x_grid, y_grid, z_grid = geodesy.llh2ecef_torch([lati_grip, long_grip, height])
timestamp = 0
d_errors = None

for j, item in first_point.iterrows():
    data = first_point_data.loc[[item["id"] - 1 + 14000000], :]
    for idx, item_v in data.iterrows():
        if abs(item_v["distance_signal_diff"]) < item_v['distance_diff']:
            xyz_1 = item_v[["x_1", "y_1", "z_1"]].values
            xyz_2 = item_v[["x_2", "y_2", "z_2"]].values
            d_1 = geodesy.ecef_distance_torch([x_grid, y_grid, z_grid], xyz_1)
            d_2 = geodesy.ecef_distance_torch([x_grid, y_grid, z_grid], xyz_2)
            d = d_1 - d_2
            d = torch.sqrt((d - item_v["distance_signal_diff"]) ** 2)
            if d_errors is None:
                d_errors = d
            else:
                d_errors += d
            break

master_d_e = d_errors
d_errors = d_errors.flatten()
sorted, index = torch.sort(d_errors)

lati_grip = lati_grip.flatten()
long_grip = long_grip.flatten()
new_lati = lati_grip[index][0]
new_long = long_grip[index][0]


new_lati = 46.743469
new_long = 7.11038
first_point = flight.iloc[11: 21, :]
first_point_data = x.loc[first_point["id"].values - 1 + 14000000, :]
first_point = first_point.reset_index()
total_time = first_point.loc[len(first_point) - 1, "timeAtServer"] - first_point.loc[0, "timeAtServer"]

delta_grip = 6000
lati_grip = torch.linspace(new_lati - 0.1, new_lati + 0.1, delta_grip).cuda()
long_grip = torch.linspace(new_long - 0.1, new_long + 0.1, delta_grip).cuda()

a = geodesy.llh2ecef([0.4, 0.4, 0])
b = geodesy.llh2ecef([0.0, 0.0, 0])
distance_per_cell = geodesy.ecef_distance(a, b)

lati_grip, long_grip = torch.meshgrid([lati_grip, long_grip])
x_grid, y_grid, z_grid = geodesy.llh2ecef_torch([lati_grip, long_grip, height])

timestamp = 0
d_errors = master_d_e
max_point = 1000
max_point_1 = 10000
xyz = []
map_seri = {}

distance_move_max = []
distance_move_min = []

for j, item in first_point.iterrows():
    data = first_point_data.loc[[item["id"] - 1 + 14000000], :]
    distance_move = (item["timeAtServer"] - timestamp) * plane_speed * 0.85
    min_distance_move = (item["timeAtServer"] - timestamp) * plane_speed * 0.7

    timestamp = item["timeAtServer"]

    for idx, item_v in data.iterrows():
        diff_real = item_v["pred_tdoa"]
        if abs(diff_real) < item_v['distance_diff']:
            xyz_1 = item_v[["x_1", "y_1", "z_1"]].values
            xyz_2 = item_v[["x_2", "y_2", "z_2"]].values
            d_1 = geodesy.ecef_distance_torch([x_grid, y_grid, z_grid], xyz_1)
            d_2 = geodesy.ecef_distance_torch([x_grid, y_grid, z_grid], xyz_2)
            d = d_1 - d_2
            d = torch.sqrt((d - diff_real) ** 2)
            if d_errors is None:
                d_errors = d
            else:
                d_errors += d

    if j != 0:
        distance_move_max.append(distance_move)
        distance_move_min.append(min_distance_move)

    tmp = d_errors.flatten()
    sorted, index = torch.sort(tmp)

    if j == 0:
        p_x = index[:max_point_1] / delta_grip
        p_y = index[:max_point_1] % delta_grip
        p1 = torch.stack((p_x, p_y), dim=1)
        map_seri[j + 1] = p1

    # if j == 1:
    #     p_x = index[:int(max_point * 4)] / delta_grip
    #     p_y = index[:int(max_point * 4)] % delta_grip
    #
    #     p2 = torch.stack((x_grid[p_x, p_y], y_grid[p_x, p_y], z_grid[p_x, p_y]), dim=1)
    #     p_1 = map_seri[1]
    #
    #     p1 = torch.stack((x_grid[p_1[:, 0], p_1[:, 1]], y_grid[p_1[:, 0], p_1[:, 1]], z_grid[p_1[:, 0], p_1[:, 1]]),
    #                      dim=1)
    #
    #     p1 = p1.unsqueeze(-1)
    #     p1 = p1.repeat(1, 1, len(p2))
    #
    #     distance = torch.sqrt(
    #         (p2[:, 0] - p1[:, 0, :]) ** 2 + (p2[:, 1] - p1[:, 1, :]) ** 2 + (p2[:, 2] - p1[:, 2, :]) ** 2)
    #     torch.max(distance)
    #     location = (distance <= distance_move) & (distance >= min_distance_move)
    #
    #     location = torch.sum(location, dim=0)
    #     location = location != 0
    #     torch.sum(location)
    #
    #     location_2 = torch.sum((distance <= distance_move) & (distance >= min_distance_move), dim=1)
    #     location_2 = location_2 != 0
    #     torch.sum(location_2)
    #
    #     map_seri[1] = torch.stack((map_seri[1][location_2, 0], map_seri[1][location_2, 1]), dim=1)
    #     map_seri[2] = torch.stack((p_x[location], p_y[location]), dim=1)
    #     #print(len(map_seri[2][:,0]))

    if j >= 1:
        counter = 1
        idx = 0
        p_x = index[:max_point] / delta_grip
        p_y = index[:max_point] % delta_grip
        while True:

            counter += 1
            if counter >= 100:
                break
            p3_ = torch.stack((x_grid[p_x, p_y], y_grid[p_x, p_y], z_grid[p_x, p_y]), dim=1)
            stack_location = []
            for idx in range(len(map_seri)):
                p_1 = map_seri[idx + 1]
                p_ = torch.stack((x_grid[p_1[:, 0], p_1[:, 1]], y_grid[p_1[:, 0], p_1[:, 1]], z_grid[p_1[:, 0], p_1[:, 1]]),
                                 dim=1)
                max_shift_stack = np.sum(distance_move_max[idx:])
                min_shift_stack = max_shift_stack * 0.9
                p_ = p_.unsqueeze(-1)
                p_ = p_.repeat(1, 1, len(p3_))
                distance = torch.sqrt((p3_[:, 0] - p_[:, 0, :]) ** 2 + (p3_[:, 1] - p_[:, 1, :]) ** 2
                                      + (p3_[:, 2] - p_[:, 2, :]) ** 2)
                with_p1_l = (distance <= max_shift_stack) & (distance >= min_shift_stack)
                location = torch.sum(with_p1_l, dim=0)
                location = location != 0

                location_2 = torch.sum((distance <= max_shift_stack) & (distance >= min_shift_stack), dim=1)
                location_2 = location_2 != 0
                stack_location.append(location_2)
                if torch.sum(location) == 0:
                    p_x = index[max_point * (counter - 1): max_point * counter] / delta_grip
                    p_y = index[max_point * (counter - 1): max_point * counter] % delta_grip
                    break

            if torch.sum(location) != 0 and len(stack_location) == len(map_seri):
                for idx in range(len(map_seri)):
                    l = stack_location[idx]
                    p_s = map_seri[idx + 1]
                    map_seri[idx + 1] = p_s[l, :]
                p_x = p_x[location]
                p_y = p_y[location]
                break

        p3 = torch.stack((p_x, p_y), dim=1)
        map_seri[j + 1] = p3
        print(len(map_seri[j + 1]))

torch.mean(long_grip[map_seri[j+1][:,0][:3],map_seri[j+1][:,1][:3]]).cpu().numpy()
torch.mean(lati_grip[map_seri[j+1][:,0][:3],map_seri[j+1][:,1][:3]]).cpu().numpy()



#
# counter += 1
# if counter >= 100:
#     break
# p3_ = torch.stack((x_grid[p_x, p_y], y_grid[p_x, p_y], z_grid[p_x, p_y]), dim=1)
# stack_location = []
# for idx in range(len(map_seri)):
#     p_1 = map_seri[idx + 1]
#     p_ = torch.stack((x_grid[p_1[:, 0], p_1[:, 1]], y_grid[p_1[:, 0], p_1[:, 1]], z_grid[p_1[:, 0], p_1[:, 1]]),
#                      dim=1)
#     max_shift_stack = np.sum(distance_move_max[idx:])
#     min_shift_stack = max_shift_stack * 0.85
#     p_ = p_.unsqueeze(-1)
#     p_ = p_.repeat(1, 1, len(p3_))
#     distance = torch.sqrt((p3_[:, 0] - p_[:, 0, :]) ** 2 + (p3_[:, 1] - p_[:, 1, :]) ** 2
#                           + (p3_[:, 2] - p_[:, 2, :]) ** 2)
#     with_p1_l = (distance <= max_shift_stack) & (distance >= min_shift_stack)
#     location = torch.sum(with_p1_l, dim=0)
#     location = location != 0
#
#     location_2 = torch.sum((distance <= max_shift_stack) & (distance >= min_shift_stack), dim=1)
#     location_2 = location_2 != 0
#     stack_location.append(location_2)
#     if torch.sum(location) == 0:
#         p_x = index[max_point * (counter - 1): max_point * counter] / delta_grip
#         p_y = index[max_point * (counter - 1): max_point * counter] % delta_grip
#         break
#
# if torch.sum(location) != 0 and len(stack_location) == len(map_seri):
#     for idx in range(len(map_seri)):
#         l = stack_location[idx]
#         p_s = map_seri[idx + 1]
#         map_seri[idx + 1] = p_s[l, :]
#         print("Aaaaaaaaaaaaa")
#     p_x = p_x[location]
#     p_y = p_y[location]
#     break
