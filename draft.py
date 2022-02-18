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
plane_speed = 800 * 1000 /3600
from sklearn.metrics import mean_squared_error
from collections import  deque
delta_lati = 3
delta_long = 4
grip_size = 16
max_history = 10

sensor_train = pd.read_csv("train/sensors.csv")
xyz = geodesy.llh2ecef([sensor_train["latitude"].values, sensor_train["longitude"].values,
                         sensor_train["height"].values])
sensor_train["x"]  = xyz[0]
sensor_train["y"]  = xyz[1]
sensor_train["z"]  = xyz[2]
sensor_train = sensor_train.set_index("serial")

list_array = ["timestamp", "signal_strength", "timeAtServer", "latitude",
              "longitude", "baroAltitude", "geoAltitude"]
index = {}
counter = 0
for i in list_array:
    index[i] = counter
    counter += 1
predict_value = ["first_location", "last_location", "first_angle",
                 "last_angle", "first_closer", "last_closer"]


(-468.13537878,  465.43805586,    9.29077896,  -97.40332785,
         93.80689728,  -36.56371076,   44.95538209,  -34.76549548,
         14.38572227)

a = geodesy.llh2ecef([50.0137225652145, 8.526207176414701, 0])
b = geodesy.llh2ecef([50.01484680175779, 8.52611340974507, 0])
geodesy.ecef_distance(a, b)
104.0187777408934 * constants.Cair/1e9
a = geodesy.llh2ecef([50.0137225652145, 8.526207176414701, 0])
b = geodesy.llh2ecef([50.01484680175779, 8.52611340974507, 0])
(np.array(e) - np.array(f)) /constants.Cair * 1e9
(np.array(a) - np.array(b)) * constants.Cair * 1e-9
def process_value(data):
    data = x
    data_process = []

    flight_id = data["aircraft"]
    flight_id = Counter(flight_id)

    for id_ in flight_id:
        id_ = 1350
        counter = 0
        if flight_id[id_] >= 20:
            aircraft = data[data["aircraft"] == id_]
            sensors_mins = {}
            time = []
            distance = []
            for j, item in aircraft.iterrows():
                counter += 1
                measure = np.array(item["measurements"])
                s_min = np.argmin(measure[:,1])
                sensor_id = measure[s_min, 0]
                s_min = measure[s_min, 1]
                s_max = np.argmax(measure[:,1])
                sensor_id_max = measure[s_max, 0]

                s_max = measure[s_max, 1]
                time.append(s_max - s_min)
                sen = sensor_train.loc[sensor_id]
                sen2 = sensor_train.loc[sensor_id_max]

                a = geodesy.llh2ecef([sen["latitude"], sen["longitude"], sen["height"]])
                b = geodesy.llh2ecef([sen2["latitude"], sen2["longitude"], sen2["height"]])
                c = geodesy.llh2ecef([item["latitude"], item["longitude"], item["geoAltitude"]])
                d = geodesy.ecef_distance(b, c) - geodesy.ecef_distance(a, c)
                distance.append(d)
                if counter == 10:
                    break
                # s_min[0] = s_min[0]
                item =  item.drop(["numMeasurements", "measurements", "id",
                                  "aircraft"])
                s_min = np.concatenate((s_min, item.values), axis=0)
                if sensor_id not in sensors_mins:
                    sensors_mins[sensor_id] = deque(maxlen=max_history)
                sensors_mins[sensor_id].append(s_min)
                lens_ = len(sensors_mins[sensor_id])
                # if lens_ >= 3:
                #     first = sensors_mins[sensor_id][lens_ -3]
                #     second = sensors_mins[sensor_id][lens_ -1]
                #

                    b.append(second[0] - first[0])
                if counter == 10:
                    break
                if len(sensors_mins[sensor_id]) == 10:
                    break
                    history = np.array(sensors_mins[sensor_id])
                    if history[-1][index["timeAtServer"]] - \
                            history[0][index["timeAtServer"]] <= 16:
                        timestamps = history[:, index["timestamp"]]
                        timestamps = timestamps - np.min(timestamps)
                        timestamps = timestamps * 1e-9 * constants.Cair
                        baroAls = history[:, index["baroAltitude"]]
                        signal_strengths = history[:, index["signal_strength"]]

                        sen = sensor_train.loc[sensor_id]
                        type = 0 if sen["type"] == "GRX1090" else 1
                        x_value = np.concatenate(([sensor_id, type, sen["height"]], timestamps, baroAls, signal_strengths))

                        range_lati = np.linspace( - delta_lati, + delta_lati, grip_size )
                        range_long = np.linspace( - delta_long, + delta_long, grip_size )

                        lati_diff = history[0, index["latitude"]] - sen["latitude"]
                        long_diff = history[0, index["longitude"]] - sen["longitude"]

                        lati_diff_2 = history[1, index["latitude"]] - sen["latitude"]
                        long_diff_2 = history[1, index["longitude"]] - sen["longitude"]
                        first_angle = np.degrees(np.arctan2(lati_diff - lati_diff_2, long_diff - long_diff_2))
                        first_closer = int(np.sqrt(lati_diff ** 2 + long_diff ** 2) >
                                           np.sqrt(long_diff_2 ** 2 + lati_diff_2 ** 2))

                        location_la = np.argmin(abs(range_lati - lati_diff))
                        location_lo = np.argmin(abs(range_long - long_diff))
                        first_location = location_la* grip_size + location_lo


                        lati_diff = history[-1, index["latitude"]] - sen["latitude"]
                        long_diff = history[-1, index["longitude"]] - sen["longitude"]

                        lati_diff_2 = history[-2, index["latitude"]] - sen["latitude"]
                        long_diff_2 = history[-2, index["longitude"]] - sen["longitude"]
                        last_angle = np.degrees(np.arctan2(lati_diff_2 - lati_diff, long_diff_2 - long_diff))
                        last_closer = int(np.sqrt(lati_diff ** 2 + long_diff ** 2) >
                                          np.sqrt(long_diff_2 ** 2 + lati_diff_2 ** 2))

                        lati_diff = np.clip(lati_diff, -delta_lati, delta_lati)
                        long_diff = np.clip(long_diff, -delta_long, delta_long)
                        location_la = np.where(((range_lati >= lati_diff - delta_lati / grip_size) &
                                                (range_lati <= lati_diff + delta_lati / grip_size)))[0]

                        location_la = np.argmin(abs(range_lati - lati_diff))
                        location_lo = np.argmin(abs(range_long - long_diff))
                        last_location = location_la* grip_size + location_lo


                        y_value = np.array([first_location, last_location,
                                   first_angle, last_angle, first_closer, last_closer])
                        data_process.append(np.array([x_value, y_value]))

                if len(data_process) % 10000 == 0:
                    print(len(data_process))

    return data_process



train_set = []
val_set = []
for i in np.arange(7) + 1:
    i = 1
    x = pd.read_pickle(f"train/training_{i}.pkl")
    val = x[pd.isna(x["latitude"])]
    val = val.set_index("id")
    result = pd.read_csv(f"train/training_{i}_category_1_result.csv")
    result = result.set_index("id")
    assert len(val) == len(result)

    val[["latitude", "longitude", "geoAltitude"]] = result[["latitude", "longitude", "geoAltitude"]]

    x = x[np.invert(pd.isna(x["latitude"]))]

    train_set = process_value(x)
    np.save(f"train/min_sensor/{i}.npy", train_set)
    print("Finish_TRAIN")
    val["id"] = val.index

    val_set = process_value(val)
    np.save(f"train/min_sensor/{i}_val.npy", val_set)

    print("Finish_Val")
    x.shape


x = distance
x = np.expand_dims(x, axis=1)
x = np.repeat(x,10,axis =1)
x = x - distance
x = np.abs(x) / constants.Cair * 1e9
x = np.round(x,2)
tmp = time
tmp = np.expand_dims(tmp, axis=1)
tmp = np.repeat(tmp,10,axis =1)
tmp = tmp - time
tmp = np.abs(tmp)
xxx = np.array(time) * constants.Cair * 1e-9
xxx = xxx -np.min(xxx)
0 2
0 5
2 5
2 9
3 5
9 * constants.Cair * 1e-9
sen = sensor_train.loc[sensor_id]

lati_grip = torch.linspace(50.0486-2.5, 50.0486+2.5,8000).cuda()
long_grip = torch.linspace(8.4879 - 3, 8.4879 +3,8000).cuda()
lati_grip, long_grip=torch.meshgrid([lati_grip,long_grip])

height_grip = 11582


x_grid, y_grid, z_grid = geodesy.llh2ecef_torch([lati_grip, long_grip, item["geoAltitude"]])
s_1 = sen[["x", "y", "z"]].values
s_2 = sen2[["x", "y", "z"]].values
d_1 = geodesy.ecef_distance_torch([x_grid, y_grid, z_grid], s_1)
d_2 = geodesy.ecef_distance_torch([x_grid, y_grid, z_grid], s_2)
d = d_2 -d_1
torch.min(d)
if time_sum is None:
    time_sum = d_1 * 0
d = d_1 - d_2


a = np.load(f"train/min_sensor/{1}.npy", allow_pickle=True)
a[0].shape


train = pd.read_pickle("train/training_1.pkl")
sensor = pd.read_csv("train/sensors.csv")
xyz = geodesy.llh2ecef([sensor["latitude"].values, sensor["longitude"].values, sensor["height"].values])
sensor["x"] = xyz[0]
sensor["y"] = xyz[1]
sensor["z"] = xyz[2]

x = train["measurements"]
plane_id = x["aircraft"]
plane_id = Counter(plane_id)
plane_id = plane_id.keys()
del_ids = []
for i in plane_id:
    if plane_id[i] <= 20:
        del_ids.append(i)
    break
for i in del_ids:
    del plane_id[i]
j = np.sum(list(plane_id.values()))

# flight_ = {}
# for id in plane_id:
#     flight = train[train["aircraft"] == id]
#     #measure = flight["measurement"].values
#     sens = {}
#     for idx, item in flight.iterrows():
#         xyz =  geodesy.llh2ecef([item["latitude"], item["longitude"],
#                                  item["geoAltitude"]])
#         measure = item["measurements"]
#         for j in measure:
#             s_ = j[0]
#             xyz_s = sensor.loc[s_ - 1, ['x', 'y', 'z']].values
#             t_diff = j[1] - item["timeAtServer"] * 1e9
#             tdoa_real = geodesy.ecef_distance(xyz, xyz_s)
#             if s_ in sens:
#                 sens[s_].append([idx, t_diff, tdoa_real, j[-1]])
#             else:
#                 sens[s_] = [[idx, t_diff, tdoa_real, j[-1]]]
#     flight_[id] = sens
    
#
# flight_id = 1322
#
# list_sensor = []
# sens = 130 =>
# sens _erros = t_by_distance - tdoa_real
# len(flight_)/len(plane_id) * 100
# flight_.keys()
# a = flight_[1605]
# x = a.keys()
# for i in x:
#     print(f"{len(a[i])} {i}")
# x = a[191]
# x = np.array(x)
# erros = x[:,1] - x[:,2]/constants.Cair
# np.argmin(erros)
# (986463899 - 932741618) * constants.Cair * 1e-9
# 16100
# x[1692]
# 1.11361000e+06
# x = train.loc[1950347,:]
# x["measurements"]

train_data = []
for i in [1,2,3,4]:
    x = pd.read_pickle(f"train/TDOA/{i}.pkl")
    train_data.append(x)
train_data = pd.concat(train_data)
train_data = train_data.set_index("id")

gbm_lat = lgb.Booster(model_file='model/lgb_lati_v2.txt')
gbm_long = lgb.Booster(model_file='model/lgb_long_v2.txt')
gbm_tdoa = lgb.Booster(model_file='model/lgb_tdoa_v2.txt')

total_fail = []
for id in plane_id:
    id = 1721
    sens_erros = {}
    flight = train[train["aircraft"] == id]
    x = train_data.loc[flight["id"].values - 1, :]

    x["distance_signal_real_diff"] = x["d_aircraft_1"] - x["d_aircraft_2"]
    #x = x[abs(x["time_diff_1"] - x["time_diff_2"]) >= 0.05]
    x["min_time"] = 0
    x["d_min"] = 0

    for j,item in flight.iterrows():
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
        x.loc[j, "min_time"] = min* 1e-9 - item["timeAtServer"]

#     x["pred_lati"]
#
#     x["errors_1"] = x["time_diff_1"]  - x["min_time"] - (x["d_aircraft_1"] - x["d_min"]) /constants.Cair
#     x["errors_2"] = x["time_diff_2"]  - x["min_time"] - (x["d_aircraft_2"] - x["d_min"]) /constants.Cair
#
#     sens = x[["serial_1", "serial_2"]].values
#     sens = sens.flatten()
#     sens = np.unique(sens)
#
#     for i in sens:
#         type_a = sensor.loc[i - 1,"type"]
#         first = x[x["serial_1"] == i]
#         first = first.loc[~first.index.duplicated(keep='first')]
#         second = x[x["serial_2"] == i]
#         second = second.loc[~second.index.duplicated(keep='first')]
#
#         sens_erros[i] = np.concatenate((first["errors_1"].values, second["errors_2"].values), axis=0)
#
#         second["serial_2"] = second["serial_1"]
#         second["serial_1"] = i
#         #second["distance_signal_real_diff"] = second["d_aircraft_2"] - second["d_aircraft_1"]
#         first = pd.concat([first,second])
#         first["errors"] = first["distance_signal_diff"] - first["distance_signal_real_diff"]
#         anothers = np.unique(first["serial_2"].values)
#
#         for an in anothers:
#             anothers_d = first[first["serial_2"] == an]
#             type_b= sensor.loc[an - 1,"type"]
#             d_erros = 0
#             # if ((type_a == "dump1090" and type_b != "dump1090") or
#             #         (type_a != "dump1090" and type_b == "dump1090")):
#             #     d_erros = 299702535
#             tmp = anothers_d["errors"].values - d_erros
#             print(f"{i}-{type_a} {an}-{type_b} - {len(tmp)} - max : {np.max(tmp)}, min: {np.min(tmp)}, mean: {np.mean(tmp)}, std: {np.std(tmp)}")
#             print("__")
#     print("   ")
#     print("XXXXXXXXXXXXXXXX")
# for s in sens_erros:
#     tmp= sens_erros[s] * constants.Cair
#     print(
#         f"{s} - {len(tmp)} - max : {np.max(tmp)}, min: {np.min(tmp)}, mean: {np.mean(tmp)}, std: {np.std(tmp)}")
#     print("__")
#
#
#
# total_larger = []
#
# for i in np.arange(15) + 1:
#     x = pd.read_pickle(f"train/TDOA/{i}.pkl")
#     x["distance_signal_real_diff"] = abs(x["d_aircraft_1"] - x["d_aircraft_2"])
#     total_larger = np.concatenate((total_larger, x["distance_signal_real_diff"]/x["distance_diff"]), axis=0)

lati_grip = torch.linspace(40, 55, 3000).cuda()
long_grip = torch.linspace(0, 15, 3000).cuda()
lati_grip, long_grip = torch.meshgrid([lati_grip, long_grip])
delta_x = []
delta_y = []
for i in range(10):
    first_point = flight.iloc[0 + i:21 + i,:]
    first_point_data = x.loc[first_point["id"].values - 1, :]
    first_point = first_point.reset_index()
    total_time = first_point.loc[len(first_point) - 1,"timeAtServer"] - first_point.loc[0,"timeAtServer"]

    distance_travel = total_time * plane_speed

    height = np.mean(first_point["baroAltitude"])
    x_grid, y_grid, z_grid = geodesy.llh2ecef_torch([lati_grip, long_grip, height])

    timestamp = 0
    d_errors = None
    for j, item in first_point.iterrows():
        data = first_point_data.loc[[item["id"] - 1],:]
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

    d_errors = d_errors.flatten()
    sorted, index = torch.sort(d_errors)

    # lati_grip = lati_grip.flatten()
    # long_grip = long_grip.flatten()
    delta_x.append([np.mean(lati_grip.flatten()[index][:4].cpu().numpy()), first_point["latitude"].values[-1]])
    delta_y.append([np.mean(long_grip.flatten()[index][:4].cpu().numpy()), first_point["longitude"].values[-1]])
    #
    # new_lati = lati_grip[index][0]
    # new_long = long_grip[index][0]
np.log10()
FSPL (dB) = 20log10(d)+ 20log10(f) + 32.44

delta_grip = 2000
lati_grip = torch.linspace(new_lati - 0.2, new_lati + 0.2, delta_grip).cuda()
long_grip = torch.linspace(new_long - 0.2, new_long + 0.2, delta_grip).cuda()

a = geodesy.llh2ecef([50.0107, 8.52628, 0])
b = geodesy.llh2ecef([0.4/delta_grip, 0, 0])
distance_per_cell = geodesy.ecef_distance(a, b)

lati_grip, long_grip = torch.meshgrid([lati_grip, long_grip])
x_grid, y_grid, z_grid = geodesy.llh2ecef_torch([lati_grip, long_grip, height])

timestamp = 0
d_errors = None
max_point = 1000
xyz = []
map_seri = {}

distance_move_max = []
distance_move_min = []
for j, item in first_point.iterrows():

    data = first_point_data.loc[[item["id"] - 1],:]
    distance_move = (item["timeAtServer"] - timestamp) * plane_speed*0.99
    min_distance_move = (item["timeAtServer"] - timestamp) * plane_speed*0.88

    timestamp = item["timeAtServer"]

    # if j == 0:
    #     xyz = data[["x", "y", "z"]].values[0]
    # else:
    #     print(geodesy.ecef_distance(data[["x", "y", "z"]].values[0], xyz))
    #     xyz = data[["x", "y", "z"]].values[0]
    #
    #     print(distance_move)
    #     print()

    #print(distance_move)
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

    distance_move_max.append(distance_move)
    distance_move_min.append(min_distance_move)

    tmp = d_errors.flatten()
    sorted, index = torch.sort(tmp)
    index = index[:max_point]

    p_x = index / delta_grip
    p_y = index % delta_grip
    lat_right = ((lati_grip[p_x, p_y] >= item["latitude"] - 0.0001) & (lati_grip[p_x, p_y] <= item["latitude"] + 0.0001))
    lat_right = ((long_grip[p_x, p_y] >= item["longitude"] - 0.0001) & (long_grip[p_x, p_y] <= item["longitude"] + 0.0001) & lat_right)
    print(torch.sum(lat_right))
    if j == 0:
        p1 = torch.stack((p_x, p_y), dim=1)
        map_seri[j + 1] = p1

    if j == 1:
        p2 = torch.stack((x_grid[p_x, p_y], y_grid[p_x, p_y], z_grid[p_x, p_y]), dim=1)

        p1 = torch.stack((x_grid[p1[:, 0], p1[:, 1]], y_grid[p1[:, 0], p1[:, 1]], z_grid[p1[:, 0], p1[:, 1]]),
                         dim=1)
        p1 = p1.unsqueeze(-1)
        p1 = p1.repeat(1, 1, max_point)

        distance = torch.sqrt((p2[:, 0] - p1[:, 0, :])**2 + (p2[:, 1] - p1[:, 1, :])**2 + (p2[:, 2] - p1[:, 2, :])**2)
        location = (distance <= distance_move) & (distance >= min_distance_move)

        location = torch.sum(location, dim=0)
        location = location != 0
        torch.sum(location)

        location_2 = torch.sum((distance <= distance_move) & (distance >= min_distance_move), dim=1)
        location_2 = location_2 != 0

        map_seri[1] = torch.stack((map_seri[1][location_2, 0], map_seri[1][location_2, 1]), dim=1)
        map_seri[2] = torch.stack((p_x[location], p_y[location]), dim=1)
    # if j == 2:
    #     break
    if j >= 2:
        p3_ = torch.stack((x_grid[p_x, p_y], y_grid[p_x, p_y], z_grid[p_x, p_y]), dim=1)
        for idx in range(j):
            p_1 = map_seri[idx + 1]
            p_ = torch.stack((x_grid[p_1[:, 0], p_1[:, 1]], y_grid[p_1[:,0], p_1[:,1]], z_grid[p_1[:,0], p_1[:,1]]), dim=1)
            max_shift_stack = np.sum(distance_move_max[idx + 1:])
            min_shift_stack = max_shift_stack *0.985
            p_ = p_.unsqueeze(-1)
            p_ = p_.repeat(1, 1, len(p3_))

            distance = torch.sqrt((p3_[:, 0] - p_[:, 0, :])**2 + (p3_[:, 1] - p_[:, 1, :])**2
                                    + (p3_[:, 2] - p_[:, 2, :])**2)
            distance.shape
            location = torch.sum((distance <= max_shift_stack) & (distance >= min_shift_stack), dim=0)
            location = location != 0
            p3_ = p3_[location, :]
            p_x = p_x[location]
            p_y = p_y[location]

            location_2 = torch.sum((distance <= max_shift_stack) & (distance >= min_shift_stack), dim=1)
            location_2 = location_2 != 0
            map_seri[idx + 1] = torch.stack((p_1[location_2,0], p_1[location_2,1]), dim=1)

        p3 = torch.stack((p_x, p_y), dim=1)

        map_seri[j + 1] = p3
        print(len(map_seri[j + 1]))
        # p2 = p2.unsqueeze(-1)
        # p2 = p2.repeat(1, 1, len(p3))
        # distance = torch.sqrt(((p3[:, 0] - p2[:, 0, :])**2 + (p3[:, 1] - p2[:, 1, :])**2).float())
        # location = torch.sum((distance <= max_shift) & (distance >= min_shift), dim=0)
        # location = location != 0
        # p3 = p3[location, :]
        #
        # location_2 = torch.sum((distance <= max_shift) & (distance >= min_shift), dim=1)
        # location_2 = location_2 != 0
        # p2 = p2[location_2, :, 0]

        # p1 = p2
        # p2 = p3
        # print(len(p3))


        # p3[:, 0].shape
        # p2[1, 0, :]
        # print(torch.min(distance))


    #counter += 1

lati = train_1["latitude"].values
long = train_1["longitude"].values
np.max(lati)
np.min(lati)
np.max(long)
np.min(long)
np.sum((long < 15) & (long > 0))/len(long) * 100
np.sum((lati < 54) & (lati > 40))/len(long) * 100

450 * 450
tmp_c = map == 1
torch.where(tmp_c)
torch.sum(tmp_c)
torch.argmax(map)
torch.max(map)

96991// delta_grip
96991% delta_grip
lat_right = ((lati_grip[map_seri[j+1][:,0], map_seri[j+1][:,1]] >= item["latitude"] - 0.0001) & (lati_grip[map_seri[j+1][:,0], map_seri[j+1][:,1]] <= item["latitude"] + 0.0001))
lat_right = ((long_grip[map_seri[j+1][:,0], map_seri[j+1][:,1]] >= item["longitude"] - 0.0001) & (long_grip[map_seri[j+1][:,0], map_seri[j+1][:,1]] <= item["longitude"] + 0.0001) & lat_right)
print(torch.sum(lat_right))
torch.mean(long_grip[map_seri[j+1][:,0],map_seri[j+1][:,1]]).cpu().numpy()
torch.mean(lati_grip[map_seri[j+1][:,0],map_seri[j+1][:,1]]).cpu().numpy()

lati_grip[215,241]
map[95,107]
740- 2**9

lati_grip = lati_grip.flatten()
long_grip = long_grip.flatten()
new_lati = lati_grip[index][0]
new_long = long_grip[index][0]
first_point["latitude"]
first_point["longitude"]
first_point["timeAtServer"]
0.408 * plane_speed



ax = geodesy.llh2ecef([0, 0, 0])
b = geodesy.llh2ecef([49.518936 -  49.518826, 0.0000, 0])
geodesy.ecef_distance(ax, b)
x = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,15, 16]).view(4, 4)
torch.roll()
300 000
x = np.array([[1, 2, 3],[4, 5, 6]])

np.pad(x,((0,0),(1,0)), mode='constant')[:, :-1]

b = torch.Tensor([1,2,3])
b.shape
b = b.unsqueeze(-1)
b = b.repeat(1,3)

c = torch.Tensor([10,11,12])
import vg

cccc =  c -b
l = cccc <= 8
l = torch.sum(l, dim=0)
a = np.array([3,3])
b = np.array([1.5,1.5])
c = np.array([0,0])
v1 = a - b
v2 = c - b
p1 = np.sqrt((a[0] -b[0]) ** 2 + (a[1] -b[1]) ** 2)
p2 = np.sqrt((c[0] -b[0]) ** 2 + (c[1] -b[1]) ** 2)
p3 = np.sqrt((a[0] -c[0]) ** 2 + (a[1] -c[1]) ** 2)

np.arccos((p1 **2 + p2 **2 - p3 ** 2) / (2 * p1 * p2))
a = train_data["signal_strength_1"]
b = train_data["d_aircraft_1"]
c = abs(a - 20 * np.log10(b)) - 32.44
np.mean(c)
np.min(c)
c = c/20
np.mean(c)
x = 10 ** c
np.median(x)
np.log10(100)
d = 20 * np.log10(b) + 20 * np.log10(1090) + 32.44
np.max(a)
np.sum(a > d)
d = 10**(c)
mean_squared_error(b, d)
c = np.abs((a - 20 * np.log10(1090)))/20
c = np.clip(c, 0, 6)
10 ** 6
1000 000
np.min(d)
10 **6

# 27.55 - (20 * np.log10(frequency)) + signalLevel) = np.log10(b)


649 346
train = []
for idx in np.arange(14) + 1:
    x = pd.read_pickle(f"train/TDOA/{idx}.pkl")
    llh = geodesy.ecef2llh([x["x"].values, x["y"].values, x["z"].values])
    x = x[["serial_2", "time_diff_2", "d_aircraft_2", "id"]]
    #print(np.sum(x["baroAltitude"] > x["z"]))
    # x = x[((x["x"] <= 48.8874) & (x["x"] >= 48.8474))]
    # x = x[((x["y"] <= 8.16452) & (x["y"] >= 8.08452))]
    print(x.shape)
    train.append(x)
train = pd.concat(train)
train.shape
np.sum(((train["x"] <= 48.946) & (train["x"] >= 48.83)
        & (train["y"] <= 8.3245) & (train["y"] >= 7.8245)))

np.sum(((train["x"] <= 48.946) & (train["x"] >=48.83)))
np.sum(((train["y"] <= 8.3245) & (train["y"] >=7.8245)))

np.mean(train["y"])
np.mean(train["x"])


train_set = train[((train["x"] <= 48.946) & (train["x"] >= 48.83)
        & (train["y"] <= 8.3245) & (train["y"] >= 7.8245))]
7364 / np.sqrt(2)

a = train["serial_1"].values
b = train["serial_2"].values
a  = np.concatenate((a,b))
a = Counter(a)

x = train[train["serial_1"] == 463]
a = x["d_aircraft_1"].values
b = x["signal_strength_1"].values
b = x["time_diff_1"].values


x = train[train["serial_2"] == 463]
a_1 = x["d_aircraft_2"].values
b_1 = x["signal_strength_2"].values
b_1 = x["time_diff_2"].values
a  = np.concatenate((a,a_1))
b  = np.concatenate((b,b_1))

index = np.argsort(a)
d = savgol_filter(b, 23, 2)

# create figure and axis objects with subplots()
fig,ax = plt.subplots()
# make a plot
ax.plot(a[index], color="red")
ax.set_xlabel("year",fontsize=14)
ax.set_ylabel("lifeExp",color="red",fontsize=14)

ax2=ax.twinx()
ax2.plot(d[index],color="blue")
ax2.set_ylabel("gdpPercap",color="blue",fontsize=14)
plt.show()
# save the plot as a file
test = pd.read_pickle("test/test.pkl")

np.min(test["latitude"])
np.min(test["latitude"])
np.max(test["longitude"])
np.min(test["longitude"])


c = 20 * np.log10((a * 4 * np.pi * 1090 * (10**6))/constants.Cair)
np.min(c)
d = c- b
np.min(d)




train = pd.read_pickle("train/training_1.pkl")
train = train[pd.isna(train["longitude"])]
train.shape
a = train["aircraft"].values
a = Counter(a)
a = train[train["aircraft"] == 1675]
a["timeAtServer"]

ids = train["id"].values
ids = np.unique(ids)

lowest_tdoa = []
train = train.set_index("id")
for id in ids:
    x = train[train["id"] == id]
    l = np.argmin(x["time_diff_2"])
    lowest_tdoa.append(x.iloc[0])
    if id % 100000 == 0:
        print(len(lowest_tdoa))
lowest_tdoa[0]
x = train.iloc[:10000]
x = train.sort_values(by='time_diff_2', ascending=True).groupby(level=0).first()
a.shape
np.sum(a["d_aircraft_2"] > 300 000)/len(a) *100
1024
np.sum(a["time_diff_2"] < 0.9)/len(a) *100
x = a["d_aircraft_2"]/constants.Cair
np.mean(x)
0.0002857091545403202



