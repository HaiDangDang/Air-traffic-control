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
    k_nn = 100
    height_grip = 11582
    max_delta = 5
    max_delta_flight = 0.5
    max_grip = 23000

    sensor = pd.read_csv("train/sensors.csv")
    stack_sensor_pair = []
    len(stack_sensor_pair)

    sensor_pair = np.load("train/sensor_pair.npy")
    sensor_pair_done = np.load("train/sensor_pair_done.npy")
    sensor_pair_done = list(sensor_pair_done)
    for s_sss in sensor_pair:
        s_1 = s_sss[0]
        s_2 = s_sss[1]
        if not(s_1 in sensor_pair_done[0] and s_2 in sensor_pair_done[1]):
            train = []
            s_1 = 10
            s_2 = 129
            for i in np.arange(1) + 1:
                train_tmp = pd.read_pickle(f"train/TDOA/{i}.pkl")
                train_tmp = train_tmp[train_tmp["distance_signal_diff"] <= 1000000]
                idx_s = ((train_tmp["serial_1"] == s_1) & (train_tmp["serial_2"] == s_2) |
                         (train_tmp["serial_2"] == s_1) & (train_tmp["serial_1"] == s_2))
                train_tmp = train_tmp[idx_s]
                train.append(train_tmp)
            train = pd.concat(train)
            print(train.shape)
            # x = train["serial_1"].values
            # y = train["serial_2"].values
            # dict_all_sensor = np.concatenate((x,y))
            # x = Counter(x)

            llh_1 = sensor[sensor["serial"] == s_1][["latitude", "longitude", "height"]].values[0]
            llh_2 = sensor[sensor["serial"] == s_2][["latitude", "longitude", "height"]].values[0]
            max_lati = np.max([llh_1[0], llh_2[0]])
            min_lati = np.min([llh_1[0], llh_2[0]])
            if max_delta < (max_lati - min_lati):
                delta_lati = max_delta/1.5
                max_lati = (max_lati - min_lati)/2
                min_lati = (max_lati - min_lati)/2
            else:
                delta_lati = (max_delta - (max_lati - min_lati))/2
            max_lati += delta_lati
            min_lati -= delta_lati

            max_long = np.max([llh_1[1], llh_2[1]])
            min_long = np.min([llh_1[1], llh_2[1]])
            if max_delta < (max_long - min_long):
                delta_long = max_delta/1.5
                max_long = (max_long - min_long)/2
                min_long = (max_long - min_long)/2
            else:
                delta_long = (max_delta - (max_long - min_long)) / 2

            max_long += delta_long
            min_long -= delta_long

            llh_fight = geodesy.ecef2llh([train["x"].values, train["y"].values, train["z"].values])
            train["latitude"] = llh_fight[0]
            train["longitude"] = llh_fight[1]
            train["geoAltitude"] = llh_fight[2]

            idx_s = ((train["latitude"] <= max_lati - max_delta_flight) & (train["latitude"] >= min_lati + max_delta_flight) &
             (train["longitude"] <= max_long - max_delta_flight) & (train["longitude"] >= min_long + max_delta_flight))
            train = train[idx_s]
            print(train.shape)
            if len(train) == 0:
                sensor_pair_done[0] = np.append(sensor_pair_done[0], s_1)
                sensor_pair_done[1] = np.append(sensor_pair_done[1], s_2)
                np.save("train/sensor_pair_done.npy", sensor_pair_done)
                continue

            print(f"{s_1} {s_2} {max_lati} - {min_lati}|{max_long} - {min_long}")

            lati_grip = torch.linspace(min_lati, max_lati, max_grip)
            long_grip = torch.linspace(min_long, max_long, max_grip)

            x_grid, y_grid = torch.meshgrid([lati_grip, long_grip])
            x_grid, y_grid, z_grid = geodesy.llh2ecef_torch([x_grid, y_grid, np.mean(train["geoAltitude"])])


            xyz_1 = geodesy.llh2ecef(llh_1)
            xyz_2 = geodesy.llh2ecef(llh_2)
            d_1 = geodesy.ecef_distance_torch([x_grid, y_grid, z_grid], xyz_1)
            d_2 = geodesy.ecef_distance_torch([x_grid, y_grid, z_grid], xyz_2)
            BIG_D = d_1 - d_2
            BIG_D = BIG_D.cuda()
            xyz_stack = torch.cat((torch.unsqueeze(x_grid, -1),torch.unsqueeze(y_grid, -1),torch.unsqueeze(z_grid, -1)),dim=-1)
            xyz_stack = xyz_stack.cuda()

            nn_S = []
            nn_B = []
            counter = 0
            del d_1, d_2, x_grid, y_grid, z_grid

            c_time = time.time()
            counter = 0
            for i, item in train.iterrows():
                rotate = 1
                TODA_dist = abs(item["d_aircraft_1"] - item["d_aircraft_2"])
                if item["serial_1"] == s_2:
                    rotate = - 1
                max_lat_f = item["latitude"] + max_delta_flight
                min_lat_f = item["latitude"] - max_delta_flight
                max_long_f = item["longitude"] + max_delta_flight
                min_long_f = item["longitude"] - max_delta_flight
                bound_l_lat = int(max_grip*(min_lat_f - min_lati)/max_delta)
                bound_u_lat = int(max_grip*(max_lat_f - min_lati)/max_delta)
                bound_l_long = int(max_grip*(min_long_f - min_long)/max_delta)
                bound_u_long = int(max_grip*(max_long_f - min_long)/max_delta)

                d = BIG_D[bound_l_lat:bound_u_lat, bound_l_long: bound_u_long]
                abs_d = torch.abs(d)

                location = (TODA_dist - 30 < abs_d) & (abs_d < TODA_dist + 30)
                #location = torch.gt(abs_d, TODA_dist - 50) * torch.lt(abs_d, TODA_dist + 50)
                time_sum = torch.abs(item["distance_signal_diff"] - d[location] * rotate)
                #
                sorted, indices = torch.sort(time_sum)
                location_time = torch.cat((xyz_stack[bound_l_lat:bound_u_lat, bound_l_long: bound_u_long,:][location,:][indices,:]
                                             ,torch.unsqueeze(sorted, -1)), dim=1)

                location_time = location_time[:k_nn,].cpu().numpy()

                data = item.values
                nn_S.append([data, location_time.flatten()])
                counter += 1
                if counter % 10000 == 0:
                    print(counter)
                    print(time.time() -c_time)
                    c_time = time.time()
            del BIG_D, xyz_stack, abs_d, location, d, time_sum
            torch.cuda.empty_cache()
            with open(f'train/sen_pair/{s_1}-{s_2}.pkl','wb') as f:
                pickle.dump(nn_S, f)
                print("SAVED")
            sensor_pair_done[0] = np.append(sensor_pair_done[0],s_1)
            sensor_pair_done[1] = np.append(sensor_pair_done[1],s_2)
            np.save("train/sensor_pair_done.npy",sensor_pair_done)

            del nn_S



    torch.eq
    tmp = torch.tensor([1,3,5,7])
    tmp2 = tmp.div(2)
    # a = train["d_aircraft_1"]
    # np.std(a)
    # max_min_ll = pd.read_pickle("train/max_min_LL.pkl")
    # all_flight = train["id"].values
    # all_flight = Counter(all_flight).keys()
    # #
    # all_flight_dict_store = []
    # all_find = []
    # #
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
    # not_found = []
    # for id in all_flight:
    #     c_delta = 0.5
    #     check = True
    #     location = None
    #     time_sum = None
    #     id_fight = train[train["id"] == id]
    #     max_min = max_min_ll[max_min_ll["id"] == id]
    #     max_lat = max_min["latitude"].values[0] + c_delta
    #     min_lat = max_min["latitude"].values[0] - c_delta
    #     max_long = max_min["longitude"].values[0] + c_delta
    #     min_long = max_min["longitude"].values[0] - c_delta
    #     height_grip = id_fight["baroAltitude"].values[0]
    #
    #     lati_grip = torch.linspace(min_lat, max_lat, grip_size).cuda()
    #     long_grip = torch.linspace(min_long, max_long, grip_size).cuda()
    #
    #     lati_grip, long_grip = torch.meshgrid([lati_grip, long_grip])
    #     x_grid, y_grid, z_grid = geodesy.llh2ecef_torch([lati_grip, long_grip, height_grip])
    #
    #     for i, item in id_fight.iterrows():
    #         s_1 = item[["x_1", "y_1", "z_1"]].values
    #         s_2 = item[["x_2", "y_2", "z_2"]].values
    #         d_1 = geodesy.ecef_distance_torch([x_grid, y_grid, z_grid], s_1)
    #         d_2 = geodesy.ecef_distance_torch([x_grid, y_grid, z_grid], s_2)
    #
    #         if time_sum is None:
    #             time_sum = d_1 * 0
    #         d = d_1 - d_2
    #         TODA_dist = item["d_aircraft_1"] - item["d_aircraft_2"]
    #         local_location = (TODA_dist-46 < d) & (d< TODA_dist + 46) & (d_1 < 1000000) & (d_2 < 1000000)
    #         if location is None:
    #             location = local_location
    #         else:
    #             location = local_location & location
    #         tmp_x = d[location] / constants.Cair
    #         time_sum[location] += torch.sqrt((item["TDOA"] - d[location] / constants.Cair) ** 2) * 10e9
    #
    #     sorted, indices = torch.sort(time_sum[location])
    #     location_time = torch.stack((x_grid[location][indices], y_grid[location][indices],
    #                                  z_grid[location]
    #                                  , time_sum[location][indices]), dim=1)
    #
    #     location_time = location_time[:k_nn].cpu().numpy()
    #     # all_find.append([torch.sum(location),np.sum(b)])
    #     all_flight_dict_store.append(location_time)
    #     if len(location_time) ==0:
    #         not_found.append(id)
    #         print(f"NOT FOUND {len(not_found)}")
    #     if id % 100 == 0:
    #         print(id)
    #         print(time.time() -c_time)
    #         c_time = time.time()

        # all_find = np.array(all_find)
        # x = all_find[:,0]
        # x = all_find[:, -1] - all_find[:,-4]
        # np.min(x)
        # len(all_find)
        # if id == 10000:
        #     break
    #
    # #
    # all_find = np.array(all_find)
    # all_find[all_find > 0]
    # all_flight_dict_store[220]
    # #
    # # a = x_grid <= 4114084
    # #
    # # mean_squared_error([564156.1875,4114397.7500,0],[563765.899786694, 41 14084.0394785465, 0])
    # #
    # # 4042234.0000, 4036351.5000
    # # 620913.5000, 625284.0625
    # # np.sqrt(mean_squared_error([32,-13.0000,0],[32.0000, -12.3500,0]))
    a = geodesy.llh2ecef([0,0,0])
    b = geodesy.llh2ecef([0.06,0,0])
    geodesy.ecef_distance(a,b)
    20 * 20
    # geodesy.llh2ecef([53.625767, 14.411346, 6537])
    # 20* 20
    # item["x"]
    # 165
    # # geodesy.ecef_distance(a,b)
    # # 311
    # # 277
    # np.sqrt(311 **2 + 277**2)
