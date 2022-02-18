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

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error
#
train = pd.read_pickle("train/train_master_fix_index.pkl")


sensors = pd.read_csv("train/sensors.csv")
# tdoa_df = pd.DataFrame(columns=["serial_1", "time_diff_1", "signal_strength_1", "x_1", "y_1", "z_1", " type_1", "d_aircraft_1",
#                                 "serial_2", "time_diff_2", "signal_strength_2", "x_2", "y_2", "z_2", " type_2", "d_aircraft_2",
#                                 "TDOA", "distance_diff", "distance_signal_real_diff", "distance_signal_diff",
#                                  'x', 'y', 'z','baroAltitude'])
#
columns=["serial_1", "time_diff_1", "signal_strength_1", "x_1", "y_1", "z_1", " type_1", "d_aircraft_1",
        "serial_2", "time_diff_2", "signal_strength_2", "x_2", "y_2", "z_2", " type_2", "d_aircraft_2",
        "TDOA", "distance_diff", "distance_signal_real_diff", "distance_signal_diff",
         'x', 'y', 'z','baroAltitude']
# tdoa_df = []
#
sensors_dict = {}
for s in sensor_use:
    sensor = sensors[sensors["serial"] == s]
    llh = sensor[["latitude", "longitude", "height"]].values[0]
    x, y, z = geodesy.llh2ecef(llh)
    type = 0
    if sensor["type"].values[0] == "GRX1090":
        type = 1

    sensors_dict[s] = {"xyz": [x, y, z],
                       "type": type}
#
#
# def pasr_data(s_1, s_2, measure_1, measure_2, aircraft):
#     dict_s1 = sensors_dict[s_1]
#     dict_s2 = sensors_dict[s_2]
#
#     timestamp_1 = measure_1[1] * 10**-9
#     timestamp_2 = measure_2[1] * 10**-9
#
#     signal_strength_1 = measure_1[2]
#     signal_strength_2 = measure_2[2]
#
#     time_diff_1 = timestamp_1 - aircraft["timeAtServer"]
#     time_diff_2 = timestamp_2 - aircraft["timeAtServer"]
#
#     TDOA = timestamp_1 - timestamp_2
#     assert TDOA >= 0
#
#     distance_signal_diff = TDOA * constants.Cair
#
#     d_aircraft_1 = geodesy.ecef_distance(aircraft["xyz"], dict_s1["xyz"])
#     d_aircraft_2 = geodesy.ecef_distance(aircraft["xyz"], dict_s2["xyz"])
#
#     distance_signal_real_diff = np.abs(d_aircraft_1 - d_aircraft_2)
#
#     distance_diff = geodesy.ecef_distance(dict_s1["xyz"], dict_s2["xyz"])
#
#     tdoa_df.append([s_1, time_diff_1, signal_strength_1, dict_s1["xyz"][0], dict_s1["xyz"][1], dict_s1["xyz"][2],
#                     dict_s1["type"], d_aircraft_1,
#                     s_2, time_diff_2, signal_strength_2, dict_s2["xyz"][0], dict_s2["xyz"][1], dict_s2["xyz"][2],
#                     dict_s2["type"], d_aircraft_2,
#                     TDOA, distance_diff, distance_signal_real_diff, distance_signal_diff,
#                     aircraft["xyz"][0], aircraft["xyz"][1], aircraft["xyz"][2], aircraft["baroAltitude"]])
#
#
d_mins = []
train.shape
13 711 223
id = []
id_c = id
id_2 = id.copy()
len(id)
for i, item in train.iterrows():
    measur = item["measurements"]
    # pairs = []
    # llh = item[["latitude", "longitude", "geoAltitude"]].values
    # x, y, z = geodesy.llh2ecef(llh)
    # aircraft = {"xyz":  [x, y, z],
    #             "baroAltitude": item["baroAltitude"],
    #             "timeAtServer": item["timeAtServer"]}

    assert len(measur) >= 2
    for j in range(len(measur) - 1):
        for _j in range(1, len(measur)):
            id.append(i)
    if i % 10000 == 0:
        print(i)
            s_1 = measur[j][0]
            s_2 = measur[_j][0]

            if measur[j][1] > measur[_j][1]:
                pasr_data(s_1, s_2, measur[j], measur[_j], aircraft)
            else:
                pasr_data(s_2, s_1, measur[_j], measur[j], aircraft)
    if i % 10000 == 0:
        print(i)
    if i % 1000000 == 0 and i !=0 or i == 13711222:

        tdoa_df = pd.DataFrame(tdoa_df, columns=columns)
        tdoa_df.to_pickle(f"train/TDOA/{i}.pkl")
        tdoa_df = []
        print()
        print("SAVE FILE")
        print()
1 260 000
#
# tdoa_df = pd.DataFrame(tdoa_df, columns=columns)
#
# tdoa_df_101.shape
# tdoa_df_101 = tdoa_df[tdoa_df["serial_1"] == 101.0]
# fig = plt.figure
# fig(num=None, figsize=(30, 6), dpi=80, facecolor='w', edgecolor='k')
# ax = plt.plot(tdoa_df_101["distance_signal_real_diff"] /1000, c="r")
# plt.show()
np.savetxt('data.csv', id, delimiter=',')
x = np.loadtxt('data.csv', delimiter=',')
len(x)
x_train = []
x_test = []
y_train = []
y_test = []
sc = StandardScaler()
train_data = []
curren
x["id"] =
id_c = np.delete
a = [1,3,4,5,6,7,8]
x = a[:3]
y = a[3:]
len(x)
id = x
id_c = x.copy()
for i in np.arange(14) + 1:
    x = pd.read_pickle(f"train/TDOA/{i}.pkl")
    x["id"] = id[:len(x)]
    x.to_pickle(f"train/TDOA/{i}_new.pkl")
    id = id[len(x):]

    x = x[x["distance_signal_diff"] <= 1000000]
    x = x[x["d_aircraft_1"] <= 8000000]
    x = x[x["d_aircraft_1"] <= 1000000]

    x = x.loc[:,["d_aircraft_1","d_aircraft_2", "distance_diff"]]
    train_data.append(x)
    train_data = pd.concat(train_data)
    np.max(train_data["distance_diff"])
    x = x[x["distance_signal_diff"] <= 1000000]
    x, _ = np.split(x, [int(.3 * len(x))])
    x[["x", "y", "z",
        "x_1", "y_1", "z_1",
           "x_2", "y_2", "z_2"
           , "distance_diff", "distance_signal_diff", "distance_signal_real_diff", "d_aircraft_1", "d_aircraft_2"]] = \
        x[["x", "y", "z",
            "x_1", "y_1", "z_1",
               "x_2", "y_2", "z_2"
               , "distance_diff", "distance_signal_diff", "distance_signal_real_diff","d_aircraft_1", "d_aircraft_2"]] / 1000
    #x["per"] = x["d_aircraft_1"] / x["d_aircraft_2"]
    #x["TDOA"] = x["TDOA"] * 10e6
    y = x.loc[:, "distance_signal_real_diff"].values
    x = x.drop(
        ["serial_1", "serial_2", "x", "y", "z", "d_aircraft_1",  "d_aircraft_2", "distance_signal_real_diff"], axis=1)
    features = x.keys()

    x = x.values

    x_train_t, x_test_t, y_train_t, y_test_t = train_test_split(x, y, test_size=0.1, random_state=10)

    if len(x_train) == 0:
        x_train_t = sc.fit_transform(x_train_t)
        x_test_t = sc.transform(x_test_t)
        x_train, x_test, y_train, y_test = x_train_t, x_test_t, y_train_t, y_test_t
    else:
        x_train_t = sc.transform(x_train_t)
        x_test_t = sc.transform(x_test_t)
        x_train = np.concatenate((x_train, x_train_t))
        x_test = np.concatenate((x_test, x_test_t))
        y_train = np.concatenate((y_train, y_train_t))
        y_test = np.concatenate((y_test, y_test_t))
    print(i)
print(x_train.shape)
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
lgb_params1['num_leaves'] = 126
lgb_params1['min_data_in_leaf'] = 200

lgb_params1['min_sum_hessian_in_leaf'] = 2e-2
lgb_params1['min_gain_to_split'] = 0
lgb_params1['bagging_fraction'] = 0.9
lgb_params1['feature_fraction'] = 0.8
lgb_params1['bagging_freq'] = 153
lgb_params1['metric'] = 'rmse'
#lgb_params1['num_threads'] = 14
# lgb_params1['n_estimators'] = 1000
# lgb_params1['colsample_bytree'] = 0.8
lgb_params1['max_depth'] = 12
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

x_val, x_test, y_val, y_test = train_test_split(x_test,y_test,test_size = 0.2, random_state = 0)


d_valid = lgb.Dataset(x_val, y_val)
d_train = lgb.Dataset(x_train, y_train)

watchlist = [d_train, d_valid]

lgb_model_ = lgb.train(lgb_params1, train_set=d_train, num_boost_round=30000,
                          valid_sets=watchlist, verbose_eval=100, early_stopping_rounds = 200,feature_name=list(features))
y_pred = lgb_model_.predict(x_test)

#lgb_model_.save_model("model/distance_diff_dist_1.txt")
#lgb_model_.save_model("model/distance_diff.txt")

lgb.plot_important(lgb_model_)
np.sqrt(mean_squared_error(y_test,y_pred))
a = abs(y_test - y_pred)
np.mean(a)
np.std(a)
y_test[1359597]
y_pred[1359597]

sns.distplot(a)
#plt.plot(np.sort(a))
plt.show()
x = np.sort(a)
np.std(x)
import pickle
pickle.dump(sc, open('model/distance_diff_SC.pkl','wb'))
bst = lgb.Booster(model_file='model.txt')

lgb.plot_importance(lgb_model_)
plt.show()

