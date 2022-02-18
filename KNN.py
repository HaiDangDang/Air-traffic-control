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


x_train = []
x_val = []
y_train = []
y_val = []
sc = StandardScaler()


for i in np.arange(7) + 1:
    x =np.load(f"train/min_sensor/{i}.npy", allow_pickle=True)
    x_t = x[:, 0]
    x_t = np.array([i.astype(float) for i in x_t])
    y_t = x[:, 1]
    y_t = np.array([i.astype(float) for i in y_t])
    #y_t = y_t[:, 0]

    val =np.load(f"train/min_sensor/{i}_val.npy", allow_pickle=True)
    x_v = val[:, 0]
    x_v = np.array([i.astype(float) for i in x_v])
    y_v = val[:, 1]
    y_v = np.array([i.astype(float) for i in y_v])
    #y_v = y_v[:, 0]

    if len(x_train) == 0:
        x_train = x_t
        y_train = y_t
        x_val = x_v
        y_val = y_v
    else:

        x_train = np.concatenate((x_train, x_t), axis=0)
        y_train = np.concatenate((y_train, y_t), axis=0)
        x_val = np.concatenate((x_val, x_v), axis=0)
        y_val = np.concatenate((y_val, y_v), axis=0)

    print(f"{x_train.shape} {y_train.shape}")

    print(f"{x_val.shape} {y_val.shape}")

tmp_train = y_train
tmp_val = y_val

y_train = tmp_train[:, 2] + 180

y_val = tmp_val[:, 2] + 180
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
#lgb_params1['num_class']= 16*16
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



d_valid = lgb.Dataset(x_val, y_val)
d_train = lgb.Dataset(x_train, y_train)

watchlist = [d_train, d_valid]

lgb_model_ = lgb.train(lgb_params1, train_set=d_train, num_boost_round=4500,
                          valid_sets=watchlist, verbose_eval=100, early_stopping_rounds = 200)

y_pred = lgb_model_.predict(x_val[:10000])
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