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
import random
x_train = []
x_test = []

# for i in np.arange(1) + 1:
#     x = np.load(f"train/TDOA/layer_2/{i}.npy")
#     x = random.sample(list(x), len(x))
#     if len(x_train) == 0:
#         x_train = x[:int(len(x)*0.999)]
#     else:
#         x_train = np.append(x_train, x[:int(len(x)*0.999)], axis=0)
#     print(i)
#     del x
for i in np.arange(1) + 1:
    a = np.load(f"train/TDOA/layer_2/main_d.npy")
    a = np.load(f"train/TDOA/layer_2/333.npy")
    x = x.sample(frac=1).reset_index(drop=True)
    y_train = x[["latitude", "longitude", "geoAltitude"]].values
    x = x.drop(["latitude", "longitude", "geoAltitude"], axis=1)
    x = x.values
    if len(x_train) == 0:
        x_train = x[:int(len(x)*0.999)]
    else:
        x_train = np.append(x_train, x[:int(len(x)*0.999)], axis=0)
    print(i)
    del x
a.shape
a = np.concatenate((a,b), axis=0)
a.shape
np.sum(a == -100)
8 375 880
72 484 200
2436 664
604035 * 4 * 30
2436664/24045240
x_train = a
y_train = x_train[:,0]
x_train = x_train[:,2:]
x_train.shape
idx = int(len(x_train)*0.9)
x_test = x_train[idx:]
y_test = y_train[idx:]

x_train = x_train[:idx]
y_train = y_train[:idx]


idx = int(len(x_test)*0.9)

x_val = x_test[:idx]
y_val = y_test[:idx]

x_test = x_test[idx:]
y_test = y_test[idx:]

y_val.shape

lgb_params1 = {}
lgb_params1['application'] = 'regression'
lgb_params1['boosting'] = 'gbdt'
lgb_params1['learning_rate'] = 0.095
lgb_params1['num_leaves'] = 103
#lgb_params1['min_data_in_leaf'] = 20

lgb_params1['min_sum_hessian_in_leaf'] = 2e-2
lgb_params1['min_gain_to_split'] = 0
lgb_params1['bagging_fraction'] = 0.7
lgb_params1['feature_fraction'] = 0.6
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

lgb_model_ = lgb.train(lgb_params1, train_set=d_train, num_boost_round=15000,
                          valid_sets=watchlist, verbose_eval=100,
                       early_stopping_rounds = 300)
# import joblib
# lgb_model_.save_model("model.pkl",num_iteration=lgb_model_.best_iteration)
x_train.shape
x_test.shape
y_pred = lgb_model_.predict(x_test)

# a = joblib.load('model.pkl')
# y_test[:10]
# y_pred[:10]
# y_test.shape
# bst = lgb.Booster(model_file='model.pkl')
# y_pred = bst.predict(x_test[:10000])
#
# #lgb_model_.save_model("model/distance_1_gbdt.txt")
# # lgb_model_.save_model("model/distance_diff.txt")
# # #
# len(x_val)
# a = abs(y_test - y_pred)
# np.std(a)
# np.sum(a < 0.3) / len(a)
# x = abs(y_test)
# np.sum(x < 0.3) / len(x)

# np.max(a)
# np.std(a)
# sns.distplot(abs(a))
# plt.show()
print(np.sqrt(mean_squared_error(y_test,y_pred)))
# # a = abs(y_test - y_pred)
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
# joblib.dump(lgb_model_, 'model/lgb_long_v1.pkl')
