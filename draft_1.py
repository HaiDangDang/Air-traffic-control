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

train = pd.read_pickle("train/training_1.pkl")
sensors = pd.read_csv("train/sensors.csv")
a = train.iloc[0,:]
filt = kalman.KalmanStateCV(1331)
filt.process_noise = 0.2
filt.min_tracking_receivers = 4
filt.outlier_mahalanobis_distance = 10.0

measurements = a["measurements"]
s_463 = sensors[sensors["serial"] == 463]
s_463 = s_463[["latitude", "longitude", "height"]].values[0]
measurements[0][0] =  geodesy.llh2ecef(s_463)


s_463 = sensors[sensors["serial"] == 424]
s_463 = s_463[["latitude", "longitude", "height"]].values[0]
measurements[1][0] = geodesy.llh2ecef(s_463)

s_463 = sensors[sensors["serial"] == 412]
s_463 = s_463[["latitude", "longitude", "height"]].values[0]
measurements[2][0] = geodesy.llh2ecef(s_463)

filt.update(a["timeAtServer"], measurements, a["baroAltitude"],3,3,[1, 1, 1], 4,1.2)

train.keys()