# -*- mode: python; indent-tabs-mode: nil -*-

# Part of mlat-server: a Mode S multilateration server
# Copyright (C) 2015  Oliver Jowett <oliver@mutability.co.uk>

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
Utility functions to convert between coordinate systems and calculate distances.
"""

import math
from . import constants
import torch
import numpy as np
# WGS84 ellipsoid Earth parameters
WGS84_A = 6378137.0
WGS84_F = 1.0/298.257223563
WGS84_B = WGS84_A * (1 - WGS84_F)
WGS84_ECC_SQ = 1 - WGS84_B * WGS84_B / (WGS84_A * WGS84_A)
WGS84_ECC = math.sqrt(WGS84_ECC_SQ)

# Average radius for a spherical Earth
SPHERICAL_R = 6371e3

# Some derived values
_wgs84_ep = math.sqrt((WGS84_A**2 - WGS84_B**2) / WGS84_B**2)
_wgs84_ep2_b = _wgs84_ep**2 * WGS84_B
_wgs84_e2_a = WGS84_ECC_SQ * WGS84_A


def llh2ecef(llh):
    """Converts from WGS84 lat/lon/height to ellipsoid-earth ECEF"""

    lat = llh[0] * constants.DTOR
    lng = llh[1] * constants.DTOR
    alt = llh[2]

    slat = np.sin(lat)
    slng = np.sin(lng)
    clat = np.cos(lat)
    clng = np.cos(lng)

    d = np.sqrt(1 - (slat * slat * WGS84_ECC_SQ))
    rn = WGS84_A / d

    x = (rn + alt) * clat * clng
    y = (rn + alt) * clat * slng
    z = (rn * (1 - WGS84_ECC_SQ) + alt) * slat

    return x, y, z

def llh2ecef_torch(llh):
    """Converts from WGS84 lat/lon/height to ellipsoid-earth ECEF"""

    lat = llh[0] * constants.DTOR
    lng = llh[1] * constants.DTOR
    alt = llh[2]
    del llh
    slat = torch.sin(lat)
    slng = torch.sin(lng)
    clat = torch.cos(lat)
    clng = torch.cos(lng)

    d = torch.sqrt(1 - (slat * slat * WGS84_ECC_SQ))
    rn = WGS84_A / d

    x = (rn + alt) * clat * clng
    y = (rn + alt) * clat * slng
    z = (rn * (1 - WGS84_ECC_SQ) + alt) * slat

    return x, y, z


def ecef2llh(ecef):
    "Converts from ECEF to WGS84 lat/lon/height"

    x, y, z = ecef[0], ecef[1], ecef[2]

    lon = np.arctan2(y, x)

    p = np.sqrt(x**2 + y**2)
    th = np.arctan2(WGS84_A * z, WGS84_B * p)
    lat = np.arctan2(z + _wgs84_ep2_b * np.sin(th)**3,
                     p - _wgs84_e2_a * np.cos(th)**3)

    N = WGS84_A / np.sqrt(1 - WGS84_ECC_SQ * np.sin(lat)**2)
    alt = p / np.cos(lat) - N

    return (lat * constants.RTOD, lon * constants.RTOD, alt)


def greatcircle(p0, p1):
    """Returns a great-circle distance in metres between two LLH points,
    _assuming spherical earth_ and _ignoring altitude_. Don't use this if you
    need a distance accurate to better than 1%."""

    lat0 = p0[0] * constants.DTOR
    lon0 = p0[1] * constants.DTOR
    lat1 = p1[0] * constants.DTOR
    lon1 = p1[1] * constants.DTOR
    return SPHERICAL_R * math.acos(
        math.sin(lat0) * math.sin(lat1) +
        math.cos(lat0) * math.cos(lat1) * math.cos(abs(lon0 - lon1)))

def greatcircle_torch(p0, p1):
    """Returns a great-circle distance in metres between two LLH points,
    _assuming spherical earth_ and _ignoring altitude_. Don't use this if you
    need a distance accurate to better than 1%."""

    lat0 = p0[0] * constants.DTOR
    lon0 = p0[1] * constants.DTOR
    lat1 = p1[0] * constants.DTOR
    lon1 = p1[1] * constants.DTOR
    return SPHERICAL_R * torch.acos(
        torch.sin(lat0) * math.sin(lat1) +
        torch.cos(lat0) * math.cos(lat1) * torch.cos(abs(lon0 - lon1)))


# direct implementation here turns out to be _much_ faster (10-20x) compared to
# scipy.spatial.distance.euclidean or numpy-based approaches
def ecef_distance(p0, p1):
    """Returns the straight-line distance in metres between two ECEF points."""
    return np.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2 + (p0[2] - p1[2])**2)

def ecef_distance_torch(p0, p1):
    """Returns the straight-line distance in metres between two ECEF points."""
    return torch.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2 + (p0[2] - p1[2])**2)

def ecef_distance_torch_stack(p0, p1):
    """Returns the straight-line distance in metres between two ECEF points."""
    return torch.sqrt((p0[:,:,0] - p1[0])**2 + (p0[:,:,1] - p1[1])**2 + (p0[:,:,2] - p1[2])**2)