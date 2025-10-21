
import numpy as np
import folium
import json
import math
from matplotlib import pyplot as plt
from sklearn.preprocessing import QuantileTransformer
import matplotlib.colors as mcolors
import cmap
from matplotlib.colors import LinearSegmentedColormap

from math import sin, asin, cos, acos, radians, fabs, sqrt, pi, atan, tan
axis = 6378245.0
offset = 0.00669342162296594323
x_pi = pi * 3000.0 / 180.0
earthR = 6371000

def restore_matrix_36(square):
    ret = np.zeros((36 * 36, 3))
    for i in range(36):
        for d in range(3):
            ret[36 * i : 36 * (i + 1), d] = square[:, i, d]
    return ret[:1260, :]

def restore_matrix_36_1(square):
    ret = np.zeros((36*36))
    for i in range(36):
        ret[36 * i : 36 * (i + 1)] = square[:, i]
    return ret[:1260]

def restore_matrix_64(square):
    ret = np.zeros((64 * 64, 3))
    for i in range(64 // 3):
        ret[64 * i: 64 * (i + 1), :] = square[:, 3 * i: 3 * i + 3]
    return ret[:1260, :]

def restore_matrix(square):
    if square.shape == (36, 36, 3):
        return restore_matrix_36(square)
    elif square.shape == (64, 64):
        return restore_matrix_64(square)
    elif square.shape == (36, 36):
        return restore_matrix_36_1(square)
    
def delta(wgLat=0, wgLon=0):
    dLat = transformLat(wgLon - 105.0, wgLat - 35.0)
    dlon = transformLon(wgLon - 105.0, wgLat - 35.0)

    radLat = wgLat / 180 * pi
    magic = sin(radLat)
    magic = 1 - offset * magic * magic
    sqrtMagic = sqrt(magic)

    dLat = (dLat * 180.0) / ((axis * (1 - offset)) / (magic * sqrtMagic) * pi)
    dLon = (dlon * 180.0) / (axis / sqrtMagic * cos(radLat) * pi)

    latLon = [dLat, dLon]

    return latLon

def transformLat(x=0, y=0):
    ret = -100.0 + 2.0 * x + 3.0 * y + 0.2 * y * y + 0.1 * x * y + 0.2 * sqrt(abs(x))
    ret = ret + (20.0 * sin(6 * x * pi) + 20.0 * sin(2.0 * x * pi)) * 2.0 / 3.0
    ret = ret + (20.0 * sin(y * pi) + 40.0 * sin(y / 3.0 * pi)) * 2.0 / 3.0
    ret = ret + (160.0 * sin(y / 12.0 * pi) + 320.0 * sin(y / 30.0 * pi)) * 2.0 / 3.0

    return ret

def transformLon(x=0, y=0):
    ret = 300.0 + x + 2.0 * y + 0.1 * x * x + 0.1 * x * y + 0.1 * sqrt(abs(x))
    ret = ret + (20.0 * sin(6 * x * pi) + 20.0 * sin(2.0 * x * pi)) * 2.0 / 3.0
    ret = ret + (20.0 * sin(x * pi) + 40.0 * sin(x / 3.0 * pi)) * 2.0 / 3.0
    ret = ret + (150.0 * sin(x / 12.0 * pi) + 300.0 * sin(x / 30.0 * pi)) * 2.0 / 3.0

    return ret

def gcj2WGS(gcjLat=0, gcjLon=0):

    latlon = delta(gcjLat, gcjLon)
    latlon[0] = gcjLat - latlon[0]
    latlon[1] = gcjLon - latlon[1]

    return latlon

def value_to_color(value, dimension):

    if dimension == 1:
        colors = ['#9c1111', '#e69138', '#f2dd29', '#6ad739', '#306850', '#306850']
        cmap = LinearSegmentedColormap.from_list('mymap', colors)
        # value = value / 150
    if dimension == 0:
        colors = ['#306850', '#6ad739', '#f2dd29', '#e69138', '#9c1111']
        cmap = LinearSegmentedColormap.from_list('mymap', colors ,5)
        # value = value /5
      
    if dimension == 2:
        colors = ['#306850', '#6ad739', '#f2dd29', '#e69138', '#9c1111']
        cmap = LinearSegmentedColormap.from_list('mymap', colors)


    # 将0-1范围内的数值转换为RGB值
    rgb = cmap(value, bytes=True)[:3]
    # print(rgb)
    # 将RGB值转换为十六进制颜色表示
    hex_color = mcolors.rgb2hex((rgb[0] / 255, rgb[1] / 255, rgb[2] / 255))

    return hex_color

def PlotLineOnMap(data, roads, path, dimension):
    san_map = folium.Map(
        location=(roads[0][0][0][0], roads[0][0][0][1]),  # 打开地图直接定位到这个区域
        zoom_start=15,
        # 高德街道图
        control_scale=True,
        tiles='Stamen Toner',
        attr='北京市五环区域')
    san_map.add_child(folium.LatLngPopup())

    if dimension == 0:
        scale = 5.0
    elif dimension == 1:
        scale = 150.0
    else:
        scale = 3600.0

    normed_data = data[:, dimension]

    print(normed_data)
    for i, road_list in enumerate(roads):
        for section in road_list:
            folium.PolyLine(section, color=value_to_color(normed_data[i], dimension), weight=5).add_to(san_map)

    san_map.save(path)


x36 = np.load('/outputs/bjt2-samples/samples/0003.npy')

r36 = restore_matrix(x36)

road_data = json.load(open('/datasets/traffic/Roads1260.json'))
roads = []
for road in road_data:
    roads.append([])
    for section in road:
        roads[-1].append([])
        coordList = section['coordList']
        for i in range(0, len(coordList), 2):
            roads[-1][-1].append((coordList[i+1], coordList[i]))
# print(len(roads[0]))
dimension = 1

PlotLineOnMap(r36, roads, f'/outputs/Map_bjt2/ldm_28936_{dimension}.html', dimension)