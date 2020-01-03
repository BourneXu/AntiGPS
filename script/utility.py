# -*- coding:utf-8 -*-
import io
import os
import scipy
import random
import geocoder
import numpy as np
from PIL import Image
from scipy import signal
from dynaconf import settings
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF

NORTHERNMOST = 49.0
SOUTHERNMOST = 25.0
EASTERNMOST = -66.0
WESTERNMOST = -124.0


class Utility:
    @staticmethod
    def image_display(image: bytes):
        image = Image.open(io.BytesIO(image))
        image.show()

    @staticmethod
    def plot_cdf(xarray):
        # xarray_sorted = np.sort(xarray)
        # percent = 1.0 * np.arange(len(xarray)) / (len(xarray) - 1)
        # line = plt.plot(xarray_sorted, percent)

        ecdf = ECDF(xarray)
        line = plt.plot(ecdf.x, ecdf.y * 100)
        return line

    @staticmethod
    def generateGPS_random(number_of_points):
        coordinate_list = []
        for _ in range(number_of_points):
            lat = round(random.uniform(SOUTHERNMOST, NORTHERNMOST), 6)
            lng = round(random.uniform(EASTERNMOST, WESTERNMOST), 6)
            gcode = geocoder.mapbox([lat, lng], method="reverse", key=settings.MAPTOKEN)
            coordinate_list.append(gcode)
        return coordinate_list

