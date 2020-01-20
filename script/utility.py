# -*- coding:utf-8 -*-
import io
import os
import scipy
import random
import geocoder
import numpy as np
import pandas as pd
from PIL import Image
from scipy import signal
import plotly.express as px
from dynaconf import settings
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
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
    def image_save(image: bytes, filename):
        image = Image.open(io.BytesIO(image))
        image.save(filename, format="JPEG")

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
            # gcode = geocoder.mapbox([lat, lng], method="reverse", key=settings.MAPTOKEN)
            # coordinate_list.append(gcode)
            coordinate_list.append((lat, lng))
        return coordinate_list

    @staticmethod
    def visualize_map(self, coords: dict):
        mapbox_access_token = settings.MAPTOKEN
        px.set_mapbox_access_token(mapbox_access_token)
        fig = px.scatter_mapbox(
            pd.DataFrame(coords),
            lat="lats",
            lon="lngs",
            color_continuous_scale=px.colors.cyclical.IceFire,
            zoom=10,
        )
        fig.show()

    @staticmethod
    def concat_images_h_resize(im1, im2, resample=Image.BICUBIC, resize_big_image=True):
        if im1.height == im2.height:
            _im1 = im1
            _im2 = im2
        elif ((im1.height > im2.height) and resize_big_image) or (
            (im1.height < im2.height) and not resize_big_image
        ):
            _im1 = im1.resize(
                (int(im1.width * im2.height / im1.height), im2.height), resample=resample
            )
            _im2 = im2
        else:
            _im1 = im1
            _im2 = im2.resize(
                (int(im2.width * im1.height / im2.height), im1.height), resample=resample
            )
        dst = Image.new("RGB", (_im1.width + _im2.width, _im1.height))
        dst.paste(_im1, (0, 0))
        dst.paste(_im2, (_im1.width, 0))
        return dst

