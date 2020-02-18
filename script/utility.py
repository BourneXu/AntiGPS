# -*- coding:utf-8 -*-
import io
import os
import random

import pandas as pd
import geocoder
import plotly.express as px
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PIL import Image
from geopy import distance
from loguru import logger
from dynaconf import settings
from statsmodels.distributions.empirical_distribution import ECDF

NORTHERNMOST = 49.0
SOUTHERNMOST = 25.0
EASTERNMOST = -66.0
WESTERNMOST = -124.0


class Utility:
    @staticmethod
    def plot(x, y, xlabel, ylabel, title, filename=None):
        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
        if filename:
            fig.savefig(filename)
        plt.show()

    @staticmethod
    def image_display(image: bytes):
        image = Image.open(io.BytesIO(image))
        image.show()

    @staticmethod
    def image_save(image, filename, google=False):
        if google:
            ## Cut image to remove google watermark
            width, height = image.size
            image = image.crop((0, 0, width, height - 40))
        image.save(filename, format="JPEG")

    @staticmethod
    def image_save_byte(image: bytes, filename, google=False):
        image = Image.open(io.BytesIO(image))
        if google:
            ## Cut image to remove google watermark
            width, height = image.size
            image = image.crop((0, 0, width, height - 40))
        image.save(filename, format="JPEG")

    @staticmethod
    def plot_cdf(xarray):
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
    def visualize_map(coords: dict):
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

    @staticmethod
    def distance_route(route):
        dist = 0
        for start, end in zip(route[:-1], route[1:]):
            coor_start, coor_end = (start["lat"], start["lng"]), (end["lat"], end["lng"])
            dist += distance.distance(coor_start, coor_end).m
        return dist


def test_concat_images():
    folder = "../results/google_img/"
    image_files = os.listdir(folder)
    img_names = set([x.split("_")[0] for x in image_files])
    for img_name in img_names:
        pano_120 = []
        for idx in range(3):
            current_img = folder + img_name + f"_{idx}.jpg"
            pano_120.append(Image.open(current_img))
        image, img1, img2 = pano_120
        image = Utility.concat_images_h_resize(image, img1)
        image = Utility.concat_images_h_resize(image, img2)
        filename = folder + img_name + ".jpg"
        if not os.path.exists(filename):
            Utility.image_save(image, filename, google=True)
        else:
            logger.warning(f"Image {filename} is existing")


if __name__ == "__main__":
    test_concat_images()
