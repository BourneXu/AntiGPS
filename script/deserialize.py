# -*- coding: utf-8 -*-
import io
import sys

import pandas as pd
import plyvel
import plotly.express as px
import matplotlib.image as mpimg
from PIL import Image
from tqdm import tqdm
from dynaconf import settings
from matplotlib import pyplot as plt

from proto import streetlearn_pb2

sys.path.append("..")



class Deserialize:
    def __init__(self, databaseDir):
        self.databaseDir = databaseDir
        self.__connect()
        self.deserialize()
        self.db.close()

    def __connect(self):
        self.db = plyvel.DB(self.databaseDir)

    def deserialize(self):
        self.pano = {}
        self.coords = {"lats": [], "lngs": []}
        for k, v in tqdm(self.db):
            pano = streetlearn_pb2.Pano()
            pano.ParseFromString(v)
            self.pano[k] = pano
            self.coords["lats"].append(pano.coords.lat)
            self.coords["lngs"].append(pano.coords.lng)

    def visualize_map(self):
        mapbox_access_token = settings.MAPTOKEN
        px.set_mapbox_access_token(mapbox_access_token)
        fig = px.scatter_mapbox(
            pd.DataFrame(self.coords),
            lat="lats",
            lon="lngs",
            color_continuous_scale=px.colors.cyclical.IceFire,
            zoom=10,
        )
        fig.show()

    @staticmethod
    def visualize_pano(pano: streetlearn_pb2.Pano):
        ## PIL raise memory error if image is large
        # image = Image.open(io.BytesIO(pano.compressed_image))
        # image.show()

        img = mpimg.imread(io.BytesIO(pano.compressed_image), format="JPG")
        plt.imshow(img)
        plt.show()


if __name__ == "__main__":
    databaseDir = settings.LEVELDB.dir[1]
    test = Deserialize(databaseDir)
    test.visualize_map()
    test.visualize_pano(test.pano[b"zhQIpFP7b4i56aavzTW9UA"])
    # test.visualize_pano(test.pano[list(test.pano.keys())[0]])
