# -*- coding: utf-8 -*-
import sys

sys.path.append("..")

import io
import plyvel
import pandas as pd
from PIL import Image
import streetlearn_pb2
import plotly.express as px
from dynaconf import settings


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
        for k, v in self.db:
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
        image = Image.open(io.BytesIO(pano.compressed_image))
        image.show()


if __name__ == "__main__":
    databaseDir = settings.LEVELDB.dir
    test = Deserialize(databaseDir)
    test.visualize_map()
    test.visualize_pano(test.pano[b"zhQIpFP7b4i56aavzTW9UA"])
