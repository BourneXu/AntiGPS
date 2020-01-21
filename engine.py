# -*- coding: utf-8 -*-
import io
import os
import sys
import json
import copy
import time
import pickle
import hashlib
import requests
import warnings
import pandas as pd
from PIL import Image
from tqdm import tqdm
from tenacity import *
from loguru import logger
from dynaconf import settings
from script.feature import Feature
from script.decider import Decider
from script.utility import Utility
from script.attacker import Attacker
from script.deserialize import Deserialize

if not sys.warnoptions:
    warnings.simplefilter("ignore")


class AntiGPS:
    def __init__(self):
        self.attacker = Attacker("./results/pano_text.json")
        self.feature = Feature()

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(120))
    def get_streetview(self, credentials):
        logger.debug(f"Getting Street View with heading {credentials['heading']}")
        credentials["apikey"] = settings.GOOGLEAPI
        url = "https://maps.googleapis.com/maps/api/streetview?size=1280x640&location={lat},{lng}&heading={heading}&pitch={pitch}&fov=120&source=outdoor&key={apikey}".format(
            **credentials
        )
        image = requests.get(url)
        if image.status_code != 200:
            logger.warning(f"{image.json}")
            raise ValueError("No Google Street View Available")
        return image.content

    def get_pano_google(self, credentials):
        logger.debug(
            f"Getting Google Street View Pano of {credentials['lat']}, {credentials['lng']}"
        )
        filename = f"./results/google_img/{hashlib.md5((str(credentials['lat']) + '_' + str(credentials['lng']) + '_' + str(credentials['heading']) + '_' + str(credentials['pitch'])).encode()).hexdigest()}.jpg"
        ## Firstly check if local pano exists
        if os.path.exists(filename):
            logger.warning(f"Image {filename} is existing")
            image = Image.open(filename)
        else:
            ## requests Google API
            pano_120 = []
            with open("./results/google_img.csv", "a") as fin:
                for idx, degree in enumerate([-120, 0, 120]):
                    cred = copy.deepcopy(credentials)
                    cred["heading"] += degree
                    img = self.get_streetview(cred)
                    pano_120.append(img)
                image, img1, img2 = [Image.open(io.BytesIO(pano)) for pano in pano_120]
                image = Utility.concat_images_h_resize(image, img1)
                image = Utility.concat_images_h_resize(image, img2)
                Utility.image_save(image, filename, google=True)
                fin.write(
                    f"{filename},{credentials['lat']},{credentials['lng']},{credentials['heading']},{credentials['pitch']}"
                    + "\n"
                )
        return image, filename

    def deblur(self, pano):
        url = settings.URL_DEBLUR
        headers = {"Content-Type": "application/octet-stream"}
        r = requests.post(url, data=pano.compressed_image, headers=headers)
        img = Image.open(io.BytesIO(r.content))
        filename = f"{settings.IMAGE_FOLDER}/{pano.id}.jpg"
        img.save(filename, format="JPEG")
        return filename

    # TODO: two service: text detection and text recognition
    def ocr(self, image_path):
        url = "http://localhost:8301/ocr"
        data = {"image_path": os.path.abspath(image_path)}
        headers = {"Content-Type": "application/json"}
        r = requests.post(url, json=data, headers=headers)
        return r.json()

    def load_database(self, databaseDir):
        self.database = Deserialize(databaseDir)

    def extract_text(self):
        ## Donelist
        donelist = set()
        with open("./results/pano_text.json", "r") as fin:
            for line in fin.readlines():
                data = json.loads(line)
                donelist.add(data["id"])

        ## write into json file
        with open("./results/pano_text.json", "a") as fout:
            for pid, pano in tqdm(self.database.pano.items()):
                if pid.decode("utf8") in donelist:
                    logger.warning(f"{pid} already processed")
                    continue
                # image_path = self.deblur(pano)
                ## No debluring
                try:
                    img = Image.open(io.BytesIO(pano.compressed_image))
                except:
                    logger.warning(f"pano {pid} cannot identify image")
                    continue
                image_path = f"./results/images/{pid}.jpg"
                img.save(image_path, format="JPEG")
                info_text = self.ocr(image_path)
                info_all = {
                    "id": pano.id,
                    "lat": pano.coords.lat,
                    "lng": pano.coords.lng,
                    "heading": pano.heading_deg,
                    "pitch": pano.pitch_deg,
                    "neighbor": [x.id for x in pano.neighbor],
                    "pano_date": pano.pano_date,
                    "text_ocr": info_text,
                }
                fout.write(json.dumps(info_all) + "\n")
                os.remove(image_path)

    def defense(self, pano_attack: dict):
        logger.debug(f"Starting defense {pano_attack['id']}")
        if "lat_attack" in pano_attack:
            lat, lng = pano_attack["lat_attack"], pano_attack["lng_attack"]
        else:
            lat, lng = pano_attack["lat"], pano_attack["lng"]

        pano_google, img_path = self.get_pano_google(pano_attack)
        text_defense_list = []
        for i_path in img_path:
            info_ocr = self.ocr(i_path)
            text_defense_list.extend(info_ocr["text"])

        text_attack = " ".join(
            [x["predicted_labels"] for x in pano_attack["text_ocr"] if x["confidence_score"] > 0.95]
        )
        text_defense = " ".join(
            [x["predicted_labels"] for x in text_defense_list if x["confidence_score"] > 0.95]
        )
        decider = Decider()
        ratio = decider.similarity_text(text_attack, text_defense)
        result = {
            "similarity_text_ratio": [ratio],
            "id": [pano_attack["id"]],
            "lat": [pano_attack["lat"]],
            "lng": [pano_attack["lng"]],
            "lat_attack": [lat],
            "lng_attack": [lng],
            "text_attack": [text_attack],
            "text_defense": [text_defense],
        }
        resultDF = pd.DataFrame(result)
        resultDF.to_csv("./results/defense_result.csv", mode="a", index=False, header=False)
        logger.info(f"Defensed {pano_attack['id']}")

    # TODO: For real system, input should be two pano bytes or image objects
    def generate_feature_vector_local(self, pano_id, pano_id_attack, google=False):
        feature_vector = []
        feature_vector.extend(
            self.feature.textbox_position(self.attacker.dataset[pano_id], height=408, width=1632)
        )
        feature_vector.extend(self.feature.sentence_vector(self.attacker.dataset[pano_id]))
        ## google means get attack pano from google API. Otherwise get pano from local database
        if not google:
            feature_vector.extend(
                self.feature.textbox_position(
                    self.attacker.dataset[pano_id_attack], height=408, width=1632
                )
            )
            feature_vector.extend(
                self.feature.sentence_vector(self.attacker.dataset[pano_id_attack])
            )
        else:
            image, image_path = self.get_pano_google(self.attacker.dataset[pano_id_attack])
            ocr_results = self.ocr(image_path)
            feature_vector.extend(self.feature.textbox_position(ocr_results))
            feature_vector.extend(self.feature.sentence_vector(ocr_results))
        return feature_vector

    def generate_train_data(self, attack=True, noattack=True):
        filename = "/home/bourne/Workstation/AntiGPS/results/routes_generate.csv"
        routesDF = pd.read_csv(filename, index_col=["route_id"])
        routes_num = int(routesDF.index[-1] + 1)
        routes_slot = round(routes_num / 3)
        ## LSTM input data N (routes) x M (time steps) x W (features)
        if attack:
            logger.info("Generating training data for attacking")
            data_attack = []
            for route_id in tqdm(range(routes_slot)):
                data_route = []
                routeDF = routesDF.loc[route_id]
                for step in range(len(routeDF)):
                    pano_id, pano_id_attack = (
                        routeDF.iloc[step]["pano_id"],
                        routesDF.iloc[step + routes_slot]["pano_id"],
                    )
                    data_route.append(self.generate_feature_vector_local(pano_id, pano_id_attack))
                data_attack.append(data_route)

            with open(
                "/home/bourne/Workstation/AntiGPS/results/data_attack_lstm.pkl", "wb"
            ) as fout:
                pickle.dump(data_attack, fout, protocol=pickle.HIGHEST_PROTOCOL)
        if noattack:
            logger.info("Generating training data for non-attacking")
            data_noattack = []
            for route_id in tqdm(range(2 * routes_slot, 3 * routes_slot)):
                data_route = []
                routeDF = routesDF.loc[route_id]
                for step in range(len(routeDF)):
                    pano_id, pano_id_attack = (
                        routeDF.iloc[step]["pano_id"],
                        routeDF.iloc[step]["pano_id"],
                    )
                    data_route.append(
                        self.generate_feature_vector_local(pano_id, pano_id_attack, google=True)
                    )
                data_noattack.append(data_route)

            with open(
                "/home/bourne/Workstation/AntiGPS/results/data_noattack_lstm.pkl", "wb"
            ) as fout:
                pickle.dump(data_noattack, fout, protocol=pickle.HIGHEST_PROTOCOL)
        # return data


def test_get_streetview():
    test_antigps = AntiGPS()
    databaseDir = settings.LEVELDB.dir
    test_de = Deserialize(databaseDir)
    pano = test_de.pano[b"zhQIpFP7b4i56aavzTW9UA"]
    credential_test = {
        "lat": pano.coords.lat,
        "lng": pano.coords.lng,
        "heading": pano.heading_deg,
        "pitch": pano.pitch_deg,
    }
    image = test_antigps.get_streetview(credential_test)
    Utility.image_display(image)


def test_attack_defense():
    antigps = AntiGPS()
    attack = Attacker("./results/pano_text.json")
    # pano_attack_list = attack.random(10)
    pano_attack_list = attack.random_same(10)
    for pano_attack in tqdm(pano_attack_list):
        antigps.defense(pano_attack)


def test_generate_train_data():
    antigps = AntiGPS()
    data = antigps.generate_train_data(attack=False)
    # print(len(data), len(data[0]), len(data[0][0]))


if __name__ == "__main__":
    # test_antigps = AntiGPS()
    # test_antigps.load_database(settings.LEVELDB.dir[1])
    # test_antigps.extract_text()

    # test_attack_defense()
    test_generate_train_data()

