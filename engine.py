# -*- coding: utf-8 -*-
import io
import os
import sys
import copy
import json
import time
import pickle
import random
import hashlib
import warnings
import concurrent.futures

import numpy as np
import pandas as pd
import plyvel
import requests
from PIL import Image
from tqdm import tqdm
from loguru import logger
from dynaconf import settings
from tenacity import *

from script.decider import Decider
from script.feature import Feature
from script.utility import Utility
from script.attacker import Attacker
from script.deserialize import Deserialize

if not sys.warnoptions:
    warnings.simplefilter("ignore")


class AntiGPS:
    def __init__(self):
        self.attacker = Attacker("./results/pano_text.json")
        self.feature = Feature()
        self.decider = Decider()

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(120))
    def get_poi_azure(self, credentials: dict, radius=50) -> dict:
        logger.debug(f"Getting Azure POIs around {credentials['lat']}, {credentials['lng']}")
        credentials["radius"] = radius
        credentials["subscription_key"] = settings.AZUREAPI.subscription_key
        headers = {"x-ms-client-id": settings.AZUREAPI.client_id}

        url = "https://atlas.microsoft.com/search/nearby/json?subscription-key={subscription_key}&api-version=1.0&lat={lat}&lon={lng}&radius={radius}".format(
            **credentials
        )
        r = requests.get(url, headers=headers)
        if r.status_code != 200:
            logger.warning(f"{r.json}")
            raise ValueError("No POIs around available")
        return r.json()

    def get_poi(self, credentials: dict, radius=50) -> dict:
        key = (str(credentials["lat"]) + "_" + str(credentials["lng"])).encode()
        pois = self.db_poi.get(key)
        if pois == None:
            pois = self.get_poi_azure(credentials, radius)
            self.db_poi.put(key, json.dumps(pois).encode())
        else:
            pois = json.loads(pois)
        return pois

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
    def ocr(self, image_path: str) -> dict:
        url = "http://localhost:8301/ocr"
        data = {"image_path": os.path.abspath(image_path)}
        headers = {"Content-Type": "application/json"}
        r = requests.post(url, json=data, headers=headers)
        return r.json()

    def load_database(self, databaseDir):
        self.database = Deserialize(databaseDir)

    def extract_text(self):
        """This func would be only used for extracting streetlearn dataset
        """
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
        """Discarded
        """

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
        ratio = self.decider.similarity_text(text_attack, text_defense)
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

    def init_leveldb(self):
        """
        Initialize training data levelDB database
        """
        logger.info("Initializing levelDB ...")
        self.db_feature = plyvel.DB(
            "/home/bourne/Workstation/AntiGPS/results/features/", create_if_missing=True
        )
        self.db_attack = plyvel.DB(
            "/home/bourne/Workstation/AntiGPS/results/train_data_attack/", create_if_missing=True
        )
        self.db_noattack = plyvel.DB(
            "/home/bourne/Workstation/AntiGPS/results/train_data_noattack/", create_if_missing=True
        )
        self.db_poi = plyvel.DB(
            "/home/bourne/Workstation/AntiGPS/results/azure_poi/", create_if_missing=True
        )
        self.db_attack_poi = plyvel.DB(
            "/home/bourne/Workstation/AntiGPS/results/train_data_attack_poi/",
            create_if_missing=True,
        )
        self.db_noattack_poi = plyvel.DB(
            "/home/bourne/Workstation/AntiGPS/results/train_data_noattack_poi/",
            create_if_missing=True,
        )
        self.db_partial_attack = plyvel.DB(
            "/home/bourne/Workstation/AntiGPS/results/test_data_partial_attack/",
            create_if_missing=True,
        )
        self.db_partial_attack_google = plyvel.DB(
            "/home/bourne/Workstation/AntiGPS/results/test_data_partial_attack_google/",
            create_if_missing=True,
        )
        self.db_partial_attack_poi = plyvel.DB(
            "/home/bourne/Workstation/AntiGPS/results/test_data_partial_attack_poi/",
            create_if_missing=True,
        )

    def close_leveldb(self):
        logger.info("Closing levelDB ...")
        self.db_feature.close()
        self.db_attack.close()
        self.db_noattack.close()
        self.db_poi.close()
        self.db_partial_attack.close()
        self.db_partial_attack_google.close()
        self.db_partial_attack_poi.close()

    # TODO: Real system get only one pano from car cam with GPS info
    def generate_feature_vector(self, pano: bytes):
        pass

    # TODO: For real system, input should be two pano bytes or image objects
    def generate_feature_vector_local(self, pano_id, pano_id_attack, valid="default"):
        """Locally generate feature vectors with three validation methods: 
        1. local database (default); 
        2. Google Street View APIs (google); 
        3. Azure POIs API (poi)
        
        Arguments:
            pano_id {str} -- real pano id
            pano_id_attack {str} -- attack pano id
        
        Keyword Arguments:
            valid {str} -- validation methods for attack pano (default: {"default"}, "google", "poi")
        
        Returns:
            list -- feature vector
        """
        key = pano_id.encode("utf-8")
        if self.db_feature.get(key):
            feature_vector = pickle.loads(self.db_feature.get(key))
        else:
            feature_vector = []
            feature_vector.extend(
                self.feature.textbox_position(
                    self.attacker.dataset[pano_id], height=408, width=1632
                )
            )
            feature_vector.extend(self.feature.sentence_vector(self.attacker.dataset[pano_id]))
            self.db_feature.put(key, pickle.dumps(feature_vector))
        ## google means get attack pano from google API. Otherwise get attack pano from local database
        if valid == "default":
            key = pano_id_attack.encode("utf-8")
            if self.db_feature.get(key):
                feature_vector.extend(pickle.loads(self.db_feature.get(key)))
            else:
                feature_vector_attack = []
                feature_vector_attack.extend(
                    self.feature.textbox_position(
                        self.attacker.dataset[pano_id_attack], height=408, width=1632
                    )
                )
                feature_vector_attack.extend(
                    self.feature.sentence_vector(self.attacker.dataset[pano_id_attack])
                )
                feature_vector.extend(feature_vector_attack)
                self.db_feature.put(key, pickle.dumps(feature_vector_attack))
        elif valid == "google":
            ## requests Google Street View and do OCR
            key = (pano_id_attack + "_google").encode("utf-8")
            if self.db_feature.get(key):
                feature_vector.extend(pickle.loads(self.db_feature.get(key)))
            else:
                image, image_path = self.get_pano_google(self.attacker.dataset[pano_id_attack])
                ocr_results = self.ocr(image_path)
                feature_vector_attack = self.feature.textbox_position(ocr_results)
                feature_vector_attack.extend(self.feature.sentence_vector(ocr_results))
                feature_vector.extend(feature_vector_attack)
                self.db_feature.put(key, pickle.dumps(feature_vector_attack))
        elif valid == "poi":
            ## requests Azure POIs
            key = (pano_id_attack + "_poi").encode("utf-8")
            if self.db_feature.get(key):
                feature_vector.extend(pickle.loads(self.db_feature.get(key)))
            else:
                pois = self.get_poi(self.attacker.dataset[pano_id_attack])
                feature_vector_attack = self.feature.poi_vector(pois)
                feature_vector.extend(feature_vector_attack)
                self.db_feature.put(key, pickle.dumps(feature_vector_attack))
        else:
            raise ValueError(f"Invalid valid param: {valid}")
        return feature_vector

    def get_route_todo(self, routes_ids: list, db, thread=1) -> list:
        if thread < 2:
            return [routes_ids]
        routes_done = {k.decode() for k, v in db}
        route_todo = list(set([str(x) for x in routes_ids]) - routes_done)
        logger.debug(f"done {len(routes_done)}, todo {len(route_todo)}")
        return np.array_split(route_todo, thread)

    # TODO: Some issues with multiple threads, need to fix
    def generate_train_data(self, filename, attack=True, noattack=True, thread=5, overwrite=False):
        logger.info("Generating training data with Google Street Views")
        routesDF = pd.read_csv(filename, index_col=["route_id"])
        routes_num = int(routesDF.index[-1] + 1)
        routes_slot = round(routes_num / 3)
        ## LSTM input data N (routes) x M (time steps) x W (features)
        if attack:

            def generate_train_data_attack(routes_todo, thread_id):
                for route_id in tqdm(routes_todo, desc=f"Thread: {thread_id}"):
                    key = str(route_id).encode("utf-8")
                    if self.db_attack.get(key) and not overwrite:
                        continue
                    data_route = []
                    routeDF = routesDF.loc[route_id]
                    routeDF_attack = routesDF.loc[route_id + routes_slot]
                    for step in range(len(routeDF)):
                        pano_id, pano_id_attack = (
                            routeDF.iloc[step]["pano_id"],
                            routeDF_attack.iloc[step]["pano_id"],
                        )
                        data_route.append(
                            self.generate_feature_vector_local(pano_id, pano_id_attack)
                        )
                    self.db_attack.put(key, pickle.dumps(data_route))
                return f"done {len(routes_todo)}"

            logger.info("Generating training data for attacking")
            routes_todos = self.get_route_todo(list(range(routes_slot)), self.db_attack, thread)
            # TODO: Figure out locks in threading
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                r = [
                    executor.submit(generate_train_data_attack, routes_todos[i], i)
                    for i in range(thread)
                ]
                for re in concurrent.futures.as_completed(r):
                    logger.debug(re.result())
                results = [re.result() for re in r]
                logger.debug(f"All done with {results}")

        if noattack:

            def generate_train_data_noattack(routes_todo, thread_id):
                for route_id in tqdm(routes_todo, desc=f"Thread: {thread_id}"):
                    key = str(route_id).encode("utf-8")
                    if self.db_noattack.get(key) and not overwrite:
                        continue
                    data_route = []
                    routeDF = routesDF.loc[route_id]
                    for step in range(len(routeDF)):
                        pano_id, pano_id_attack = (
                            routeDF.iloc[step]["pano_id"],
                            routeDF.iloc[step]["pano_id"],
                        )
                        data_route.append(
                            self.generate_feature_vector_local(
                                pano_id, pano_id_attack, valid="google"
                            )
                        )
                    self.db_noattack.put(key, pickle.dumps(data_route))
                return f"done {len(routes_todo)}"

            logger.info("Generating training data for non-attacking")
            routes_todos = self.get_route_todo(
                list(range(2 * routes_slot, 3 * routes_slot)), self.db_noattack, thread
            )
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                r = [
                    executor.submit(generate_train_data_noattack, routes_todos[i], i)
                    for i in range(thread)
                ]
                for re in concurrent.futures.as_completed(r):
                    logger.debug(re.result())
                results = [re.result() for re in r]
                logger.debug(f"All done with {results}")

    def generate_train_data_poi(
        self, filename, attack=True, noattack=True, thread=5, overwrite=False
    ):
        logger.info("Generating training data with Azure POIs")
        routesDF = pd.read_csv(filename, index_col=["route_id"])
        routes_num = int(routesDF.index[-1] + 1)
        routes_slot = round(routes_num / 3)
        ## LSTM input data N (routes) x M (time steps) x W (features)
        if attack:

            def generate_train_data_attack(routes_todo, thread_id):
                for route_id in tqdm(routes_todo, desc=f"Thread: {thread_id}"):
                    key = str(route_id).encode("utf-8")
                    if self.db_attack_poi.get(key) and not overwrite:
                        continue
                    data_route = []
                    routeDF = routesDF.loc[route_id]
                    routeDF_attack = routesDF.loc[route_id + routes_slot]
                    for step in range(len(routeDF)):
                        pano_id, pano_id_attack = (
                            routeDF.iloc[step]["pano_id"],
                            routeDF_attack.iloc[step]["pano_id"],
                        )
                        data_route.append(
                            self.generate_feature_vector_local(pano_id, pano_id_attack, valid="poi")
                        )
                    self.db_attack_poi.put(key, pickle.dumps(data_route))
                return f"done {len(routes_todo)}"

            logger.info("Generating training data for attacking")
            routes_todos = self.get_route_todo(list(range(routes_slot)), self.db_attack_poi, thread)
            # TODO: Figure out locks in threading
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                r = [
                    executor.submit(generate_train_data_attack, routes_todos[i], i)
                    for i in range(thread)
                ]
                for re in concurrent.futures.as_completed(r):
                    logger.debug(re.result())
                results = [re.result() for re in r]
                logger.debug(f"All done with {results}")

        if noattack:

            def generate_train_data_noattack(routes_todo, thread_id):
                for route_id in tqdm(routes_todo, desc=f"Thread: {thread_id}"):
                    key = str(route_id).encode("utf-8")
                    if self.db_noattack_poi.get(key) and not overwrite:
                        continue
                    data_route = []
                    routeDF = routesDF.loc[route_id]
                    for step in range(len(routeDF)):
                        pano_id, pano_id_attack = (
                            routeDF.iloc[step]["pano_id"],
                            routeDF.iloc[step]["pano_id"],
                        )
                        data_route.append(
                            self.generate_feature_vector_local(pano_id, pano_id_attack, valid="poi")
                        )
                    self.db_noattack_poi.put(key, pickle.dumps(data_route))
                return f"done {len(routes_todo)}"

            logger.info("Generating training data for non-attacking")
            routes_todos = self.get_route_todo(
                list(range(2 * routes_slot, 3 * routes_slot)), self.db_noattack_poi, thread
            )
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                r = [
                    executor.submit(generate_train_data_noattack, routes_todos[i], i)
                    for i in range(thread)
                ]
                for re in concurrent.futures.as_completed(r):
                    logger.debug(re.result())
                results = [re.result() for re in r]
                logger.debug(f"All done with {results}")

    def generate_partial_attack(
        self,
        routes: list,
        keys: list,
        rate: float,
        method="random",
        valid="default",
        overwrite=False,
    ):
        """Only for research purposes. Generate partial attacked routes.
        
        Arguments:
            routes {str} -- routes with pano_ids
            keys {list} -- routes id list
            rate {float} -- partial attacked attack rate
        
        Keyword Arguments:
            method {str} -- ways to define 'partial' (default: {"random"})
        """
        logger.info(f"Generating partial attack routes for testing with rate {rate} ...")
        if valid == "default":
            db = self.db_partial_attack
        elif valid == "google":
            db = self.db_partial_attack_google
        elif valid == "poi":
            db = self.db_partial_attack_poi
        else:
            raise ValueError(f"Invalid valid value: {valid}")
        if method == "random":
            routes_slot = round(len(routes) / 3)
            route_len = len(routes[int(keys[0])])
            for key in tqdm(keys):
                key_db = (str(key) + "_" + str(rate)).encode("utf-8")
                if db.get(key_db) and not overwrite:
                    continue
                route_attack = routes[int(key) + routes_slot]
                attack_idx = random.sample(list(range(route_len)), round(route_len * rate))
                data_route = []
                for idx, pano in enumerate(routes[int(key)]):
                    if idx in attack_idx:
                        pano_id, pano_id_attack = pano["id"], route_attack[idx]["id"]
                        data_route.append(
                            self.generate_feature_vector_local(
                                pano_id, pano_id_attack, valid="default"
                            )
                        )
                    else:
                        pano_id, pano_id_attack = pano["id"], pano["id"]
                        data_route.append(
                            self.generate_feature_vector_local(pano_id, pano_id_attack, valid=valid)
                        )
                db.put(key_db, pickle.dumps(data_route))


def test_get_poi():
    test_antigps = AntiGPS()
    credential_test = {"lat": 40.7461106, "lng": -73.9941583}
    pois = test_antigps.get_poi_azure(credential_test)
    print(pois)


def test_get_poi_dataset():
    test_antigps = AntiGPS()
    db = plyvel.DB(settings.LEVELDB.poi, create_if_missing=True)
    filename = "/home/bourne/Workstation/AntiGPS/results/routes_generate_longer.csv"
    routes = test_antigps.attacker.read_route(filename)
    routes_num, routes_slot = len(routes), round(len(routes) / 3)
    for route in tqdm(routes[routes_slot:]):
        for pano in route:
            key = (str(pano["lat"]) + "_" + str(pano["lng"])).encode()
            if db.get(key) == None:
                pois = test_antigps.get_poi_azure(pano)
                db.put(key, json.dumps(pois).encode())
            else:
                logger.warning(f"{key} is existing")
    db.close()


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
    filename = "/home/bourne/Workstation/AntiGPS/results/routes_generate_longer_split.csv"
    antigps.init_leveldb()
    antigps.generate_train_data(filename, attack=True, thread=1, overwrite=True)
    antigps.generate_train_data_poi(filename, thread=1, overwrite=True)
    antigps.close_leveldb()


def test_generate_partial_attack():
    antigps = AntiGPS()
    filename = "/home/bourne/Workstation/AntiGPS/results/routes_generate_longer_split.csv"
    antigps.init_leveldb()
    with open("/home/bourne/Workstation/AntiGPS/results/keys_test.pkl", "rb") as fout:
        route_keys = pickle.load(fout)
    route_keys = [x.decode("utf-8") for x in route_keys]
    routes = antigps.attacker.read_route(filename, only_id=True)
    for rate in [round(x * 0.1, 2) for x in range(0, 11)]:
        antigps.generate_partial_attack(
            routes, route_keys, rate, method="random", valid="default", overwrite=False
        )
    antigps.close_leveldb()


if __name__ == "__main__":
    # test_antigps = AntiGPS()
    # test_antigps.load_database(settings.LEVELDB.dir[1])
    # test_antigps.extract_text()

    # test_attack_defense()
    # test_generate_train_data()
    # test_get_poi()
    # test_get_poi_dataset()

    ### Partial attack generation and test
    test_generate_partial_attack()

    from script.decider import test_partial_attack_predict

    poi = False
    modelpath = "/home/bourne/Workstation/AntiGPS/results/trained_models/lstm_{}.h5".format(
        poi * "poi"
    )
    rates = [round(x * 0.1, 2) for x in range(0, 11)]
    acc_all = test_partial_attack_predict(modelpath=modelpath, rates=rates, valid="default")
    Utility.plot(rates, acc_all)
