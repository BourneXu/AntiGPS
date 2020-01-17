# -*- coding: utf-8 -*-
import io
import os
import json
import copy
import requests
import pandas as pd
from PIL import Image
from tqdm import tqdm
from loguru import logger
from dynaconf import settings
from script.decider import Decider
from script.utility import Utility
from script.attacker import Attacker
from script.deserialize import Deserialize


class AntiGPS:
    def __init__(self):
        pass

    def get_streetview(self, credentials):
        logger.debug(f"Getting Street View with heading {credentials['heading']}")
        credentials["apikey"] = settings.GOOGLEAPI
        url = "https://maps.googleapis.com/maps/api/streetview?size=1280x640&location={lat},{lng}&heading={heading}&pitch={pitch}&fov=120&key={apikey}".format(
            **credentials
        )
        image = requests.get(url)
        if image.status_code != 200:
            raise ValueError("No Google Street View Available")
        return image.content

    def get_pano_google(self, credentials):
        logger.debug(
            f"Getting Google Street View Pano of {credentials['lat']}, {credentials['lng']}"
        )
        # TODO: merge 3 120° pano into a 360° pano
        pano_120 = []
        img_path = []
        with open("./results/google_img.csv", "a") as fin:
            for idx, degree in enumerate([-120, 0, 120]):
                cred = copy.deepcopy(credentials)
                cred["heading"] += degree
                img = self.get_streetview(cred)
                filename = f"./results/google_img/{hash(str(credentials['lat']) + '_' + str(credentials['lng']))}_{idx}.jpg"
                Utility.image_save(img, filename)
                pano_120.append(img)
                img_path.append(filename)
                fin.write(f"{filename},{credentials['lat']},{credentials['lng']}" + "\n")
        return pano_120, img_path

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
            info_text = self.ocr(i_path)
            text_defense_list.extend(info_text)

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


if __name__ == "__main__":
    # test_antigps = AntiGPS()
    # test_antigps.load_database(settings.LEVELDB.dir[1])
    # test_antigps.extract_text()

    test_attack_defense()

