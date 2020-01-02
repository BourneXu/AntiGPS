import io
import os
import json
import requests
from PIL import Image
from tqdm import tqdm
from loguru import logger
from dynaconf import settings
from script.deserialize import Deserialize


class AntiGPS:
    def __init__(self):
        pass

    def get_streetview(self, credentials):
        credentials["apikey"] = settings.GOOGLEAPI
        url = "https://maps.googleapis.com/maps/api/streetview?size=1280x640&location={lat},{lng}&heading={heading}&pitch={pitch}&fov=120&key={apikey}".format(
            **credentials
        )
        image = requests.get(url)
        return image

    def deblur(self, pano):
        url = settings.URL_DEBLUR
        headers = {"Content-Type": "application/octet-stream"}
        r = requests.post(url, data=pano.compressed_image, headers=headers)
        img = Image.open(io.BytesIO(r.content))
        filename = f"{settings.IMAGE_FOLDER}/{pano.id}.jpg"
        img.save(filename, format="JPEG")
        return filename

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
                image_path = f"./results/images/{pano.id}.jpg"
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


def test_get_streetview():
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

    from script.utility import Utility

    Utility.image_display(image.content)


if __name__ == "__main__":
    test_antigps = AntiGPS()
    test_antigps.load_database(settings.LEVELDB.dir[1])
    test_antigps.extract_text()

