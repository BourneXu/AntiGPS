# TODO: main modul
import requests
from dynaconf import settings


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


if __name__ == "__main__":
    test_antigps = AntiGPS()

    ## Get test credentials
    from script.deserialize import Deserialize

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
