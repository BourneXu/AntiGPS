import io
from PIL import Image


class Utility:
    @staticmethod
    def image_display(image: bytes):
        image = Image.open(io.BytesIO(image))
        image.show()
