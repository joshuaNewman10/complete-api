import os
from uuid import uuid4

from PIL import Image


class ImageFileSink:
    def __init__(self, directory):
        self._directory = directory

    def sink(self, image, identifier=None):
        if not identifier:
            identifier = str(uuid4())

        image = Image.fromarray(image)
        image_file_path = os.path.join(self._directory, identifier + ".jpg")
        image.save(image_file_path)
        return image_file_path
