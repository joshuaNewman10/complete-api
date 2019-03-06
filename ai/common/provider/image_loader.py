import cv2
import numpy
import requests


class ImageDecodeError(Exception):
    pass


class ImageLoader:
    """
    Helper class to download, transform and encode image
    """
    def __init__(self, image_size):
        """
        :param image_size: tuple of (width, height)
        """
        self.image_size = tuple(image_size)

    def download(self, image_url):
        resp = requests.get(image_url, timeout=3)
        resp.raise_for_status()
        image_content = resp.content
        return image_content

    def decode(self, content, flags=cv2.IMREAD_COLOR):
        bitmap = numpy.asarray(bytearray(content), dtype=numpy.uint8)
        image = cv2.imdecode(bitmap, flags)

        if image is None:
            raise ImageDecodeError('Could not decode image.')

        return image

    def encode(self, image):
        """ Encodes an image from a numpy array, saves space"""
        img_str = cv2.imencode('.jpg', image)[1].tostring()
        return img_str

    def transform(self, image):
        image = self.resize(image, self.image_size).astype(numpy.float32)
        return image

    def resize(self, image, size):
        width, height = image.shape[1], image.shape[0]
        target_width, target_height = int(size[0]), int(size[1])
        if width != target_width or height != target_height:
            # NOTE: _resize is used at download time to cache the image in smaller size
            # TODO: Test in depth what are the differences between these two and how they affect
            # detections in # and QA., i.e.: use interpolation=cv2.INTER_CUBIC
            image = cv2.resize(image, tuple(size), interpolation=cv2.INTER_AREA)
        return image
