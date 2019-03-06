class ImageTransformer:
    """
    Transforms image into correct format and normalizes it for cnn model
    """

    def __init__(self, width, height, num_channels):
        self.width = width
        self.height = height
        self.num_channels = num_channels

    def transform(self, image):
        image = image.astype("float32")
        image = image / 255
        image = image.reshape((self.height, self.width, self.num_channels))
        return image
