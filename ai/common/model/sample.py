from collections import namedtuple

Sample = namedtuple("Sample", ("input", "class_name", "source", "vector"))
TextSample = namedtuple("TextSample", ("text", "class_name", "source", "vector"))
ImageSample = namedtuple("ImageSample", ("image", "class_name", "source", "vector"))