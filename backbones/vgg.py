from tensorflow.keras.applications import VGG16, VGG19
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess
from tensorflow.keras.applications.vgg19 import preprocess_input as vgg19_preprocess

from .base import ImagenetBackbone


class VGG16Backbone(ImagenetBackbone):
    key = "vgg16"
    application = VGG16
    preprocess_fn = vgg16_preprocess
    input_size = (224, 224)


class VGG19Backbone(ImagenetBackbone):
    key = "vgg19"
    application = VGG19
    preprocess_fn = vgg19_preprocess
    input_size = (224, 224)
