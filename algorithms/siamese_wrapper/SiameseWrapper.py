# Author: Yenyung Chang <yenyungc@student.unimelb.edu.au>

from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.applications import inception_v3
from PIL import Image
import io
import numpy as np
from shared.ModelWrapper import ModelWrapper
from tensorflow import Graph, Session


class FaceVerification(ModelWrapper):
    def __init__(self, model_path):
        super().__init__(model_path=model_path)

    def im_decoder(self, image):
        image = Image.open(io.BytesIO(image))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = image.resize((128, 128))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = inception_v3.preprocess_input(image)
        return image

    def load(self):
        self.graph1 = Graph()
        with self.graph1.as_default():
            self.session1 = Session()
            with self.session1.as_default():
                self.model = load_model(self.model_path)

    def preprocess(self, input_images):
        image1, image2 = input_images
        image1 = image1.read()
        image2 = image2.read()

        return [self.im_decoder(image1), self.im_decoder(image2)]

    def predict(self, input_images):
        with self.graph1.as_default():
            with self.session1.as_default():
                result = self.model.predict(self.preprocess(input_images))
        return result


if __name__ == "__main__":
    fv = FaceVerification("/mnt/models/face_comparison/model.h5")
    fv.load()
    image1 = open("trump1.jpg", "rb")
    image2 = open("trump2.jpg", "rb")
    res = fv.predict([image1, image2])
    print(res)
