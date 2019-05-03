from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.applications import inception_v3
from PIL import Image
import io
import numpy as np
from shared import ModelWrapper

class face_verification(ModelWrapper):
    def __init__(self, model_path):
        self.model_path = model_path
        pass
       
    def im_decoder(self,image):
        image = Image.open(io.BytesIO(image))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = image.resize((128,128))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = inception_v3.preprocess_input(image)
        return image
    
    def load(self):
        
        self.model = load_model(self.model_path)
        
        
    def preprocess(self, input_images):
        
        image1, image2 = input_images
        
        return [self.im_decoder(image1), self.im_decoder(image1)]
                    
        
    def predict(self, input_images):    
        model = self.model
        result = model.predict(self.preprocess(input_images))
        return result
        



