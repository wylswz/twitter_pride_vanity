
"""
Maintainer: Yunlu Wen <yunluw@student.unimelb.edu.au>
This defines an abstract model wrapper schema for other machine learning guys
to wrap their models

Each wrapper should be able to:

- Be instantiated with the path to the model
- Load the model into memory using load() method
- Accept a piece of data when prediction() is invoked
- Pre-process the data
- Do predictions
- Post-process data
- Return the prediction result at the end of prediction() method

"""

class ModelWrapper:


    def __init__(self, model_path):
        self.model_path = model_path
        pass

    def load(self):
        """
        Load model using self.model_path
        :return:
        """
        raise NotImplementedError

    def preprocess(self, *args, **kwargs):
        
        """
        Pre-process the input to the model
        Executed at the beginning of predict()
        :param args:
        :param kwargs:
        :return:
        """
        raise NotImplementedError

    def postprocess(self, *args, **kwargs):
        """
        Post-process the output of the model
        Executed before the return clause in predict()
        :param args:
        :param kwargs:
        :return:
        """
        raise NotImplementedError


    def predict(self, *args, **kwargs):
        """
        Predict the result giving data

        self.preprocess()

        result = ... Do sth here

        result = self.postprocess()

        return result
        """
        raise NotImplementedError
