# Author: Yunlu Wen <yunluw@student.unimelb.edu.au>


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
