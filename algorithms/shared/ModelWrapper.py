class ModelWrapper:

    model_path = None
    sample = None
    result = None
    ckpt = None

    def __init__(self, model_path):
        self.model_path = model_path
        pass

    def load(self):
        """
        Load model
        :return:
        """
        raise NotImplementedError

    def preprocess(self, *args, **kwargs):
        
        """
        Pre-process the input to the model
        :param args:
        :param kwargs:
        :return:
        """
        raise NotImplementedError

    def postprocess(self, *args, **kwargs):
        """
        Post-process the output of the model
        :param args:
        :param kwargs:
        :return:
        """
        raise NotImplementedError

    def get_result(self, *args, **kwargs):
        raise NotImplementedError

    def load_sample(self, *args, **kwargs):
        raise NotImplementedError

    def predict(self, *args, **kwargs):
        raise NotImplementedError
