import tensorflow as tf
from google.protobuf import text_format
from siamese_net.model_utils import model_reader
dir(tf.contrib)

def build_net(config):
    graph = model_reader.saved_model_reader(config["saved_model_path"],"text")




if __name__ == "__main__":
    config = {
        "saved_model_path":"/home/johnny/Documents/TF_CONFIG/finetune/resv2/eval.graph"
    }

    build_net(config)