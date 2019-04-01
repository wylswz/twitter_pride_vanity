import tensorflow as tf
from google.protobuf import text_format
def saved_model_reader(path, type="text"):

    saved_model_path = path
    graph_def = tf.GraphDef()
    with tf.gfile.FastGFile(saved_model_path, 'rb') as f:
        if type=="text":
            data = f.read()
            text_format.Merge(data, graph_def)
        else:
            data = f.read()
            graph_def = graph_def.ParseFromString(data)

    graph = tf.graph_util.import_graph_def(graph_def)
    return graph