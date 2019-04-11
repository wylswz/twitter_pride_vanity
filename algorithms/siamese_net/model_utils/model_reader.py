import tensorflow as tf
from google.protobuf import text_format


def saved_model_reader(path, type="text", namespace=''):
    """
    Load a pb file and return a list of tensors, which defines the structure of the model
    :param path:
    :param type:
    :return:
    """

    saved_model_path = path
    graph_def = tf.GraphDef()
    with tf.gfile.FastGFile(saved_model_path, 'rb') as f:
        if type == "text":
            data = f.read()
            text_format.Merge(data, graph_def)
        else:
            data = f.read()
            graph_def = graph_def.ParseFromString(data)
    tensor_list = [n.name for n in graph_def.node]

    tensor_list = list(map(lambda x : x + ':0', tensor_list))
    res = tf.graph_util.import_graph_def(graph_def, return_elements=tensor_list, name=namespace)
    return res


def get_to_restore_graph(tensor_list, exclude):
    """
    Build a diction for restoring graph
    :param tensor_list: A list of tensors imported from pb file
    :param exclude: Name of exclude tensors.
    :return: dict <str:Tensor>
    """
    res = {}
    for n in tensor_list:
        n: tf.Tensor
        tensor_op_name = n.op.node_def.op
        tensor_index = n.value_index
        tensor_dtype = n.dtype
        print(tensor_index)
        if n.name not in exclude and tensor_op_name == "Variable":
            res[n.name[:-2]] = tf.Tensor(n.op, tensor_index, tensor_dtype)
    print(res)
    return res

