from shared.ModelWrapper import ModelWrapper
import tensorflow as tf
import PIL.Image as IMG
import numpy as np
from object_detection.utils import ops as utils_ops
import time

class FaceDetectionWrapper(ModelWrapper):

    def __init__(self, model_path):
        super().__init__(model_path=model_path)
        graph = tf.Graph()
        sess = tf.Session(graph=graph)
        self.graph = graph
        self.sess = sess

    def _load_binary_image_into_numpy_array(self, image: bytes):

        temp_img: IMG = IMG.open(image)
        (im_width, im_height) = temp_img.size
        mode = temp_img.mode
        return temp_img

    def load_model(self):
        with self.graph.as_default():
            with self.sess.as_default():
                ckpt = tf.train.latest_checkpoint(self.model_path)
                saver = tf.train.import_meta_graph(str(ckpt) + '.meta')
                saver.restore(self.sess, ckpt)
                ops = self.graph.get_operations()
                all_tensor_names = {output.name for op in ops for output in op.outputs}
                tensor_dict = {}
                for key in [
                    'num_detections', 'detection_boxes', 'detection_scores',
                    'detection_classes', 'detection_masks']:

                    tensor_name = key + ':0'
                    if tensor_name in all_tensor_names:
                        tensor_dict[key] = self.graph.get_tensor_by_name(tensor_name)
                self.tensor_dict = tensor_dict
                self.image_tensor = self.graph.get_tensor_by_name('image_tensor:0')

    def load(self):
        self.load_model()

    def preprocess(self, image, *args, **kwargs):
        rgb_img = image.convert("RGB")

        (im_width, im_height) = rgb_img.size
        return np.array(rgb_img).reshape(
            (im_height, im_width, 3)).astype(np.uint8)

    def postprocess(self, res_dict, *args, **kwargs):
        return res_dict

    def predict(self, image):
        with self.graph.as_default():
            with self.sess.as_default() as sess:
                start = time.time()
                sample_image = self._load_binary_image_into_numpy_array(image)
                sample = self.preprocess(sample_image)

                print("Pre process: ", time.time() - start)

                if 'detection_masks' in self.tensor_dict:
                    # The following processing is only for single image
                    detection_boxes = tf.squeeze(self.tensor_dict['detection_boxes'], [0])
                    detection_masks = tf.squeeze(self.tensor_dict['detection_masks'], [0])
                    real_num_detection = tf.cast(self.tensor_dict['num_detections'][0], tf.int32)
                    detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                    detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                        detection_masks, detection_boxes, sample.shape[0], sample.shape[1])
                    detection_masks_reframed = tf.cast(
                        tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                    # Follow the convention by adding back the batch dimension
                    self.tensor_dict['detection_masks'] = tf.expand_dims(
                        detection_masks_reframed, 0)
                print("Reframe", time.time() - start)
                # Run inference
                output_dict = sess.run(self.tensor_dict,
                                            feed_dict={self.image_tensor: np.expand_dims(sample, 0)})
                print("Prediction", time.time() - start)
                # all outputs are float32 numpy arrays, so convert types as appropriate
                output_dict['num_detections'] = int(output_dict['num_detections'][0])
                output_dict['detection_classes'] = output_dict[
                    'detection_classes'][0].astype(np.uint8)
                output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
                output_dict['detection_scores'] = output_dict['detection_scores'][0]
                if 'detection_masks' in output_dict:
                    output_dict['detection_masks'] = output_dict['detection_masks'][0]
                print(time.time() - start)
                # box: [ymin, xmin, ymax, xmax]
                return self.postprocess(output_dict)


class SSDWrapper(ModelWrapper):

    def __init__(self, model_path):
        super().__init__(model_path=model_path)
        graph = tf.Graph()
        sess = tf.Session(graph=graph)
        self.graph = graph
        self.sess = sess

    def _load_binary_image_into_numpy_array(self, image: bytes):

        temp_img: IMG = IMG.open(image)
        (im_width, im_height) = temp_img.size
        mode = temp_img.mode
        return temp_img

    def load_model(self):
        with self.graph.as_default():
            with self.sess.as_default():
                ckpt = tf.train.latest_checkpoint(self.model_path)

                saver = tf.train.import_meta_graph(str(ckpt) + '.meta')
                saver.restore(self.sess, ckpt)
                ops = self.graph.get_operations()
                all_tensor_names = {output.name for op in ops for output in op.outputs}

                print(all_tensor_names)
                tensor_dict = {}

                for key in [
                    'num_detections', 'detection_boxes', 'detection_scores',
                    'detection_classes', 'detection_masks']:

                    tensor_name = key + ':0'
                    if tensor_name in all_tensor_names:
                        tensor_dict[key] = self.graph.get_tensor_by_name(tensor_name)
                self.tensor_dict = tensor_dict
                self.image_tensor = self.graph.get_tensor_by_name('image_tensor:0')

    def load(self):
        self.load_model()

    def preprocess(self, image, *args, **kwargs):
        rgb_img = image.convert("RGB")

        (im_width, im_height) = rgb_img.size
        return np.array(rgb_img).reshape(
            (im_height, im_width, 3)).astype(np.uint8)

    def postprocess(self, res_dict, *args, **kwargs):
        return res_dict

    def predict(self, image):
        with self.graph.as_default():
            with self.sess.as_default() as sess:
                start = time.time()
                sample_image = self._load_binary_image_into_numpy_array(image)
                sample = self.preprocess(sample_image)

                print("Pre process: ", time.time() - start)

                if 'detection_masks' in self.tensor_dict:
                    # The following processing is only for single image
                    detection_boxes = tf.squeeze(self.tensor_dict['detection_boxes'], [0])
                    detection_masks = tf.squeeze(self.tensor_dict['detection_masks'], [0])
                    real_num_detection = tf.cast(self.tensor_dict['num_detections'][0], tf.int32)
                    detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                    detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                        detection_masks, detection_boxes, sample.shape[0], sample.shape[1])
                    detection_masks_reframed = tf.cast(
                        tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                    # Follow the convention by adding back the batch dimension
                    self.tensor_dict['detection_masks'] = tf.expand_dims(
                        detection_masks_reframed, 0)
                print("Reframe", time.time() - start)
                # Run inference
                print(sess)
                output_dict = sess.run(self.tensor_dict,
                                            feed_dict={self.image_tensor: np.expand_dims(sample, 0)})
                print("Prediction", time.time() - start)
                # all outputs are float32 numpy arrays, so convert types as appropriate
                output_dict['num_detections'] = int(output_dict['num_detections'][0])
                output_dict['detection_classes'] = output_dict[
                    'detection_classes'][0].astype(np.uint8)
                output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
                output_dict['detection_scores'] = output_dict['detection_scores'][0]
                if 'detection_masks' in output_dict:
                    output_dict['detection_masks'] = output_dict['detection_masks'][0]
                print(time.time() - start)
                # box: [ymin, xmin, ymax, xmax]
                return self.postprocess(output_dict)

if __name__ == "__main__":
    MODEL_PATH = "/home/johnny/RCNN/temp_final"

    wrapper = FaceDetectionWrapper(MODEL_PATH)
    wrapper.load_model()
    pred_dict = wrapper.predict()
