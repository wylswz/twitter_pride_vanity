from flask import Flask, request, jsonify
from face_detection_wrapper.wrapper import FaceDetectionWrapper
from siamese_wrapper.SiameseWrapper import FaceVerification
import os
import traceback
import requests
import codecs
import uuid
import numpy as np

'''
export FLASK_APP=face_detection_wrapper/web_server.py
flask run
'''
TEMP_PATH='/var/flask_temp/'

app = Flask(__name__)
try:
    model_path = os.environ['DETECTION_MODEL_PATH']
    comparison_model_path = os.environ['COMPARISON_MODEL_PATH']
except KeyError:
    traceback.print_exc()
    exit(1)

fdw = FaceDetectionWrapper(model_path)
fvw = FaceVerification(comparison_model_path)
fdw.load_model()
fvw.load()


if not os.path.isdir(TEMP_PATH):
    os.makedirs(TEMP_PATH)


@app.route('/api/v1/face_comparison', methods=['POST'])
def comparison():
    try:
        if request.files.get('face_1') is not None:
            face_1 = request.files['face_1']


        if request.files.get('face_2') is not None:
            face_2 = request.files['face_2']

        res = fvw.predict([face_1, face_2])
        print(res)
        return jsonify({
            "similarity": float(res[0][0]),
            "Version":"Keras Application 1.0.7 @ Tensorflow 1.13 Backend",
            "Model": {
                "Info": "Siamese Network with Inception ResNet @ Iteration 4313 Epoch 4",
                "url": "https://github.com/wylswz/twitter_pride_vanity/tree/master/algorithms/siamese_net",
                "author": "Yenyung Chang",
                "dataset": "vggface2",
            }
        })
    except Exception as e:
        return jsonify({
            'error': str(e),

        })



@app.route('/api/v1/face_detection', methods=['POST'])
def detection():
    if request.method == 'POST':
        try:
            ml_res = {
                'detection_boxes': np.array([]),
                'detection_scores': np.array([])
            }
            if request.files.get('image_file') is not None:
                file = request.files['image_file']
                ml_res = fdw.predict(file)

            elif request.form.get('image_url') is not None:
                file = request.form['image_url']
                resp = requests.get(file)
                affix = file.split('.')[-1]
                temp_filename = str(uuid.uuid4()) + '.' + affix
                with open(temp_filename, 'wb') as fp:
                    fp.write(resp.content)
                ml_res = fdw.predict(open(temp_filename, 'rb'))
                os.remove(temp_filename) 

            limit = request.form.get('limit')
            faces = ml_res['detection_boxes'].tolist()
            scores = ml_res['detection_scores'].tolist()
            if limit is not None:
                limit = int(limit)
                faces = faces[:limit]
                scores = scores[:limit]


            return jsonify({
                'faces': faces,

                'scores': scores,
                'format': ['ymin', 'xmin', 'ymax', 'xmax'],
                'version': 'Tensorflow 1.13.1',
                'model': {
                    "url":"https://github.com/tensorflow/models/tree/master/research/object_detection",
                    "info":"Faster RCNN Object-Detection @Iteration 4066",
                    "dataset": "WIDERFace",
                    "author": "Refer to url"
                }
            })
        except Exception as e:
            return jsonify({
                'error':str(e)
            })

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)
