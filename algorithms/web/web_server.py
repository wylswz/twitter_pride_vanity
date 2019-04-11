from flask import Flask, request, jsonify
from face_detection_wrapper.wrapper import FaceDetectionWrapper
import os
import traceback
import codecs

'''
export FLASK_APP=face_detection_wrapper/web_server.py
flask run
'''

app = Flask(__name__)
try:
    model_path = os.environ['DETECTION_MODEL_PATH']
except KeyError:
    traceback.print_exc()
    exit(1)

fdw = FaceDetectionWrapper(model_path)
fdw.load_model()

@app.route('/api/v1/face_detection', methods=['POST'])
def detection():
    if request.method == 'POST':
        if request.files.get('image_file') is not None:
            file = request.files['image_file']
            ml_res = fdw.predict(file)

        elif request.files.get('image_url') is not None:
            file = request.files['image_file']
            ml_res = fdw.predict(file)

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
            'version': 'Tensorflow Models Object-Detection @Iteration 4066'
        })