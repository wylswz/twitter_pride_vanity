# API Reference

## Face detection
### /api/v1/face_detection

### Method: POST

### Request body (form data)
- image_file: Send file objects
- image_url: Or you can give url of the image. If url and file are given at the 
same time, file will be used
- limit: Limit on number of bounding boxes.
- model: The model used for face detection. Available models
are
    ```
    'SSD_MOBILENET_V2'
    'FASTER_RCNN_RESNET_101' # default
    ```
FASTER_RCNN_RESNET_101 has high precision and recall

SSD_MOBILENET_V2 is very fast, but low precision. It can 
detect at most 20 faces in one image

### Response

```json

{
    "faces": [
        [
            0.05298403650522232,
            0.2934165894985199,
            0.8788653612136841,
            0.7332860827445984
        ]
    ],
    "format": [
        "ymin",
        "xmin",
        "ymax",
        "xmax"
    ],
    "model": {
        "dataset": "WIDERFace",
        "info": "Faster RCNN Object-Detection @Iteration 4066",
        "url": "https://github.com/tensorflow/models/tree/master/research/object_detection"
    },
    "scores": [
        0.9723447561264038
    ],
    "version": "Tensorflow 1.13.1"
}
``` 

## Face Comparison

### /api/v1/face_comparison

### Method: Post

### Request body (form data)
- face_1: Image file that has face 1
- face_2: Image file that has face 2
- face_1_url(optional)
- face_2_url(optional)

Both images should be in same form. Mixing file and url is not
 supported currently

### Response

```json
{
    "Model": {
        "Info": "Siamese Network with Inception ResNet @ Iteration 4313 Epoch 4",
        "dataset": "vggface2",
        "url": "https://github.com/wylswz/twitter_pride_vanity/tree/master/algorithms/siamese_net"
    },
    "Version": "Keras Application 1.0.7 @ Tensorflow 1.13 Backend",
    "similarity": 0.9989643096923828
}

```

# Project Structure
```
├── APISample
├── archive
├── BenchMark
├── bibli
├── docker-compose.yml
├── Dockerfile
├── export.sh
├── face_detection_config
├── face_detection_dataset
├── face_detection_wrapper
├── model_history_log.csv
├── README.md
├── requirements.txt
├── shared
├── siamese_net
├── siamese_wrapper
├── test_pics
├── train.sh
├── venv
└── web

13 directories, 7 files
(venv) johnny@johnny-Blade:~/twitter_pride_vanity/algorithms$ tree -L 2
.
├── APISample
│   ├── __init__.py
│   ├── __pycache__
│   ├── Visualization.py
│   └── WEB.py
├── archive
│   ├── builders
│   ├── feature_extractor
│   ├── __init__.py
│   ├── settings.py
│   ├── siamese_arch
│   ├── stream.py
│   └── test.py
├── BenchMark
│   ├── batch4lr0.004
│   ├── batch8lr0.004
│   ├── batch8lr0.004decay0.8
│   ├── benchmark
│   ├── benchmark_frcnn
│   ├── postprocess.py
│   ├── pressure.py
│   └── test.jpg
├── bibli
│   ├── facerecog.pdf
│   ├── ImagecompareCNN.pdf
│   ├── resnet.pdf
│   ├── siamese.pdf
│   ├── speed.pdf
│   ├── ssd.pdf
│   └── TGRS-RICNN.pdf
├── docker-compose.yml
├── Dockerfile
├── export.sh
├── face_detection_config
│   ├── faster_rcnn_resnet101_voc07.config
│   ├── __init__.py
│   ├── SSD.config
│   └── SSD_oom.config
├── face_detection_dataset
│   ├── config.py
│   ├── face.pbtxt
│   └── WIDER_FACE_TFRecord.py
├── face_detection_wrapper
│   ├── __init__.py
│   ├── __pycache__
│   ├── start.sh
│   └── wrapper.py
├── model_history_log.csv
├── README.md
├── requirements.txt
├── shared
│   ├── __init__.py
│   ├── ModelWrapper.py
│   └── __pycache__
├── siamese_net
│   ├── __init__.py
│   ├── model_utils
│   ├── __pycache__
│   └── siamese.py
├── siamese_wrapper
│   ├── __init__.py
│   ├── __pycache__
│   └── SiameseWrapper.py
├── test_pics
│   ├── chan4.jpg
│   ├── chan.jpg
│   ├── chrisevans.jpg
│   ├── test.jpg
│   ├── test.png
│   ├── thor_bro
│   ├── thor.jpeg
│   ├── trump1.jpg
│   └── trump2.jpg
├── train.sh
└── web
    ├── __init__.py
    ├── pressure_test.py
    ├── __pycache__
    └── web_server.py


```