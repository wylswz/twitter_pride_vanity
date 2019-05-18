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