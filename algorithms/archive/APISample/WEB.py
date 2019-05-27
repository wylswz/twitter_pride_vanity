import requests
from PIL import Image
from APISample.Visualization import bounding_box
import json
detection_url = 'http://103.6.254.149:8000/api/v1/face_detection'


def detection(image_file, limit, model):
    resp = requests.post(
            detection_url,
            files={
                'image_file': image_file
            },
            data={
                'limit': limit,
                'model':model
            }
        )
    return resp.content

if __name__ == "__main__":
    image_file = open('../test_pics/chan4.jpg', 'rb')
    res = detection(image_file, limit=4, model='SSD_MOBILENET_V2')
    res = json.loads(res)
    face = res['faces']
    bounding_box(Image.open(image_file), face)