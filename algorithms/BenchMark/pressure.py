import requests
from multiprocessing import Pool
test_url = "https://www.faceplusplus.com/scripts/demoScript/images/demo-pic4.jpg"

comparison_url = 'http://103.6.254.149:8000/api/v1/face_detection'

def detection(image_file, limit, model):
    resp = requests.post(
            comparison_url,
            files={
                'image_file': image_file
            },
            data={
                'limit': limit,
                'model':model
            }
        )
    return resp.content
 
def ssd_wrapper(n):
    test_img = open('./test.jpg', 'rb')
    r = detection(test_img, 1, 'SSD_MOBILENET_V2')
    print(r)
    return r

def frcnn_wrapper(n):
    test_img = open('./test.jpg', 'rb')
    r = detection(test_img, 1, 'FASTER_RCNN_RESNET_101')
    print(r)
    return r

def pressure():
    a = list(range(100))
    with Pool(4) as p:
        p.map(ssd_wrapper, a)
    with Pool(4) as p:
        p.map(frcnn_wrapper,a)

if __name__ == "__main__":
    pressure()