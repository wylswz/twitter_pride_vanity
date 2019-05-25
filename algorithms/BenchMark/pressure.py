import requests
import time
from multiprocessing import Pool
test_url = "https://www.faceplusplus.com/scripts/demoScript/images/demo-pic4.jpg"

comparison_url = 'http://45.113.235.235:8000/api/v1/face_detection'

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
    return r

def frcnn_wrapper(n):
    test_img = open('./test.jpg', 'rb')
    r = detection(test_img, 1, 'FASTER_RCNN_RESNET_101')
    return r

def pressure_ssd(num_instances=1000, num_threads=10):
    a = list(range(num_instances))
    with Pool(num_threads) as p:
        p.map(ssd_wrapper, a)


def pressure_frcnn(num_instances=1000, num_threads=10):
    a = list(range(num_instances))
    with Pool(num_threads) as p:
        p.map(frcnn_wrapper, a)


if __name__ == "__main__":
    for num_thread in range(2, 26, 2):
        start = time.time()
        pressure_frcnn(100, num_thread)
        time_consumption = time.time() - start
        print("Tested ssd with {0} threads on {1} samples, total time: {2}".format(num_thread,300,time_consumption))

