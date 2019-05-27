"""
Maintainer: Yunlu Wen <yunluw@student.unimelb.edu.au>
This is a pressure test script for face detection web service. The concurrency
performance of web service is tested by sending requests with different number
of threads. This is also used to figure out the optimal concurrency level of the 
API. One recent test result is：

Tested ssd with 2 threads on 300 samples, total time: 8.13038158416748
Tested ssd with 4 threads on 300 samples, total time: 5.729494094848633
Tested ssd with 6 threads on 300 samples, total time: 5.330998420715332
Tested ssd with 8 threads on 300 samples, total time: 4.239251136779785
Tested ssd with 10 threads on 300 samples, total time: 3.9432687759399414
Tested ssd with 12 threads on 300 samples, total time: 3.3242716789245605
Tested ssd with 14 threads on 300 samples, total time: 3.2233057022094727
Tested ssd with 16 threads on 300 samples, total time: 3.334256172180176
Tested ssd with 18 threads on 300 samples, total time: 3.332061767578125
Tested ssd with 20 threads on 300 samples, total time: 3.233137607574463
Tested ssd with 22 threads on 300 samples, total time: 3.2334392070770264
Tested ssd with 24 threads on 300 samples, total time: 3.336984872817993
"""

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

