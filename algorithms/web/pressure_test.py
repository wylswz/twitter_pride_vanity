import requests

FILE_1 = "../test_pics/trump1.jpg"
FILE_2 = "../test_pics/trump2.jpg"

def detection():
    resp = requests.post(
        "http://115.146.92.137:8000/api/v1/face_detection",
        data={
            "model": "SSD_MOBILENET_V2",
            "limit": 5
        },
        files={
            'image_file':open(FILE_1, 'rb')
        }
    )
    return resp.content

def comparison():
    pass

if __name__ == "__main__":
    for i in range(100):
        c = detection()
        print(i)
    print(c)