import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import requests
import os
import uuid
from PIL import Image


def bounding_box(img: Image, bboxes: list):
    (x, y) = img.size
    for bbox in bboxes:
        [ymin, xmin, ymax, xmax] = [
            y * bbox[0],
            x * bbox[1],
            y * bbox[2],
            x * bbox[3]
        ]

        plt.imshow(img)
        rect = Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=1, edgecolor='r', facecolor='none')
        plt.gca().add_patch(rect)
    plt.show()


def im_get(url):
    resp = requests.get(url)
    temp_file_name = os.path.join("/zeppelin/data/", str(uuid.uuid4()))
    with open(temp_file_name, 'wb') as fptr:
        fptr.write(resp.content)

    img = Image.open(temp_file_name)
    os.remove(temp_file_name)
    return img
