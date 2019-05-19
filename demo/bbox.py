import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import requests
import os
import uuid
from PIL import Image


test_url = "https://www.faceplusplus.com/scripts/demoScript/images/demo-pic4.jpg"
def bounding_box(img:Image, bbox:list):
    (x,y) = img.size
    [ymin, xmin, ymax, xmax] = [
        y * bbox[0],
        x * bbox[1],
        y * bbox[2],
        x * bbox[3]
        ]
    
    plt.imshow(img)
    rect = Rectangle((xmin,ymin),xmax-xmin,ymax-ymin, linewidth=1, edgecolor='r',facecolor='none')
    plt.gca().add_patch(rect)
    plt.show()

def im_get(url):
    resp = requests.get(url)
    temp_file_name = os.path.join("/zeppelin/data/",str(uuid.uuid4()))
    with open(temp_file_name, 'wb') as fptr:
        fptr.write(resp.content)
    
    img = Image.open(temp_file_name)
    os.remove(temp_file_name)
    return img
        
    
    
img = im_get(test_url)
bbox = [
            0.1269296258687973,
            0.23900650441646576,
            0.7658754587173462,
            0.731124222278595
        ]
bounding_box(img, bbox)