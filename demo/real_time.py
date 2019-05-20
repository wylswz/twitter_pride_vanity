
import matplotlib.pyplot as plt
import threading
from multiprocessing import Process
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import matplotlib
import requests
import json
import os
import uuid
import tweepy
import tweepy
import json
import traceback
import time
from PIL import Image
import io

BASE_DIR = './temp/'

BASE_URL = 'http://45.113.235.235:8000'
DETECTION_URL = '/api/v1/face_detection'
COMPARISON_URL = '/api/v1/face_comparsion'


def bounding_box(img_file, bboxes:list):
    
    img = Image.open(img_file)
    (x,y) = img.size
    for bbox in bboxes:
        [ymin, xmin, ymax, xmax] = [
            y * bbox[0],
            x * bbox[1],
            y * bbox[2],
            x * bbox[3]
            ]
        
        
        rect = Rectangle((xmin,ymin),xmax-xmin,ymax-ymin, linewidth=1, edgecolor='r',facecolor='none')
        plt.gca().add_patch(rect)
    plt.imshow(img)
    print("Nonblocking")
    plt.show()
    #print("show finished")
    
def face_with_high_scores(faces,scores, threshold):
    real_faces = []
    for i in range(len(faces)):
        if scores[i] > threshold:
            real_faces.append(faces[i])
    return real_faces
def detect_face(image_file, limit=5, model='SSD_MOBILENET_V2'):
    resp = requests.post(BASE_URL+DETECTION_URL,
        files={
            'image_file':image_file  
        },
        data={
            'limit':limit,
            'model':model,
        })
    
    res =resp.content
    return res
    
#override tweepy.StreamListener to add logic to on_status
GEOBOX_AUSTRALIA = [112.921114, -43.740482, 159.109219, -9.142176]

class MyStreamListener(tweepy.StreamListener):
    
    counter = 0
    def on_data(self, data):
        self.counter += 1
        try:
            all_data = json.loads(data)
            user = all_data['user']
            if user['screen_name'] == 'comp90048test':
                print("catch me")
            user_profile = user['profile_image_url'].replace('_normal','')
            media = all_data['entities']['media']
            media_urls = []
            for m in media:
                if m['type'] == 'photo':
                    media_url = m['media_url']
                    media_urls.append(media_url)
            user_path = os.path.join(BASE_DIR,'twitter', user['id_str'], all_data['id_str'])
            
            if not os.path.isdir(user_path):
                os.makedirs(user_path)
                
            for url in media_urls:
                resp = requests.get(url)
                bin = resp.content
                image_file_name = url.split('/')[-1]
                image_file_name = os.path.join(user_path, image_file_name)
                with open(image_file_name, 'wb') as fp:
                    fp.write(bin)
                with open(image_file_name, 'rb') as fp:
                    resp = detect_face(fp)
                    resp = json.loads(resp.decode())
                    faces = resp['faces']
                    faces = face_with_high_scores(faces,resp['scores'],0.4)
                    t = threading.Thread(target=bounding_box, args=(fp, faces))
                    t.start()
                    print(faces)
                    time.sleep(2)

                    
        except Exception as e:
            print(e)
        return True
        
    def on_error(self, err):
        print(err)
        
auth = tweepy.OAuthHandler('lmi60axPXTB1XZc7Us7o3PKWE', 'gMkhRWPnnwcUpNhg2QqsADuLC4PR66opqlX1J6srpfBONunR7X')
auth.set_access_token('1107074333698981889-bzffZA2f5djoFeV30veP1oHBLNXeab', 'CMckhw2NaULIqKnp5DcIqcwJaE1nAyfT2SMqzbnxllGoY')
api = tweepy.API(auth)

myStreamListener = MyStreamListener()
myStream = tweepy.Stream(auth = api.auth, listener=myStreamListener)
myStream.filter(follow=['1128897542630981632'])