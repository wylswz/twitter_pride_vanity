import json
import os
import threading
import time
import traceback
import requests
import tweepy

from demo.face_utils import face_with_high_scores, detect_face,face_pair
from demo.config import BASE_DIR, face_detection_threshold, face_similarity_threshold, TMP_DIR
from demo.sys_utils import cache_image


class MyStreamListener(tweepy.StreamListener):
    counter = 0

    def on_data(self, data):
        self.counter += 1
        try:
            plt_imgs = []
            plt_bboxes = []
            all_data = json.loads(data)
            user = all_data['user']
            if user['screen_name'] == 'comp90048test':
                print("catch me")
            user_profile = user['profile_image_url'].replace('_normal', '')
            media = all_data['entities']['media']
            media_urls = []
            for m in media:
                if m['type'] == 'photo':
                    media_url = m['media_url']
                    media_urls.append(media_url)
            user_path = os.path.join(BASE_DIR, 'twitter', user['id_str'], all_data['id_str'])

            if not os.path.isdir(user_path):
                os.makedirs(user_path)
            profile_file_name = user_profile.split('/')[-1]
            profile_file_name = os.path.join(user_path, profile_file_name)
            cache_image(user_profile, profile_file_name)
            with open(profile_file_name, 'rb') as fp:
                resp = detect_face(fp)
                faces = resp['faces']
                faces = face_with_high_scores(faces, resp['scores'], face_detection_threshold)
                plt_imgs.append(profile_file_name)
                plt_bboxes.append(faces)


            for url in media_urls:
                image_file_name = url.split('/')[-1]
                image_file_name = os.path.join(user_path, image_file_name)
                cache_image(url, image_file_name)
                with open(image_file_name, 'rb') as fp:
                    resp = detect_face(fp)
                    faces = resp['faces']
                    faces = face_with_high_scores(faces, resp['scores'], face_detection_threshold)
                    plt_imgs.append(image_file_name)
                    plt_bboxes.append(faces)
            face_pair(plt_imgs, plt_bboxes)
        except Exception as e:
            traceback.print_exc()
        return True

    def on_error(self, err):
        print(err)


def get_client(custom_token, custom_secret, access_token, access_secret):
    auth = tweepy.OAuthHandler(custom_token, custom_secret)
    auth.set_access_token(access_token, access_secret)
    api = tweepy.API(auth)
    my_stream_listener = MyStreamListener()
    my_stream = tweepy.Stream(auth=api.auth, listener=my_stream_listener)
    return my_stream