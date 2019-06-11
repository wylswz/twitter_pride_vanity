"""
Real time tweet streaming demo

Functionality
- Catch tweets sent by people in the following list
- Capture human faces in tweets
- Compare the faces with the face in user's profile picture

"""

from demo.config import custom_secret, custom_key, access_secret, access_token, following_list
from demo.tweepy_client import get_client

def main():
    GEOBOX_AUSTRALIA = [112.921114, -43.740482, 159.109219, -9.142176]
    my_stream = get_client(custom_key, custom_secret, access_token, access_secret)
    my_stream.filter(follow=following_list)


if __name__ == '__main__':
    main()