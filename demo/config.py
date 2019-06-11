"""
Maintainer: Yunlu Wen <yunluw@student.unimelb.edu.au>

Configurations for demo
"""

import os
BASE_DIR = './temp/'
TMP_DIR = os.path.join(BASE_DIR, 'tmp')
BASE_URL = 'http://45.113.235.235:8000'
DETECTION_URL = '/api/v1/face_detection'
COMPARISON_URL = '/api/v1/face_comparison'

custom_key = 'lmi60axPXTB1XZc7Us7o3PKWE'
custom_secret = 'gMkhRWPnnwcUpNhg2QqsADuLC4PR66opqlX1J6srpfBONunR7X'
access_token = '1107074333698981889-bzffZA2f5djoFeV30veP1oHBLNXeab'
access_secret = 'CMckhw2NaULIqKnp5DcIqcwJaE1nAyfT2SMqzbnxllGoY'

face_detection_threshold = 0.4
face_similarity_threshold = 0.75

min_face_size = 48

following_list = ['1128897542630981632']