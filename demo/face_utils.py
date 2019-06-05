"""
Maintainer: Yunlu Wen <yunluw@student.unimelb.edu.au>

Utilities
- Showing images
- Plotting bounding boxes
- Detecting faces
- Comparing faces
- Face with high scores
"""

import matplotlib.pyplot as plt
import requests
import json
from io import BytesIO
from PIL import Image
from matplotlib.patches import Rectangle

from demo.config import BASE_URL, DETECTION_URL, COMPARISON_URL, TMP_DIR, face_similarity_threshold, min_face_size


def bounding_box(img_file, bboxes: list):
    plt.figure()
    img = Image.open(img_file)
    (x, y) = img.size
    for bbox in bboxes:
        [ymin, xmin, ymax, xmax] = [
            y * bbox[0],
            x * bbox[1],
            y * bbox[2],
            x * bbox[3]
        ]

        rect = Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=1, edgecolor='r', facecolor='none')
        plt.gca().add_patch(rect)
    plt.imshow(img)
    plt.show(block=False)


def face_pair(image_files, bboxes_lst, prof_file, prof_bboxes):
    """
    - Crop all the faces in profile picture
    - Crop all the faces in each post
    - Compare the similarity between faces in profile pic and post
    - Take the face with highest similarity
    - Plot the pie chart
    :param prof_bboxes: List of bounding boxes in profile picture
    :param prof_file: [str] profile image file
    :param image_files: The image file in tweet
    :param bboxes_lst: List of bounding boxes in tweet picture
    :return:
    """
    plt.clf()
    num_picture = len(image_files) + 1
    # Plus profile photo
    fig = plt.figure(num_picture)
    cnt = 1
    assert len(image_files) == len(bboxes_lst)
    for bboxes in bboxes_lst:
        if len(bboxes) < 1:
            return
    sim = []
    profile_faces = []
    profile_img = prof_file[0]
    profile_bboxes = prof_bboxes
    profile = plt.imread(profile_img)

    ax = fig.add_subplot(1, num_picture + 1, cnt, aspect='equal')
    ax.title.set_text("Profile picture")
    ax.title.set_text("Faces occurred in post")
    ax.set_axis_off()
    plt.imshow(profile)

    for bbox in profile_bboxes:
        """
        Construct a list of faces cropped from the profile picture
        """
        (y, x, _) = profile.shape
        [ymin, xmin, ymax, xmax] = [
            int(y * bbox[0]),
            int(x * bbox[1]),
            int(y * bbox[2]),
            int(x * bbox[3])
        ]
        tmp_file = BytesIO()
        Image.fromarray(profile[ymin:ymax, xmin:xmax]).save(tmp_file,format='png')
        tmp_file.seek(0)
        profile_faces.append(
            tmp_file
        )

    for (image, bboxes) in zip(image_files, bboxes_lst):
        """
        For each photo in the tweet, create a new subplot
        """
        img = plt.imread(image)

        (y, x, _) = img.shape
        cnt += 1
        ax = fig.add_subplot(1, num_picture+1, cnt, aspect='equal')
        ax.title.set_text("Profile picture")
        ax.title.set_text("Faces occurred in post")

        ax.set_axis_off()
        plt.imshow(img)
        for bbox in bboxes:
            """
            For each face in the post, compare it to all the photos in profile picture
            if the similarity is higher than the threshold, draw a green bounding box
            otherwize draw a red one
            """
            [ymin, xmin, ymax, xmax] = [
                int(y * bbox[0]),
                int(x * bbox[1]),
                int(y * bbox[2]),
                int(x * bbox[3])
            ]

            if abs(ymin - ymax) > min_face_size and abs(xmax - xmin) > min_face_size:
                this_face = BytesIO()
                Image.fromarray(img[ymin:ymax, xmin:xmax]).save(this_face,format='png')
                this_face.seek(0)
                tmp_sims = []
                for profile_face in profile_faces:
                    this_face.seek(0)
                    profile_face.seek(0)
                    similarity = compare_face(this_face, profile_face)['similarity']
                    sim.append(
                        similarity
                    )
                    tmp_sims.append(similarity)
                color = 'r'
                if max(tmp_sims) > face_similarity_threshold:
                    color = 'g'
                rect = Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=1, edgecolor=color, facecolor='none')
                ax.add_patch(rect)
            else:
                pass
    if len(sim) < 1:
        return

    max_sim = max(sim)
    labels = ['Narcissism', 'Non-Narcissism']
    fig.add_subplot(1, num_picture + 1, num_picture + 1, aspect='equal')
    plt.pie([max_sim, 1-max_sim], labels=labels)
    plt.show()




def face_with_high_scores(faces, scores, threshold):
    """
    Find bboxes of faces in the image with a score above threshold
    :param faces: List of bboxes
    :param scores: List of scores
    :param threshold:
    :return:
    """
    real_faces = []
    for i in range(len(faces)):
        if scores[i] > threshold:
            real_faces.append(faces[i])
    return real_faces


def detect_face(image_file, limit=5, model='SSD_MOBILENET_V2'):
    """
    Detect all the faces in an image
    :param image_file:
    :param limit:
    :param model:
    :return:
    """
    resp = requests.post(BASE_URL + DETECTION_URL,
                         files={
                             'image_file': image_file
                         },
                         data={
                             'limit': limit,
                             'model': model,
                         })

    res = resp.content
    return json.loads(res)


def compare_face(face_1, face_2):
    resp = requests.post(
        BASE_URL + COMPARISON_URL,
        files={
            'face_1': face_1,
            'face_2': face_2
        }
    )

    return json.loads(resp.content)