import requests

def cache_image(im_url, path):
    resp = requests.get(im_url)
    im_bytes = resp.content
    with open(path, 'wb') as fptr:
        fptr.write(im_bytes)
