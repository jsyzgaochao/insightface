from __future__ import print_function
import os
import cv2
import pickle
import gzip
import numpy as np
from retinaface import RetinaFace

thresh = 0.7
flip = False
im_scale = 0.5
img_dir = "pic"
gpuid = 0
detector = RetinaFace('./model/R50', 0, gpuid, 'net3')

data = []
img_list = os.listdir(img_dir)
img_list.sort(key=lambda x: int(os.path.splitext(x)[0]))
for idx, img_name in enumerate(img_list):
    img_path = os.path.join(img_dir, img_name)
    img = cv2.imread(img_path)
    faces, landmarks = detector.detect(img, thresh, scales=[im_scale], do_flip=flip)
    data.append((img_name, faces, landmarks))
    num = faces.shape[0]
    print("[{}/{}]".format(idx+1, len(img_list)), img_name, num, faces.shape, landmarks.shape)

with gzip.open('pred.pkl.gz', "wb") as fp:
    pickle.dump(data, fp)
