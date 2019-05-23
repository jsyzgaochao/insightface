import os
import cv2
import pickle
import gzip
import numpy as np
from retinaface import RetinaFace

thresh = 0.8
flip = False
im_scale = 0.5
img_dir = "pic"
gpuid = 0
detector = RetinaFace('./model/R50', 0, gpuid, 'net3')

data = []
for img_name in os.listdir(img_dir):
    img_path = os.path.join(img_dir, img_name)
    img = cv2.imread(img_path)
    faces, landmarks = detector.detect(img, thresh, scales=[im_scale], do_flip=flip)
    print(faces.shape, landmarks.shape)
    num = faces.shape[0]
    data.append({"img_name": img_name, "faces": faces, "landmarks": landmarks})

with gzip.open('pred.pkl.gz') as fp:
    pickle.dump(data, fp)
