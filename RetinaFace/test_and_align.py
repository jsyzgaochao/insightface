from __future__ import print_function
import os
import cv2
import pickle
import gzip
import numpy as np
from skimage import transform as trans
from retinaface import RetinaFace

thresh = 0.7
flip = False
im_scale = 1.0
img_dir = "pic"
cut_dir = "cut"

gpuid = 0
# detector = RetinaFace('./model/R50', 0, gpuid, 'net3')
detector = RetinaFace('./model_mnet/mnet.25', 0, gpuid, 'net3')

arcface_src = np.array([
  [38.2946, 51.6963],
  [73.5318, 51.5014],
  [56.0252, 71.7366],
  [41.5493, 92.3655],
  [70.7299, 92.2041] ], dtype=np.float32 )

def makedir_if_not_exist(p):
    if not os.path.exists(p):
        os.makedirs(p)
        
def estimate_norm(lmk):
    assert lmk.shape==(5,2)
    tform = trans.SimilarityTransform()
    tform.estimate(lmk, arcface_src)
    M = tform.params[0:2,:]
    return M

def norm_crop(img, landmark, image_size=112):
    M = estimate_norm(landmark)
    warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue = 0.0)
    return warped

makedir_if_not_exist(cut_dir)
img_list = os.listdir(img_dir)
img_list.sort(key=lambda x: int(os.path.splitext(x)[0]))
for idx, img_name in enumerate(img_list):
    img_path = os.path.join(img_dir, img_name)
    img = cv2.imread(img_path)
    faces, landmarks = detector.detect(img, thresh, scales=[im_scale], do_flip=flip)
    data.append((img_name, faces, landmarks))
    num = faces.shape[0]
    if num != 1:
        print("[{}/{}]".format(idx+1, len(img_list)), img_name, num, faces.shape, landmarks.shape)
        continue
    sub_img = norm_crop(img, landmarks[0].reshpae(5, 2))
    cv2.imwrite(os.path.join(cut_dir, img_name), sub_img)
