import os
import numpy as np
import glob
import os
import re
import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--debug', type=bool, default=False)
parser.add_argument('--tree_id', type=str)
parser.add_argument('--images_path', type=str)
parser.add_argument('--mask_path', type=str)
parser.add_argument('--output_path', type=str)
args = parser.parse_args()

try:
    os.makedirs(os.path.join(args.output_path,args.tree_id))
except:
    pass

img_path = os.path.join(args.images_path, args.tree_id)
mask_path = os.path.join(args.mask_path, args.tree_id)
img_names = os.listdir(img_path)
img_pathes = [os.path.join(img_path,i) for i in os.listdir(img_path)]
msk_pathes = [os.path.join(mask_path,i) for i in os.listdir(mask_path)]
img_names = [name for _,name in sorted(zip(img_pathes,img_names))]
img_pathes.sort()
msk_pathes.sort()

for path in zip(img_names, img_pathes, msk_pathes):

    msk = cv2.imread(path[2], cv2.IMREAD_UNCHANGED)
    mask = np.zeros(msk.shape[:2], dtype = np.uint8)

    mask[msk[:,:,0]==107] = 255
    mask[msk[:,:,1]==142] = 255
    mask[msk[:,:,2]==35] = 255

    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
    largest_contour = max(contour_sizes, key = lambda x: x[0])[1]
    mask = np.zeros(msk.shape, np.uint8)
    cv2.drawContours(mask, [largest_contour], -1, 255, -1)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30))
    msk_dilated = cv2.dilate(mask, kernel, iterations=1)

    img = cv2.imread(path[1])
    msk_dilated[msk_dilated == 255] = 1
    for i in range(3):
        img[:,:,i] = img[:,:,i] * msk_dilated[:,:,0]

    cv2.imwrite(os.path.join(args.output_path,args.tree_id,path[0].split('.')[0]+'.jpg'), img)
