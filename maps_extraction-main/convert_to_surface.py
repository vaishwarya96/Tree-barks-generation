import cv2
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--surface_path', type=str, help='Path to the txt files')
args = parser.parse_args()

rad_files = os.listdir(args.surface_path)

for rad in rad_files:
    filename = os.path.splitext(rad)[0]
    im = np.loadtxt(os.path.join(args.surface_path,rad))
    im = im.reshape(-1, 720)
    im = (im*65535).astype(np.uint16)
    cv2.imwrite(os.path.join(args.surface_path,filename+'.png'), im)

