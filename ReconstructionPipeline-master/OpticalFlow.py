import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse
import os
def make_folder(path):
    try:
        os.mkdir(path)
    except:
        pass

# ARGS
parser = argparse.ArgumentParser()
parser.add_argument('--debug', type=bool, default=False)
parser.add_argument('--use_angular_mask', type=bool, default=False)
parser.add_argument('--video_path', type=str)
parser.add_argument('--tree_id', type=str)
parser.add_argument('--output', type=str)
args = parser.parse_args()

make_folder(args.output)
make_folder(os.path.join(args.output,'labels'))
make_folder(os.path.join(args.output,'images'))
make_folder(os.path.join(args.output,'labels',args.tree_id))
make_folder(os.path.join(args.output,'images',args.tree_id))
if args.debug:
    make_folder(os.path.join(args.output,'optical_flow'))
    make_folder(os.path.join(args.output,'optical_flow',args.tree_id))

cap = cv2.VideoCapture(args.video_path)
ret, frame = cap.read()
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
hsvImg = np.zeros_like(frame)
hsvImg[..., 1] = 255
count = 0  
L = []
L2 = []
kernel = np.ones((30,30),np.uint8)


if (cap.isOpened() == False): 
    raise Exception("Error opening the video")

while(cap.isOpened()):
    if count%2 == 0: 
        previousGray = gray
    ret , frame = cap.read()

    if ret:
        if count%2 == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if count%10 == 0:
            flow = cv2.calcOpticalFlowFarneback(previousGray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
            #if args.debug:
            #    cv2.imshow('mag', mag); cv2.waitKey(30)
        
            hsvImg[..., 0] = 0.5 * ang * 180 / np.pi
            hsvImg[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

            #if args.debug:
            #    rgbImg = cv2.cvtColor(hsvImg, cv2.COLOR_HSV2BGR)
            #    cv2.imshow('mag', rgbImg); cv2.waitKey(30)
            #    cv2.imwrite(os.path.join(args.output,'optical_flow','flow'+str(count)+'.png', rgbImg))
            
            threshold = 0.40
            max_mag = np.max(mag)
        
            mag_mask = (mag >= (threshold * max_mag))

            if args.use_angular_mask:
                ang_degrees = ang * 180/np.pi
                max_ang = np.max(ang_degrees)
                ang_mask = (ang_degrees < (threshold * max_ang))
                mask = np.logical_or(mag_mask, ang_mask)
            else:
                mask = mag_mask
        
            m = np.zeros(mask.shape, dtype=np.uint8)
            m[np.where(mask == 0)] = 0
            m[np.where(mask == 1)] = 255
            
            if np.mean(mag)<20:
                if ((1.0*np.sum(m==255))/(m.shape[0]*m.shape[1]))>0.25:
                    num_labels, labels_im = cv2.connectedComponents(m)
                    max_ = 0
                    max_id = 0
                    for i in range(1,num_labels):
                        if np.sum(labels_im  == i) > max_:
                            max_ = np.sum(labels_im  == i)
                            max_id = i
                    m = (labels_im==max_id)*255
                    m = m.astype(np.uint8)
                    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)

                    contours, hierarchy = cv2.findContours(m, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                    m = np.zeros_like(m,dtype=np.uint8)
                    c = max(contours, key = cv2.contourArea)
                    m = cv2.drawContours(m, [c], -1, (255), -1)
                    edges = cv2.Canny(m*30,50,150,apertureSize = 3)
                    met = np.sum(edges,axis=0)
                    if np.sum(met[0:250]) + np.sum(met[1500:]) == 0:
                        m2 = np.ones_like(m,dtype=np.uint8)*8
                        m2[m==255] = 7
                        assert(np.sum(m2==7) == np.sum(m==255))
                        if args.debug:
                            print(np.mean(mag), np.mean(ang), 1.0*np.sum(m2==7)/(m.shape[0]*m.shape[1]), np.max(ang), np.min(ang), np.max(mag), np.min(mag))
                            plt.imshow(m,cmap='jet')
                            plt.show()
                        cv2.imwrite(os.path.join(args.output,'labels',args.tree_id,str(count)+'.png'), m2)
                        L.append(os.path.join(args.tree_id,str(count)+'.png'))
            cv2.imwrite(os.path.join(args.output,'images',args.tree_id,str(count)+'.png'), frame)
            L2.append(os.path.join(args.tree_id,str(count)+'.png'))

        count += 1
    else:
        break

cap.release()
cv2.destroyAllWindows()

with open(os.path.join(args.output,args.tree_id+'.txt'), 'w') as f:
    for item in L:
        f.write("%s\n" % item)
with open(os.path.join(args.output,args.tree_id+'_full.txt'), 'w') as f:
    for item in L2:
        f.write("%s\n" % item)
