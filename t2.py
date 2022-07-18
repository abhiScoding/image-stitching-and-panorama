# 1. Only add your code inside the function (including newly improted packages). 
#  You can design a new function and call the new function in the given functions. 
# 2. For bonus: Give your own picturs. If you have N pictures, name your pictures such as ["t3_1.png", "t3_2.png", ..., "t3_N.png"], and put them inside the folder "images".
# 3. Not following the project guidelines will result in a 10% reduction in grades

import cv2
import numpy as np
import matplotlib.pyplot as plt
import json


def show_image(img, delay=5000):
    cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('image', img)
    cv2.waitKey(delay)
    cv2.destroyAllWindows()

# extract image rectangle from padding
def extImg(pimg):
    grImg = cv2.cvtColor(pimg, cv2.COLOR_BGR2GRAY)
    slice = cv2.findNonZero(grImg)
    a, b, c, r = cv2.boundingRect(slice)
    img = pimg[b:b+r, a:a+c] 
    
    return img


# perform matching
def match(kp_des1,kp_des2):

    srcPts = []
    destPts = []
    for k1, d1 in kp_des1.items():
        min_dist = np.inf
        for k2, d2 in kp_des2.items():
            diff = d1 - d2
            sqr = np.square(diff)
            SSD = np.sum(sqr)
            srt = np.sqrt(SSD)

            if srt<min_dist:
                min_dist = srt
                kdst = k2

        # print(min_dist)   
        if min_dist < 70:
            srcPts.append(k1)
            destPts.append(kdst)

    spts = np.array(srcPts)
    dpts = np.array(destPts)
    return spts, dpts

def findCentroid(pts):
    sumArr = np.sum(pts,axis=0)
    x_c = sumArr[0]/len(pts)
    y_c = sumArr[1]/len(pts)
    
    return x_c, y_c

# returns True if overlap else False
def overlap(img1,img2):
    sift = cv2.xfeatures2d.SIFT_create(600)

    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    # store kpts and descriptors in a dictionary
    minKpts = min(len(kp1),len(kp2))
    kp_des1 = {}
    kp_des2 = {}
    for i in range(minKpts):
        kp_des1[kp1[i].pt] = des1[i]
        kp_des2[kp2[i].pt] = des2[i]

    # perform matching to get good matching points
    srcPoints, dstPoints  = match(kp_des1,kp_des2)
    # number of good maching points
    noOFmatches = len(srcPoints)
    # print("#of matches: ",noOFmatches)

    return noOFmatches > 15

# join two images
def join(img1, img2):
   

    # using SIFT feature extraction
    sift = cv2.xfeatures2d.SIFT_create(600)

    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    # store kpts and descriptors in a dictionary
    minKpts = min(len(kp1),len(kp2))
    kp_des1 = {}
    kp_des2 = {}
    for i in range(minKpts):
        kp_des1[kp1[i].pt] = des1[i]
        kp_des2[kp2[i].pt] = des2[i]

    # perform matching to get points for homography 
    srcPoints, dstPoints  = match(kp_des1,kp_des2)

    # computing Homography
    M, mask = cv2.findHomography(srcPoints, dstPoints, cv2.RANSAC, 5.0)
    
    # applying padding to img1 
    # calculating all lengths of imgs
    h1, w1 = img1.shape[0], img1.shape[1]
    h2, w2 = img2.shape[0], img2.shape[1]
    d1 = int(np.sqrt(h1**2+w1**2))
    d2 = int(np.sqrt(h2**2+w2**2))
    # pad amout img1 equal to biggest length in img1
    img1_pd = np.pad(img1, ((d1, d1), (d1, d1), (0, 0)), mode='constant', constant_values=0)

    # diagonal of padded img1
    h1p, w1p = img1_pd.shape[0], img1_pd.shape[1]
    d1p = int(np.sqrt(h1p**2+w1p**2))
    
    # warped img size equal to sum of diagonals of padded img1 and img2
    wimg1 = cv2.warpPerspective(img1_pd,M,(d1p+d2,d1p+d2))

    # overlap warped img1 and img2 by translating img2
    # finding coordinates of matching points 
    kp3, des3 = sift.detectAndCompute(wimg1,None)

    kp_des3 = {}
    for i in range(len(kp3)):
        kp_des3[kp3[i].pt] = des3[i]

    pts3, pts2 = match(kp_des3,kp_des2)
    
    # find cetroid of matching points in both the imgs
    x3_c, y3_c = findCentroid(pts3)
    x2_c, y2_c = findCentroid(pts2)

    # translation img2 by amount of cetroid difference to overlap with warped img1
    xtrans, ytrans = np.rint(x3_c - x2_c), np.rint(y3_c - y2_c)

    # put pixels of img2 in warped image1 to get stitched img
    wimg1[int(ytrans):int(ytrans)+img2.shape[0],int(xtrans):int(xtrans)+img2.shape[1]] = img2
    
    # extract img from padding(black surrounding)
    stitchedImg = extImg(wimg1)

    return stitchedImg


def stitch(imgmark, N=5, savepath=''): #For bonus: change your input(N=*) here as default if the number of your input pictures is not 4.
    "The output image should be saved in the savepath."
    "The intermediate overlap relation should be returned as NxN a one-hot(only contains 0 or 1) array."
    "Do NOT modify the code provided."
    imgpath = [f'./images/{imgmark}_{n}.png' for n in range(1,N+1)]
    imgs = []
    for ipath in imgpath:
        img = cv2.imread(ipath)
        imgs.append(img)
    "Start you code here"

    overlap_arr = np.zeros((N,N))

    # element of overlap array assign value 1 if two images overlap or both images are the same
    for img1idx in range(N):
        for img2idx in range(N):
            if img1idx == img2idx:
                overlap_arr[img1idx][img2idx] = 1
            elif overlap(imgs[img1idx],imgs[img2idx]):
                overlap_arr[img1idx][img2idx] = 1
    print(overlap_arr)
    print("creating panorama....")

    # geting list of images to be stitched
    sumCols = np.sum(overlap_arr,axis=0)

    imgsToStitch = []
    for i in range(N):
        if sumCols[i] > 1:
            imgsToStitch.append(imgs[i])

    # creating panorama by stitching all images from the list,imgsToStitch
    panorama = join(imgsToStitch[0],imgsToStitch[1])
    for imgIdx in range(2,len(imgsToStitch)):
        panorama = join(panorama,imgsToStitch[imgIdx])

    # show_image(panorama,4000)
    cv2.imwrite(savepath, panorama)
    print("No of images stitched: ",len(imgsToStitch))
    print("Bouns part panorama...")

    return overlap_arr



if __name__ == "__main__":
    #task2
    overlap_arr = stitch('t2', N=4, savepath='task2.png')
    with open('t2_overlap.txt', 'w') as outfile:
        json.dump(overlap_arr.tolist(), outfile)
    #bonus
    overlap_arr2 = stitch('t3', savepath='task3.png')
    with open('t3_overlap.txt', 'w') as outfile:
        json.dump(overlap_arr2.tolist(), outfile)
