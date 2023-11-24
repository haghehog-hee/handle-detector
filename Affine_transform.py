import cv2 as cv
import numpy as np
import os

#path = "C:\\Users\\MuhametovRD\\AppData\\Roaming\\EasyClient\\Picture\\"
path = "C:\\Tensorflow\\Dataset\\Images\\"
savepath = "C:\\Tensorflow\\Dataset\\affine\\"
thresh = 0
IMAGE_PATHS = os.listdir(path)
canny = True
cn = ""

for PATH in IMAGE_PATHS:
    img = cv.imread(path+PATH)
    assert img is not None, "file could not be read, check with os.path.exists()"
    rows,cols,ch = img.shape

    pts1 = np.float32([[0,rows*0.05],[cols*1.05,rows*0.05],[cols*0.1,rows*0.98],[cols*0.95,rows]])
    pts2 = np.float32([[0,0],[cols,0],[0,rows],[cols,rows]])
    M = cv.getPerspectiveTransform(pts1,pts2)
    dst = cv.warpPerspective(img,M,(cols,rows))
    first_third = int(cols/3)
    second_third = cols - first_third
    if canny:
        cn = "canny"
        if img[0][0][0] == img[0][0][1] and img[0][0][0] == img[0][0][2]:
            alpha = 1
            beta = 1.2
            dst = cv.convertScaleAbs(dst, alpha, beta)
            dst = cv.Canny(dst, 30, 40)
        else:
            dst = cv.Canny(dst, 80, 110)

    cropped_image1 = dst[0:rows, 0:first_third]
    cropped_image2 = dst[0:rows, first_third:second_third]
    cropped_image3 = dst[0:rows, second_third:cols]

    #cv.imwrite(savepath + PATH, dst)
    cv.imwrite(savepath + cn + "cropped1" + PATH, cropped_image1)
    cv.imwrite(savepath + cn + "cropped2" + PATH, cropped_image2)
    cv.imwrite(savepath + cn + "cropped3" + PATH, cropped_image3)

