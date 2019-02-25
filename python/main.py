import cv2
import matplotlib
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import tkinter
import tkinter.messagebox
import tkinter.filedialog
from tkinter.filedialog import *
from tkinter import *
from tkinter import Label,Tk
from tkinter.ttk import *
from tkinter import font
from PIL import Image, ImageTk
import glob
import plotly.plotly as py

import plotly.graph_objs as go
from plotly.graph_objs import *
import imutils






global co
co = 0

def uploadimage1():
        global ori
        global filename1
        global co
        co = 0

        filename1 = filedialog.askopenfilename()
        img = cv2.imread(filename1)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imshow('image',img)
        ori = img

def uploadimage(event):
        global ori
        global filename1
        global co
        co = 0

        filename1 = filedialog.askopenfilename()
        img = cv2.imread(filename1)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imshow('image',img)
        ori = img

def camera(event):
    cap = cv2.VideoCapture(0)

    while(True):
    # Capture frame-by-frame
        ret, frame = cap.read()

    # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
        cv2.imshow('frame',gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


# When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()







def cameraErosion(event):
    cap = cv2.VideoCapture(0)

    while(True):
    # Capture frame-by-frame
        ret, frame = cap.read()

    # Our operations on the frame come here
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kernel = np.ones((5,5),np.uint8)
        erosion = cv2.erode(img,kernel,iterations = 1)

    # Display the resulting frame
        cv2.imshow('frame',erosion)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()



def cameraSobel(event):
    cap = cv2.VideoCapture(0)

    while(True):
    # Capture frame-by-frame
        ret, frame = cap.read()

    # Our operations on the frame come here
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        sobel = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)

    # Display the resulting frame
        cv2.imshow('frame',sobel)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


def camerafacedetector(event):
    cap = cv2.VideoCapture(0)

    while(True):
    # Capture frame-by-frame
        ret, frame = cap.read()

    # Our operations on the frame come here
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

    # Display the resulting frame
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()






def camerargb(event):
    cap = cv2.VideoCapture(0)

    while(True):
    # Capture frame-by-frame
        ret, frame = cap.read()

    # Our operations on the frame come here


    # Display the resulting frame
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()



def cameraHSV(event):
    cap = cv2.VideoCapture(0)

    while(True):
    # Capture frame-by-frame
        ret, frame = cap.read()

    # Our operations on the frame come here
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Display the resulting frame
        cv2.imshow('frame',hsv)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()







def cameraHarris(event):
    cap = cv2.VideoCapture(0)

    while(True):
    # Capture frame-by-frame
        ret, frame = cap.read()

    # Our operations on the frame come here
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = np.float32(img)
        dst = cv2.cornerHarris(gray,2,3,0.04)

        #result is dilated for marking the corners, not important
        dst = cv2.dilate(dst,None)

        # Threshold for an optimal value, it may vary depending on the image.
        frame[dst>0.01*dst.max()]=[255]

    # Display the resulting frame
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()




def rgb2gray():
    close()
    global res
    img = cv2.cvtColor(ori, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Rgb2Grayscale',img)
    cv2.imshow('Original image',ori)
    res = img




def SaltPepper():
        close()
        global res
        # row,col,ch = ori.shape
        # s_vs_p = 0.5
        # amount = 0.004
        # out = np.copy(ori)
        # # Salt mode
        # num_salt = np.ceil(amount * out.size * s_vs_p)
        # coords = [np.random.randint(0, i - 1, int(num_salt))
        #     for i in a.shape]
        # out[coords] = 1
        #
        # # Pepper mode
        # num_pepper = np.ceil(amount* out.size * (1. - s_vs_p))
        # coords = [np.random.randint(0, i - 1, int(num_pepper))
        #       for i in a.shape]
        # out[coords] = 0

        image = ori
        row,col,ch = ori.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(ori)
        # Salt mode
        num_salt = np.ceil(amount * ori.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in ori.shape]
        out[coords] = 1

      # Pepper mode
        num_pepper = np.ceil(amount* ori.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in ori.shape]
        out[coords] = 0

        cv2.imshow('Salt & pepper',out)
        cv2.imshow('original',image)
        res = out



def add_logo_bottom_right():
    close()

    filename = filedialog.askopenfilename()
    img = cv2.imread(filename)


    limage = Image.open(filename)
    mimage = Image.open(filename1)


    # resize logo
    wsize = int(min(mimage.size[0], mimage.size[1]) * 0.25)
    wpercent = (wsize / float(limage.size[0]))
    hsize = int((float(limage.size[1]) * float(wpercent)))

    simage = limage.resize((wsize, hsize))
    mbox = mimage.getbbox()
    sbox = simage.getbbox()

    # right bottom corner
    box = (mbox[2] - sbox[2], mbox[3] - sbox[3])
    mimage.paste(simage, box)
    mimage.show()


def add_logo_bottom_left():
    close()
    filename = filedialog.askopenfilename()
    img = cv2.imread(filename)


    limage = Image.open(filename)
    mimage = Image.open(filename1)


    # resize logo
    wsize = int(min(mimage.size[0], mimage.size[1]) * 0.25)
    wpercent = (wsize / float(limage.size[0]))
    hsize = int((float(limage.size[1]) * float(wpercent)))

    simage = limage.resize((wsize, hsize))
    mbox = mimage.getbbox()
    sbox = simage.getbbox()


    box = (mbox[1] - sbox[1], mbox[3] - sbox[3])
    mimage.paste(simage, box)
    mimage.show()

def add_logo_top_right():
    close()
    filename = filedialog.askopenfilename()
    img = cv2.imread(filename)


    limage = Image.open(filename)
    mimage = Image.open(filename1)


    # resize logo
    wsize = int(min(mimage.size[0], mimage.size[1]) * 0.25)
    wpercent = (wsize / float(limage.size[0]))
    hsize = int((float(limage.size[1]) * float(wpercent)))

    simage = limage.resize((wsize, hsize))
    mbox = mimage.getbbox()
    sbox = simage.getbbox()

    # right bottom corner
    box = (mbox[2] - sbox[2], mbox[1] - sbox[1])
    mimage.paste(simage, box)
    mimage.show()

def add_logo_top_left():
    close()
    filename = filedialog.askopenfilename()
    img = cv2.imread(filename)


    limage = Image.open(filename)
    mimage = Image.open(filename1)

    # resize logo
    wsize = int(min(mimage.size[0], mimage.size[1]) * 0.25)
    wpercent = (wsize / float(limage.size[0]))
    hsize = int((float(limage.size[1]) * float(wpercent)))

    simage = limage.resize((wsize, hsize))
    mbox = mimage.getbbox()
    sbox = simage.getbbox()

    # right bottom corner
    box = (mbox[1] - sbox[1], mbox[1] - sbox[1])
    mimage.paste(simage, box)
    mimage.show()


def ShowColorSpaces():
        close()
        global res
        global res1
        global res2

        hsv = cv2.cvtColor(ori, cv2.COLOR_BGR2HSV)
        ycb = cv2.cvtColor(ori, cv2.COLOR_BGR2YCrCb)
        lab = cv2.cvtColor( ori, cv2.COLOR_BGR2LAB)
        cv2.imshow("Original Image", ori)
        cv2.imshow("hsv image", hsv)
        cv2.imshow("ycb image", ycb)
        cv2.imshow("lab image", lab)
        res = hsv
        res1 = ycb
        res2 = lab


def Histo():
        close()
        plt.hist(ori.ravel(),256,[0,256]);
        plt.savefig("histogram.png")
        plt.cla()
        x = cv2.imread("histogram.png")
        cv2.imshow("Histogram",x)
        res = x





def Equalization():
        close()
        hist = cv2.calcHist([ori],[0],None,[256],[0,256])

        color = ('b','g','r')
        for i,col in enumerate(color):
            hist = cv2.calcHist([ori],[i],None,[256],[0,256])

        cdf = hist.cumsum()
        cdf_normalized = cdf * hist.max()/ cdf.max()

        plt.plot(hist,color = col)
        plt.xlim([0,256])

        plt.plot(cdf_normalized, color = 'b')
        plt.legend(('cdf','histogram'), loc = 'upper right')
        plt.savefig("histogram_normalized.png")
        plt.cla()
        x = cv2.imread("histogram_normalized.png")
        cv2.imshow("Histogram",x)
        res = x
def Erosion():
        close()
        img = cv2.cvtColor(ori, cv2.COLOR_BGR2GRAY)
        kernel = np.ones((5,5),np.uint8)
        erosion = cv2.erode(img,kernel,iterations = 1)
        cv2.imshow('image', img)
        cv2.imshow('erosion', erosion)
#         cv2.imwrite("result_erosion.jpg", erosion)
        res = erosion
def Dilation():
        close()
        img = cv2.cvtColor(ori, cv2.COLOR_BGR2GRAY)
        kernel = np.ones((5,5),np.uint8)
        dilation = cv2.dilate(img,kernel,iterations = 1)
        cv2.imshow('image', img)
        cv2.imshow('dilation', dilation)

        res =  dilation
def Open():
        close()
        img = cv2.cvtColor(ori, cv2.COLOR_BGR2GRAY)
        kernel = np.ones((5,5),np.uint8)
        opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        cv2.imshow('image', img)
        cv2.imshow('openingn', opening)
#         cv2.imwrite("result_open.jpg", opening)
        res = opening
def Closei():
        close()
        img = cv2.cvtColor(ori, cv2.COLOR_BGR2GRAY)
        kernel = np.ones((5,5),np.uint8)
        closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        cv2.imshow('original image', Ori)
        cv2.imshow('closing', closing)
#         cv2.imwrite("result_close.jpg", closing)
        res=closing

def Sobel ():
        close()
        img = cv2.cvtColor(ori, cv2.COLOR_BGR2GRAY)

        sobel = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
        cv2.imshow('original-image',ori)
        cv2.imshow('sobel',sobel)
        res = sobel

def Log ():
        close()
        img = cv2.cvtColor(ori, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(img,cv2.CV_64F)
        cv2.imshow('original-image',ori)
        cv2.imshow('log',laplacian)
        res = laplacian


def Canny ():
        close()
        img = cv2.cvtColor(ori, cv2.COLOR_BGR2GRAY)

        edges = cv2.Canny(img,100,200)

        cv2.imshow('original-image', ori)
        cv2.imshow('canny', edges)
        res = edges
def Hough ():
        close()
        img = cv2.cvtColor(ori, cv2.COLOR_BGR2GRAY)

        img = cv2.medianBlur(img,5)

        cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

        circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,20,
                                 param1=50,param2=30,minRadius=5,maxRadius=70)

        circles = np.uint16(np.around(circles))
        for i in circles[0,:]:
            # draw the outer circle
            cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
            # draw the center of the circle
            cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)

        cv2.imshow('image', ori)
        cv2.imshow('cimg', cimg)
        res = cimg
def BoundingBox ():
        close()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret,thresh = cv2.threshold(img,127,255,0)
        contours = cv2.findContours(thresh, 1, 2)

        cnt = contours[0]
        x,y,w,h = cv2.boundingRect(cnt)
        rect = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

        cv2.imshow('rect', rect)
#         cv2.imwrite("result_bb.jpg", rect)
        res = rect
def CentroidFinderBinaryImage():

        close()
        image = ori
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         thresh = 127
#         im_bw = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)[1]

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]
        for c in cnts:

            M = cv2.moments(c)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            # draw the contour and center of the shape on the image
            cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
            cv2.circle(image, (cX, cY), 7, (255, 255, 255), -1)
            cv2.putText(image, "center", (cX - 20, cY - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            cv2.imshow("Centeriods Found", image)


        res = image


def Harris():
        close()
        img = cv2.cvtColor(ori, cv2.COLOR_BGR2GRAY)
        gray = np.float32(img)
        dst = cv2.cornerHarris(gray,2,3,0.04)

        #result is dilated for marking the corners, not important
        dst = cv2.dilate(dst,None)

        # Threshold for an optimal value, it may vary depending on the image.
        img[dst>0.01*dst.max()]=[255]

        cv2.imshow('original image',ori)
        cv2.imshow('Harris Corners',img)
#         cv2.imwrite("result_harris.jpg", img)
        res = img
def SIFT():
        close()
        img1 = cv2.cvtColor(ori, cv2.COLOR_BGR2GRAY)
        filename2 = filedialog.askopenfilename()
        img2 = cv2.imread(filename2)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        sift = cv2.xfeatures2d.SIFT_create()

        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(img1,None)
        kp2, des2 = sift.detectAndCompute(img2,None)

        # BFMatcher with default params
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1,des2, k=2)

        # Apply ratio test
        good = []
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                good.append([m])

        # cv2.drawMatchesKnn expects list of lists as matches.
        img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2)

        cv2.imshow('Matching points',img3)
#         cv2.imwrite("result_sift.jpg",img3)
        res = img3
def Calibration():
        close()
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((6*7,3), np.float32)
        objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.

        images = glob.glob('calib_radial.jpg')

        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (7,6),None)

            # If found, add object points, image points (after refining them)
            if ret == True:
                objpoints.append(objp)

                corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
                imgpoints.append(corners2)

                # Draw and display the corners
                img = cv2.drawChessboardCorners(img, (7,6), corners2,ret)
                cv2.imshow('img',img)

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)###HERE Problem

        #http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html#calibration
        filename = filedialog.askopenfilename()
        img = cv2.imread(filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        h,  w = img.shape[:2]
        newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

        # undistort
        dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

        # crop the image
        x,y,w,h = roi
        dst = dst[y:y+h, x:x+w]
        cv2.imshow('img',img)
        cv2.imshow('results',dst)
#         cv2.imwrite("result_calibration.jpg", dst)
        res = dst

def Epipolar():
        close()
        img = cv2.cvtColor(ori, cv2.COLOR_BGR2GRAY)
        filename2 = filedialog.askopenfilename()
        img2 = cv2.imread(filename2)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        img1 = cv2.resize(img, (0,0), fx=0.2, fy=0.2)
        img2 = cv2.resize(img2, (0,0), fx=0.2, fy=0.2)

        sift = cv2.xfeatures2d.SIFT_create()

        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(img1,None)
        kp2, des2 = sift.detectAndCompute(img2,None)

        # FLANN parameters
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)

        flann = cv2.FlannBasedMatcher(index_params,search_params)
        matches = flann.knnMatch(des1,des2,k=2)

        good = []
        pts1 = []
        pts2 = []

        # ratio test as per Lowe's paper
        for i,(m,n) in enumerate(matches):
            if m.distance < 0.8*n.distance:
                good.append(m)
                pts2.append(kp2[m.trainIdx].pt)
                pts1.append(kp1[m.queryIdx].pt)

        pts1 = np.int32(pts1)
        pts2 = np.int32(pts2)
        F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)

        # We select only inlier points
        pts1 = pts1[mask.ravel()==1]
        pts2 = pts2[mask.ravel()==1]


        def drawlines(img1,img2,lines,pts1,pts2):
            ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
            r,c = img1.shape
            img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
            img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
            for r,pt1,pt2 in zip(lines,pts1,pts2):
                color = tuple(np.random.randint(0,255,3).tolist())
                x0,y0 = map(int, [0, -r[2]/r[1] ])
                x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
                img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
                img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
                img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
            return img1,img2

        # Find epilines corresponding to points in right image (second image) and
        # drawing its lines on left image
        lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
        lines1 = lines1.reshape(-1,3)
        img5,img6 = drawlines(img1,img2,lines1,pts1,pts2)

        # Find epilines corresponding to points in left image (first image) and
        # drawing its lines on right image
        lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
        lines2 = lines2.reshape(-1,3)
        img3,img4 = drawlines(img2,img1,lines2,pts2,pts1)

        cv2.imshow('img5', img5)
        cv2.imshow('img3', img3)
#         cv2.imwrite("result_epipolar.jpg", img3)
        res = img3

def Homography():
        close()
        MIN_MATCH_COUNT = 5


        img1 = cv2.cvtColor(ori, cv2.COLOR_BGR2GRAY)
        filename2 = filedialog.askopenfilename()
        img2 = cv2.imread(filename2)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # Initiate SIFT detector
        sift = cv2.xfeatures2d.SIFT_create()

        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(img1,None)
        kp2, des2 = sift.detectAndCompute(img2,None)

        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)

        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(des1,des2,k=2)

        # store all the good matches as per Lowe's ratio test.
        good = []
        for m,n in matches:
            if m.distance < 0.7*n.distance:
                good.append(m)


        if len(good)>MIN_MATCH_COUNT:
            src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
            matchesMask = mask.ravel().tolist()

            h,w = img1.shape
            pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
            dst = cv2.perspectiveTransform(pts,M)

            img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

        else:
            print ("Not enough matches are found - %d/%d"  (len(good),MIN_MATCH_COUNT))
            matchesMask = None

        draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

        img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)

        cv2.imshow('img3', img3)
#         cv2.imwrite('homography.jpg', img3)
        res = img3

def Stitching ():

        close()
        filename = filedialog.askopenfilename()
        img = cv2.imread(filename)

        stitcher = cv2.createStitcher(False)


        result = stitcher.stitch((ori,img))

#         cv2.imwrite("stitch.png", result[1])
        cv2.imshow("Stitched",result[1])
        res = result[1]




def FaceDetection():

        close()
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

        gray = cv2.cvtColor(ori, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x,y,w,h) in faces:
            cv2.rectangle(ori,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.imshow('detected face',ori)
        res = ori

def close():
    cv2.destroyAllWindows()
    app.title("CV")


def save():
    close()
    global co
    cv2.imwrite('CurrentOutputWindow'+str(co)+'.png',res)
    co=co+1



# Create data matrix from a list of images
# def createDataMatrix(images):
#     print("Creating data matrix",end=" ... ")
#     '''
#     Allocate space for all images in one data matrix.
#   The size of the data matrix is

#   ( w  * h  * 3, numImages )

#   where,

#   w = width of an image in the dataset.
#   h = height of an image in the dataset.
#   3 is for the 3 color channels.
#   '''

#     numImages = len(images)
#     sz = images[0].shape
#     data = np.zeros((numImages, sz[0] * sz[1] * sz[2]), dtype=np.float32)
#     for i in xrange(0, numImages):
#         image = images[i].flatten()
#         data[i,:] = image

#     print("DONE")
#     return data

# # Read images from the directory
# def readImages(path):
#       print("Reading images from " + path, end="...")
#   # Create array of array of images.
#       images = []
#   # List all files in the directory and read points from text files one by one
#       for filePath in sorted(os.listdir(path)):
#             fileExt = os.path.splitext(filePath)[1]
#             if fileExt in [".jpg", ".jpeg"]:

#       # Add to array of images
#               imagePath = os.path.join(path, filePath)
#               im = cv2.imread(imagePath)

#             if im is None :
#                 print("image:{} not read properly".format(imagePath))
#             else :
#           # Convert image to floating point
#                 im = np.float32(im)/255.0
#            # Add image to list
#                 images.append(im)
#            # Flip image
#                 imFlip = cv2.flip(im, 1);
#           # Append flipped image
#                 images.append(imFlip)

#             numImages = len(images) / 2
#   # Exit if no image found
#       if numImages == 0 :
#         print("No images found")
#         sys.exit(0)

# print(str(numImages) + " files read.")
# return images

# # Add the weighted eigen faces to the mean face
# def createNewFace(*args):
#     # Start with the mean image
#     output = averageFace

#     # Add the eigen faces with the weights
#     for i in xrange(0, NUM_EIGEN_FACES):

#         sliderValues[i] = cv2.getTrackbarPos("Weight" + str(i), "Trackbars");
#         weight = sliderValues[i] - MAX_SLIDER_VALUE/2
#         output = np.add(output, eigenFaces[i] * weight)

#     # Display Result at 2x size
#     output = cv2.resize(output, (0,0), fx=2, fy=2)
#     cv2.imshow("Result", output)

# def resetSliderValues(*args):
#     for i in xrange(0, NUM_EIGEN_FACES):
#         cv2.setTrackbarPos("Weight" + str(i), "Trackbars", MAX_SLIDER_VALUE/2);
#     createNewFace()

# def runeigenface():
#     # Number of EigenFaces
#     NUM_EIGEN_FACES = 10

#     # Maximum weight
#     MAX_SLIDER_VALUE = 255

#     # Directory containing images
#     dirName = "images"

#     # Read images
#     images = readImages(dirName)

#     # Size of images
#     sz = images[0].shape

#     # Create data matrix for PCA.
#     data = createDataMatrix(images)

#     # Compute the eigenvectors from the stack of images created
#     print("Calculating PCA ", end="...")
#     mean, eigenVectors = cv2.PCACompute(data, mean=None, maxComponents=NUM_EIGEN_FACES)
#     print ("DONE")

#     averageFace = mean.reshape(sz)

#     eigenFaces = [];

#     for eigenVector in eigenVectors:
#         eigenFace = eigenVector.reshape(sz)
#         eigenFaces.append(eigenFace)

#     # Display result at 2x size
#     output = cv2.resize(averageFace, (0,0), fx=2, fy=2)
#     cv2.imshow("Result", output)

'''
    Copyright 2017 by Satya Mallick ( Big Vision LLC )
    http://www.learnopencv.com
'''

def fillHoles(mask):
    '''
        This hole filling algorithm is decribed in this post
        https://www.learnopencv.com/filling-holes-in-an-image-using-opencv-python-c/
    '''
    maskFloodfill = mask.copy()
    h, w = maskFloodfill.shape[:2]
    maskTemp = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(maskFloodfill, maskTemp, (0, 0), 255)
    mask2 = cv2.bitwise_not(maskFloodfill)
    return mask2 | mask


# def cameraRedeyeRemover():
#      # Read image
#
#
#     img = cv2.imread("red_eyes.jpg", cv2.IMREAD_COLOR)
#
#     # Output image
#     imgOut = img.copy()
#
#     # Load HAAR cascade
#     eyesCascade = cv2.CascadeClassifier("haarcascade_eye.xml")
#
#     # Detect eyes
#     eyes = eyesCascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(100, 100))
#
#     # For every detected eye
#     for (x, y, w, h) in eyes:
#
#         # Extract eye from the image
#         eye = img[y:y+h, x:x+w]
#
#         # Split eye image into 3 channels
#         b = eye[:, :, 0]
#         g = eye[:, :, 1]
#         r = eye[:, :, 2]
#
#         # Add the green and blue channels.
#         bg = cv2.add(b, g)
#
#         # Simple red eye detector.
#         mask = (r > 150) &  (r > bg)
#
#         # Convert the mask to uint8 format.
#         mask = mask.astype(np.uint8)*255
#
#         # Clean mask -- 1) File holes 2) Dilate (expand) mask.
#         mask = fillHoles(mask)
#         mask = cv2.dilate(mask, None, anchor=(-1, -1), iterations=3, borderType=1, borderValue=1)
#
#         # Calculate the mean channel by averaging
#         # the green and blue channels
#         mean = bg / 2
#         mask = mask.astype(np.bool)[:, :, np.newaxis]
#         mean = mean[:, :, np.newaxis]
#
#         # Copy the eye from the original image.
#         eyeOut = eye.copy()
#
#         # Copy the mean image to the output image.
#         #np.copyto(eyeOut, mean, where=mask)
#         eyeOut = np.where(mask, mean, eyeOut)
#
#         # Copy the fixed eye to the output image.
#         imgOut[y:y+h, x:x+w, :] = eyeOut
#
#     # Display Result
#     cv2.imshow('Red Eyes', img)
#     cv2.imshow('Red Eyes Removed', imgOut)
#     cv2.waitKey(0)



def facerecognitionPCA(event):

        close()
        recognizer = cv2.face.EigenFaceRecognizer_create()
        gray = cv2.cvtColor(ori, cv2.COLOR_BGR2GRAY)
        trainfaces, Ids = getImagesAndLabels("./images")

        recognizer.train(trainfaces, np.array(Ids))
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")





        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x,y,w,h) in faces:
            roi_gray = gray[y:y+10 + h+10, x:x+10 + w+10]


        resized_image = cv2.resize(roi_gray, (64, 64))

        label = recognizer.predict(resized_image)

        pred_img = trainfaces[label[0]-1,:,:]

        cv2.imshow('Nearest Face To',pred_img)
        cv2.imshow('Original Face',ori)
        res = pred_img








def getImagesAndLabels(path):

        dirs = os.listdir(path)
        dirs = sorted(dirs)

        labels = []
        images = []

        #  go through each directory and read images within it

        for label in dirs:

            labels.append(int(label.strip('./jpg')))

            subject_dir_path = path + "/" + label

            image = cv2.imread(subject_dir_path)

            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image, (64, 64))
            images.append(image)


        return np.array(images), labels



















#GUI interface
app = Tk()
app.title("openCv Library")
app.geometry('1000x1000')

hell16 = font.Font(family='Helvetica', size=18, weight='bold')
hell166 = font.Font(family='Helvetica', size=16, weight='bold')
font.families()

menubar = Menu(app)

file = Menu(menubar,tearoff=0)
file.add_command(label="Load-Image",command=uploadimage1)
file.add_command(label="Save current O",command=save)
file.add_command(label="Good-bye",command=app.destroy)
menubar.add_cascade(label="File", menu=file)


basicbar = Menu(menubar,tearoff=0)
basicbar.add_command(label="Histogram",command=Histo)
basicbar.add_command(label="Histo Equalization",command=Equalization)
basicbar.add_command(label="rgb2gray",command=rgb2gray)
basicbar.add_command(label="Color Spaces",command=ShowColorSpaces)
basicbar.add_command(label="Salt & Pepper noise",command=SaltPepper)
menubar.add_cascade(label="Basic", menu=basicbar)


logo = Menu(menubar,tearoff=0)
logo.add_command(label="Top Right",command=add_logo_top_right)
logo.add_command(label="Top left",command= add_logo_top_left)
logo.add_command(label="bottom right",command=add_logo_bottom_right)
logo.add_command(label="bottom left",command=add_logo_bottom_left)
basicbar.add_cascade(label="Add logo", menu=logo)



morphobar = Menu(menubar,tearoff=0)

morphobar.add_command(label="Dilate",command=Dilation)
morphobar.add_command(label="Erode",command=Erosion)
morphobar.add_command(label="Open",command=Open)
morphobar.add_command(label="Close",command=Closei)
menubar.add_cascade(label="Morph", menu=morphobar)


edgebar = Menu(menubar,tearoff=0)
edgebar.add_command(label="Sobel",command=Sobel)
edgebar.add_command(label="Log",command=Log)
edgebar.add_command(label="Canny",command=Canny)
menubar.add_cascade(label="Edge Detectors", menu=edgebar)


ShapeAndBoundries= Menu(menubar,tearoff=0)
ShapeAndBoundries.add_command(label="Hough",command=Hough)
ShapeAndBoundries.add_command(label="Bounding Box",command=BoundingBox)
ShapeAndBoundries.add_command(label="binary image Centroid",command=CentroidFinderBinaryImage)
ShapeAndBoundries.add_command(label="Harris",command=Harris)
menubar.add_cascade(label="Shapes and Countours", menu=ShapeAndBoundries)



AdvancedBar = Menu(menubar,tearoff=0)
AdvancedBar.add_command(label="SIFT",command=SIFT)
AdvancedBar.add_command(label="Calibration",command=Calibration)
AdvancedBar.add_command(label="Epipolar",command=Epipolar)
AdvancedBar.add_command(label="Homography",command=Homography)
AdvancedBar.add_command(label="Stitching",command=Stitching)
AdvancedBar.add_command(label="Face Detection",command=FaceDetection)
menubar.add_cascade(label="Advanced Operations", menu=AdvancedBar)





app.config(menu=menubar)






#Displaying it
b1 = tkinter.Button(app,height=2, width=20,text="Upload image",bg="gray",font=hell16)
b1.pack(padx=40, pady=50)
b1.place(x=350,y=100)
b1.bind("<Button-1>",uploadimage)



b2 = tkinter.Button(app,height=2, width=20,text="CameraFeed in Gray",bg="gray",font=hell16)
b2.pack(padx=40, pady=50)
b2.place(x=350,y=200)
b2.bind("<Button-1>",camera)



b3 = tkinter.Button(app,height=2, width=20,text="CameraFeed in rgb",bg="gray",font=hell16)
b3.pack(padx=40, pady=50)
b3.place(x=350,y=300)
b3.bind("<Button-1>",camerargb)


b4 = tkinter.Button(app,height=2, width=20,text="CameraFeed with erosion",bg="gray",font=hell16)
b4.pack(padx=40, pady=50)
b4.place(x=350,y=400)
b4.bind("<Button-1>",cameraErosion)

b5 = tkinter.Button(app,height=2, width=20,text="CameraFeed with sobel",bg="gray",font=hell16)
b5.pack(padx=40, pady=50)
b5.place(x=350,y=500)
b5.bind("<Button-1>",cameraSobel)

b6 = tkinter.Button(app,height=2, width=20,text="Camera Face detector",bg="gray",font=hell16)
b6.pack(padx=40, pady=50)
b6.place(x=350,y=600)
b6.bind("<Button-1>",camerafacedetector)

b6 = tkinter.Button(app,height=2, width=20,text="CameraFeed Harris Corner",bg="gray",font=hell16)
b6.pack(padx=40, pady=50)
b6.place(x=350,y=700)
b6.bind("<Button-1>",cameraHarris)

L1 = tkinter.Label(app,height=3, width=28,text="press Q to Quit Camera Feed",bg="red",font=hell166)
L1.pack(padx=40, pady=50)
L1.place(x=330,y=10)


b7 = tkinter.Button(app,height=2, width=20,text="CameraFeed in hsv",bg="gray",font=hell16)
b7.pack(padx=40, pady=50)
b7.place(x=350,y=800)
b7.bind("<Button-1>",cameraHSV)

b8 = tkinter.Button(app,height=2, width=20,text="Face Recognition",bg="gray",font=hell16)
b8.pack(padx=40, pady=50)
b8.place(x=350,y=900)
b8.bind("<Button-1>",facerecognitionPCA)



def main():
  app.mainloop()


if __name__ == "__main__":
    main()



