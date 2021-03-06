from __future__ import print_function
from collections import deque
from imutils.video import VideoStream
from matplotlib import pyplot as plt
import numpy as np
import argparse
import cv2
import imutils
import time

img_counter = 0

faceCascadeFilePath = "haarcascade_frontalface_default.xml"
noseCascadeFilePath = "haarcascade_nose.xml"
smileCascadeFilePath = "haarcascade_smile.xml"

# build our cv2 Cascade Classifiers
faceCascade = cv2.CascadeClassifier(faceCascadeFilePath)
noseCascade = cv2.CascadeClassifier(noseCascadeFilePath)
smileCascade = cv2.CascadeClassifier(smileCascadeFilePath)

imgMustache = cv2.imread('mustache.png', -1)
imgSmiley = cv2.imread('smiley.png', -1)

orig_mask = imgMustache[:, :, 3]
orig_mask1 = imgSmiley[:, :, 2] #/ 255.0

alpha_l = 1.0 - imgSmiley

orig_mask_inv = cv2.bitwise_not(orig_mask)
orig_mask_inv1 = cv2.bitwise_not(orig_mask1)

imgMustache = imgMustache[:, :, 0:3]
imgSmiley = imgSmiley[:, :, 0:3]
origMustacheHeight, origMustacheWidth = imgMustache.shape[:2]
origSmileyHeight, origSmileyWidth = imgSmiley.shape[:2]

# imgSmiley = cv2.cvtColor(imgSmiley, cv2.COLOR_BGR2RGBA, 0, 0)
# imgSmiley[np.all(imgSmiley == [0, 0, 0, 255], axis=2)] = [0, 0, 0, 0]

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
                help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64,
                help="max buffer size")
args = vars(ap.parse_args())

# orange in the HSV color space
orangeLower = (1, 190, 200)
orangeUpper = (18, 255, 255)
pts = deque(maxlen=args["buffer"])

if not args.get("video", False):
    vs = VideoStream(src=0).start()
else:
    vs = cv2.VideoCapture(args["video"])

time.sleep(2.0)


while True:
    frame = vs.read()
    frame = frame[1] if args.get("video", False) else frame
    if frame is None:
        break

    frame = imutils.resize(frame, width=800)
    frame = cv2.flip(frame, 1)
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, orangeLower, orangeUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)


    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # (x, y) center
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    center = None

    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        if radius > 10:
            cv2.circle(frame, (int(x), int(y)), int(radius),
                       (0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)

    pts.appendleft(center)

    for i in range(1, len(pts)):
        if pts[i - 1] is None or pts[i] is None:
            continue

    key = cv2.waitKey(1) & 0xFF

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    x_offset = 550
    y_offset = 10

    frame[y_offset:y_offset + imgSmiley.shape[0], x_offset:x_offset + imgSmiley.shape[1]] = imgSmiley

    cv2.putText(frame, "Widze wasze buzie: {}".format(str(len(faces))), (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    for (x, y, w, h) in faces:
        face = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)

        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        nose = noseCascade.detectMultiScale(roi_gray)

        if x > x_offset & x < x + imgSmiley.shape[0] & y > y_offset & y > y + imgSmiley.shape[1]:

            break

        else:

            for (nx, ny, nw, nh) in nose:
                mustacheWidth = 2 * nw
                mustacheHeight = mustacheWidth * origMustacheHeight / origMustacheWidth

                x1 = nx - (mustacheWidth / 4)
                x2 = nx + nw + (mustacheWidth / 4)
                y1 = ny + nh - (mustacheHeight / 2)
                y2 = ny + nh + (mustacheHeight / 2)

                if x1 < 0:
                    x1 = 0
                if y1 < 0:
                    y1 = 0
                if x2 > w:
                    x2 = w
                if y2 > h:
                    y2 = h

                mustacheWidth = x2 - x1
                mustacheHeight = y2 - y1

                mustache = cv2.resize(imgMustache, (mustacheWidth, mustacheHeight), interpolation=cv2.INTER_AREA)
                mask = cv2.resize(orig_mask, (mustacheWidth, mustacheHeight), interpolation=cv2.INTER_AREA)
                mask_inv = cv2.resize(orig_mask_inv, (mustacheWidth, mustacheHeight), interpolation=cv2.INTER_AREA)

                roi = roi_color[y1:y2, x1:x2]

                roi_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

                roi_fg = cv2.bitwise_and(mustache, mustache, mask=mask)

                dst = cv2.add(roi_bg, roi_fg)

                roi_color[y1:y2, x1:x2] = dst

                break

    cv2.imshow('Video', frame)

    if key%256 == 27:
        print("Escape hit, closing...")
        break
    elif key%256 == 32:
        img_name = "selfie_{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1

    if key == ord("q"):
        break

if not args.get("video", False):
    vs.stop()

else:
    vs.release()

cv2.destroyAllWindows()
