import cv2
import numpy as np
import HandTrackingModule as htm
import mouse
import tkinter as tk

##########################
wCam, hCam = 640, 480
frameR = 100  # Frame Reduction
smoothening = 7
#########################

pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
detector = htm.handDetector(maxHands=1)
root = tk.Tk()

scrW = root.winfo_screenwidth()
scrH = root.winfo_screenheight()

clicked = False
dclicked = False
rclicked = False

while True:
    #1. Preprocess
    success, img_o = cap.read()
    # 1.1 Contrast stetching with CLACHE
    # img_o = cv2.normalize(img_o, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # img_hsv = cv2.cvtColor(img_o, cv2.COLOR_BGR2HSV)
    # h, s, v = img_hsv[:, :, 0], img_hsv[:, :, 1], img_hsv[:, :, 2]
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    # v = clahe.apply(v)
    # hsv_img = np.dstack((h, s, v))
    # bgr_stretched = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)

    # 1.2 Image sharpening
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    image_sharp = cv2.filter2D(src=img_o, ddepth=-1, kernel=kernel)

    # 2. Find hand Landmarks
    img = detector.findHands(image_sharp)
    lmList, bbox = detector.findPosition(img)
    if lmList != None:
        # 3. Get the tip of the index and middle fingers
        if len(lmList) != 0:
            x1, y1 = lmList[8][1:]
            x2, y2 = lmList[12][1:]

        # 4. Check which fingers are up
        try:
            fingers = detector.fingersUp()
        except:
            # flip = cv2.flip(img_o, 1)
            # cv2.imshow("Image", flip)
            continue

        cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR),
                      (255, 0, 255), 2)
        # 4.1 Only Index Finger : Moving Mode
        if fingers[1] == 1 and fingers[2] == 0 and fingers[3] == 0 and fingers[4] == 0:
            clicked = False
            dclicked = False
            rclicked = False
            # 5. Convert Coordinates
            x3 = np.interp(x1, (frameR, wCam - frameR), (0, scrW))
            y3 = np.interp(y1, (frameR, hCam - frameR), (0, scrH))
            # 6. Smoothen Values
            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening

            # 7. Move Mouse
            mouse.move(scrW - clocX, clocY)
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            delta = y3 - plocY
            plocX, plocY = clocX, clocY

        # 4.2 : Index and Middle fingers: Single Left Click
        if fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 0 and fingers[4] == 0:
            print("left click")
            if not clicked:
                clicked = True
                mouse.click()

        # 4.2 : Index, Middle, Ring and Little fingers: Double Left Click
        if fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 1 and fingers[4] == 1:
            print("dleft click")
            if not dclicked:
                dclicked = True
                mouse.double_click()

        # 4.3 : Index and Little fingers: Single Right Click
        if fingers[1] == 1 and fingers[2] == 0 and fingers[3] == 0 and fingers[4] == 1:
            print("right click")
            if not rclicked:
                rclicked = True
                mouse.right_click()

        # 4.4 Wheel Mode
        # if fingers[1] == 0 and fingers[0] == 0 and fingers[2] == 0 and fingers[3] == 0 and fingers[4] == 0:
        #     print("wheel")
        #     if delta < 0:
        #         mouse.wheel(delta=1)
        #     if delta > 0:
        #         mouse.wheel(delta=-1)

    # 5. Display
    flip = cv2.flip(img, 1)
    cv2.imshow("Image", flip)
    cv2.waitKey(1)