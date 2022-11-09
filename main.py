import cv2
import numpy as np
import HandTrackingModule as htm
import mouse
import tkinter as tk

##########################
wCam, hCam = 640, 480
frameR = 100  # Frame Reduction
smoothening = 6
#########################

clocX, clocY = 0, 0

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
detector = htm.handDetector(maxHands=1)
root = tk.Tk()

scrW = root.winfo_screenwidth()
scrH = root.winfo_screenheight()

leftClicked = False
leftPress = False
doubleLeftClicked = False
rightClicked = False
wheel = False

handPositionX = 0
handPositionY = 0

mousePositionX = 0
mousePositionY = 0

scrollStartX = 0
scrollStartY = 0

delta = 0
run = True
turnOffCounter = 5

while run:
    # 1. Preprocess
    success, img = cap.read()

    # 1.2 Image sharpening
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    img = cv2.filter2D(src=img, ddepth=-1, kernel=kernel)

    # 2. Find hand Landmarks
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)

    if lmList != None:
        # 3. Get the tip of the index and middle fingers
        if len(lmList) != 0:
            pointingFingerBaseX, pointingFingerBaseY = lmList[5][1:]
            pinkyFingerBaseX, pinkyFingerBaseY = lmList[17][1:]

            handPositionX = int((pointingFingerBaseX + pinkyFingerBaseX) / 2)
            handPositionY = int((pointingFingerBaseY + pinkyFingerBaseY) / 2)

            # 5. Convert Coordinates
            mousePositionX = np.interp(handPositionX, (frameR, wCam - frameR), (0, scrW))
            mousePositionY = np.interp(handPositionY, (frameR, hCam - frameR), (0, scrH))

        # 4. Check which fingers are up
        try:
            fingers = detector.fingersUp()
        except:
                fingers = []

        cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 255), 2)
        if len(fingers) != 0:
            # 4.1 (index middle ring and pinky up)) : Moving Mode
            if fingers[2] == 1 and fingers[3] == 1 and fingers[4] == 1:
                # 4.1.1 set clicks to false
                doubleLeftClicked = False
                rightClicked = False
                wheel = False

                # 4.1.2 Smoothen Values
                clocX = clocX + (mousePositionX - clocX) / smoothening
                clocY = clocY + (mousePositionY - clocY) / smoothening

                # 4.1.3 Move Mouse
                mouse.move(scrW - clocX, clocY)
                cv2.circle(img, (handPositionX, handPositionY), 10, (0, 0, 255), cv2.FILLED)

            # 4.2 : Index Left press
            if fingers[1] == 0 and fingers[2] == 1 and fingers[3] == 1 and fingers[4] == 1:
                if not leftPress:
                    mouse.press('left')
                    leftPress = True
                    print("left press")

            if fingers[1] == 1:
                if leftPress:
                    mouse.release('left')
                    leftPress = False
                    print("left release")

            # 4.2 : Index, Middle, Ring and Little fingers: Double Left Click
            if fingers[1] == 0 and fingers[2] == 0 and fingers[3] == 1 and fingers[4] == 1:
                if not doubleLeftClicked and not leftPress:
                    mouse.double_click()
                    doubleLeftClicked = True
                    print("double left click")

            # 4.3 : Middle finger down: Single Right Click
            if fingers[1] == 1 and fingers[2] == 0 and fingers[3] == 1 and fingers[4] == 1:
                if not rightClicked and not leftPress:
                    mouse.right_click()
                    rightClicked = True
                    print("right click")

            # 4.4 Wheel Mode
            if fingers[1] == 0 and fingers[2] == 0 and fingers[3] == 0 and fingers[4] == 0:
                if not wheel and not leftPress:
                    scrollStartX = mousePositionX
                    scrollStartY = mousePositionY
                    wheel = True
                    print("wheel")
                delta = scrollStartY-mousePositionY
                if abs(delta) > 50:
                    mouse.wheel(delta/200)

            # 5 Stop
            if fingers[1] == 0 and fingers[2] == 1 and fingers[3] == 0 and fingers[4] == 0:
                turnOffCounter += 1
                if turnOffCounter > 20:
                    print("fuck")
                    run = False
            else:
                turnOffCounter = 0

    # 5. Display
    flip = cv2.flip(img, 1)
    cv2.imshow("Detected", flip)
    cv2.waitKey(1)
