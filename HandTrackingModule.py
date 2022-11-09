import cv2
import mediapipe as mp
import time
import math
import numpy as np


class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, 0,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]
        self.baseIds = [1, 5, 9, 13, 17]

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)

        retigm = img
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(retigm, handLms,
                                               self.mpHands.HAND_CONNECTIONS)

        return retigm

    def findPosition(self, img, handNo=0, draw=False):
        xList = []
        yList = []
        bbox = []
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
                # print(id, cx, cy)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 4, (255, 0, 255), cv2.FILLED)

            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax

            if draw:
                cv2.rectangle(img, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20),
                              (0, 255, 0), 2)

        return self.lmList, bbox

    def fingersUp(self):
        fingers = []
        # Thumb
        tipToRingBAseDistance = self.findDistance(self.tipIds[0], 0)
        wristToRingBaseDistance = self.findDistance(self.baseIds[3], 0)

        if wristToRingBaseDistance > tipToRingBAseDistance:
            fingers.append(0)
        else:
            fingers.append(1)

        # Fingers
        for id in range(1, 5):

            tipToWristDistance = self.findDistance(self.tipIds[id], 0)
            firstJoinToWristDistance = self.findDistance(self.tipIds[id]-1, 0)

            if tipToWristDistance < firstJoinToWristDistance:
                fingers.append(0)
            else:
                fingers.append(1)

        return fingers

    def findDistance(self, p1, p2):
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]

        length = math.hypot(x2 - x1, y2 - y1)

        return length
