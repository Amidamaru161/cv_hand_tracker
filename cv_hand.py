from inspect import trace
import cv2
import mediapipe as mp
import numpy as np

hsv_min = np.array((53, 0, 0), np.uint8)
hsv_max = np.array((83, 255, 255), np.uint8)

cap = cv2.VideoCapture(0)
hands = mp.solutions.hands.Hands(max_num_hands=2)
draw = mp.solutions.drawing_utils

while cap.isOpened():
    success, image = cap.read()
    image = cv2.flip(image, 180)
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    thresh = cv2.inRange(imageRGB, hsv_min, hsv_max)
    results = hands.process(imageRGB)
    
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, c = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)

            draw.draw_landmarks(image, handLms, mp.solutions.hands.HAND_CONNECTIONS)

    cv2.imshow("Hand", image)
    cv2.imshow("--", thresh)

    if cv2.waitKey(1) & 0xFF == 27:
        cap.release()
