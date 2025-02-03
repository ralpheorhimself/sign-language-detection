import pickle

import cv2
import mediapipe as mp
import numpy as np

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 
10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 
19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}

while True:

    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hlm in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame,hlm,mp_hands.HAND_CONNECTIONS)

        for hlm in results.multi_hand_landmarks:
            for i in range(len(hlm.landmark)):
                x = hlm.landmark[i].x
                y = hlm.landmark[i].y

                x_.append(x)
                y_.append(y)

            for i in range(len(hlm.landmark)):
                x = hlm.landmark[i].x
                y = hlm.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10

        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        prediction = model.predict([np.asarray(data_aux)])

        predicted_character = labels_dict[int(prediction[0])]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3,cv2.LINE_AA)
    cv2.putText(frame,'Press Q to quit:',(100,50),cv2.FONT_HERSHEY_DUPLEX,1.0,(0,255,0),2,cv2.LINE_AA)
    if cv2.waitKey(25) == ord('q'):
            break
    cv2.imshow('fingering-detector', frame)
    cv2.waitKey(1)


cap.release()
cv2.destroyAllWindows()
