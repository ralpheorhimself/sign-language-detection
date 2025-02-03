import os
import cv2

DATA_DIR = './data'

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

class_numbers = 26
dataset_size = 100

cap = cv2.VideoCapture(0)
for index in range(class_numbers):
    if not os.path.exists(os.path.join(DATA_DIR,str(index))):
        os.makedirs(os.path.join(DATA_DIR,str(index)))
    
    print('Collecting data for class {}'.format(index))
    
    done = False
    while True:
        success, frame = cap.read()
        cv2.putText(frame,'Press Q to capture:',(100,50),cv2.FONT_HERSHEY_DUPLEX,1.0,(0,255,0),2,cv2.LINE_AA)
        cv2.imshow('frame',frame)
        if cv2.waitKey(25) == ord('q'):
            break
    
    cnt = 0
    while cnt < dataset_size:
        success,frame = cap.read()
        cv2.imshow('frame',frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(DATA_DIR, str(index), '{}.jpg'.format(cnt)), frame)
        
        cnt+=1
cap.release()
cv2.destroyAllWindows()

        
