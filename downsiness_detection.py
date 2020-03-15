import cv2 
from pygame import mixer
import numpy as np 
from keras.models import load_model
import time 

def load_haarcascade(face_path, eyes_path):
    face_cascade = cv2.CascadeClassifier(face_path)
    eyes_cascade = cv2.CascadeClassifier(eyes_path)
    return face_cascade, eyes_cascade

def face_detector(gray):
    faces = face_cascade.detectMultiScale(gray)
    return faces

def eye_classification(img):
    img = cv2.resize(img, (50, 50))
    img = img.reshape(-1, 50, 50, 1)/255.0
    pred = model.predict(img)
    return 1 if pred[0][0] > 0.5 else 0

def eyes_detector(frame, face, gray_face):
    eyes = eyes_cascade.detectMultiScale(gray_face)
    eyes = sorted(eyes, key=keysort, reverse=True)
    # for (x, y, w, h) in eyes:
    #     cv2.rectangle(face, (x, y), (x+w, y+h), (255, 0, 0), 2)
    #     cv2.imshow('Detected faces.', frame)
    if len(eyes) == 1:
        eyes.append(eyes[-1])
    if len(eyes) >= 2:
        eyes = eyes[:2]
        x, y, w, h = eyes[0]
        leye_img = np.zeros((x+w, y+h))
        leye_img = gray_face[y:y+int(3/4*h), x:x+w]
        x, y, w, h = eyes[1]
        reye_img = np.zeros((x+w, y+h))
        reye_img = gray_face[y:y+int(3/4*h), x:x+w]
        return eye_classification(leye_img), eye_classification(reye_img)
    return None
    
def keysort(elem):
    return elem[2]*elem[3]

# load model predict eyes open or close
model = load_model('./model_3.h5')

cap = cv2.VideoCapture(0)
four_cc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('out.mp4', four_cc, 5.0, (640, 480))

mixer.init()
sound = mixer.Sound('alarm.wav')

face_path = './haar cascade files/haarcascade_frontalface_alt.xml'
eyes_path = './haar cascade files/haarcascade_eye.xml'
face_cascade, eyes_cascade = load_haarcascade(face_path, eyes_path)

SCORE = 0
max_thick = 16

while True:
    # 23 frames per second only read image, 5 frames per second to process
    _, frame = cap.read()
    cv2.imshow('camera', frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray)
    if len(faces)>0:
        sorted_faces = sorted(faces, key=keysort, reverse=True)
        main_faces = sorted_faces[0]
        x, y, w, h = main_faces
        # face = np.zeros((x+w, y+h))
        gray_face = gray[y:y+h, x:x+w]
        face = frame[y:y+h, x:x+w]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 0), 2)
        eyes_value = eyes_detector(frame, face, gray_face)
        if eyes_value is not None:
            leye, reye = eyes_value
            if leye==1 and reye == 1:
                text = 'OPEN'
                SCORE -= 1 
            else:
                text = 'CLOSE'
                SCORE += 1
        
            if SCORE < 0: 
                SCORE = 0 
            elif SCORE > 10:
                thick = max(max_thick, SCORE-9)
                sound.play()
                cv2.rectangle(frame, (0, 0), (640, 480), (0, 0, 255), thick)
            font = cv2.FONT_HERSHEY_COMPLEX_SMALL 
            cv2.putText(frame, text, (10, 420), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
            score_text = 'SCORE : {}'.format(SCORE)
            cv2.putText(frame, score_text, (120, 420), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
    out.write(frame)
    cv2.imshow('camera', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
