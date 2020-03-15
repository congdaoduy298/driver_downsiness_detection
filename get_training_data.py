import cv2 
import numpy as np 

cap = cv2.VideoCapture(0)

def keysort(elem):
    return elem[2]*elem[3]

def load_haarcascade(face_path, eyes_path):
    face_cascade = cv2.CascadeClassifier(face_path)
    eyes_cascade = cv2.CascadeClassifier(eyes_path)
    return face_cascade, eyes_cascade

def face_detector(gray):
    faces = face_cascade.detectMultiScale(gray)
    return faces   

def eyes_detector(frame, face, gray_face):
    eyes = eyes_cascade.detectMultiScale(gray_face)
    eyes = sorted(eyes, key=keysort, reverse=True)
    global count 
    print(len(eyes))
    for (x, y, w, h) in eyes:
        cv2.rectangle(face, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.imshow('Detected faces.', frame)
    if len(eyes) == 1:
        eyes.append(eyes[-1])
    if len(eyes) >= 2:
        eyes = eyes[:2]
        x, y, w, h = eyes[0]
        leye_img = np.zeros((x+w, y+h))
        leye_img = gray_face[y:y+h, x:x+w]
        x, y, w, h = eyes[1]
        reye_img = np.zeros((x+w, y+h))
        reye_img = gray_face[y:y+h, x:x+w]
        count += 1
        cv2.imwrite(f'./sleep_image/image_{count}_0.jpg', leye_img)
        cv2.imwrite(f'./sleep_image/image_{5000+count}_0.jpg', reye_img)
    return None

face_path = './haar cascade files/haarcascade_frontalface_alt.xml'
eyes_path = './haar cascade files/haarcascade_eye.xml'
face_cascade, eyes_cascade = load_haarcascade(face_path, eyes_path)

count = 0
while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray)
    if len(faces)>0:
        sorted_faces = sorted(faces, key=keysort, reverse=True)
        main_faces = sorted_faces[0]
        x, y, w, h = main_faces
        gray_face = gray[y:y+h, x:x+w]
        face = frame[y:y+h, x:x+w]
        eyes_value = eyes_detector(frame, face, gray_face)

    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()