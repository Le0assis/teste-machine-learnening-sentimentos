

import cv2
import pandas as pd
import numpy as np
import joblib

model = joblib.load("model.pkl")

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

cam = cv2.VideoCapture(0)

data =[]
labels = []
if face_cascade.empty():
    print("Erro ao carregar o cascade!")
else:
    print("Cascade carregado com sucesso!")
if not cam.isOpened():
    print("erro")

while True:
    ret, frame = cam.read() 
    #ret para verificar se a camera funciona
    # frame, imagem da camera em matriz Numpy

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    # Detectores clássicos trabalham melhor em escala de cinza

    faces = face_cascade.detectMultiScale(gray, 1.1, 8)

    for(x, y, w, h) in faces:
    # comprimento, altura, largura, e altura
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
        
        cv2.imshow('Face Detection', frame)
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))
        face = face.flatten()
        key = cv2.waitKey(1) & 0xFF

        prediction = model.predict([face])
        cv2.putText(frame, prediction[0], (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
    cv2.imshow('Face Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):

        cam.release()
        cv2.destroyAllWindows
        df = pd.DataFrame(data)
        df["label"] = labels
        df.to_csv("faces.csv", mode="a", index=False)

        break



