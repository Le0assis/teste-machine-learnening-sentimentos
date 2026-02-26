
#Transformar em função
#Fazer com que salve apenas quando aperta botão e escreva a emoção


import cv2
import pandas as pd
import numpy as np

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascades_frontalface_default.xml'
)

cam = cv2.VideoCapture(0)

data =[]
labels = []

while True:
    ret, frame = cam.read() 
    #ret para verificar se a camera funciona
    # frame, imagem da camera em matriz Numpy

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detectores clássicos trabalham melhor em escala de cinza

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for(x, y, w, h) in faces:
    # comprimento, altura, largura, e altura
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
        
        cv2.imshow('Face Detection', frame)
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))
        face = face.flatten()

        data.append(face)
        labels.append("feliz")

    cv2.imshow("Face", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



cam.realese()
cv2.destroyAllWindows

df = pd.DataFrame(data)
df["label"] = labels
df.to_csv("faces.csv", index=False)