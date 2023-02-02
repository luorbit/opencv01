import numpy as np
import cv2

# Carrega o modelo de reconhecimento facial
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Inicia a captura da webcam
cap = cv2.VideoCapture(0)

while True:
    # Le a imagem da webcam
    _, frame = cap.read()
    # Converte a imagem para escala de cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detecta as faces na imagem
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    print(faces)
    # Desenha um círculo de cor vermelha nas faces detectadas
   # for (x, y, w, h) in faces:
    #    cv2.circle(frame, (int(x + (w / 2)), int(y + (h / 2))), int(w / 2), (0, 0, 255), -1)
    #    print(x,y,w,h)
    # Mostra a imagem
    cv2.imshow('frame', frame)
    # Pressione q para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera e fecha a webcam
cap.release()
cv2.destroyAllWindows()