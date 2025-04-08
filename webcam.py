import cv2
from ultralytics import YOLO
import random
import requests

model = YOLO("yolo12n")
facemodel = YOLO("model.pt")

colors = {}
num_classes = len(model.names)
colors = {i: tuple(random.randint(0, 255) for _ in range(3)) for i in range(num_classes)}

kameraVideo = cv2.VideoCapture(0)
if not kameraVideo.isOpened(): exit()
frame_count = 0

def translate(url):
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
    if "result" in data:
        return data['result']
    else:
        return 


def makeBox(result, kerangka, model):
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # Ambil class_id dan nama class
        class_id = int(box.cls[0])
        class_name = model.names[class_id] if hasattr(model, 'names') else str(class_id)

        # Ambil warna dari dict
        color = colors.get(class_id, (255, 255, 255))

        if class_name == "Face": color = (0, 255, 0)

        print(class_name)
        url = f'https://kamusrejang.vercel.app/api/word/translate/Indonesia?word={class_name}&lang=en'
        hasil = translate(url)

        # Gambar kotak dan label
        cv2.rectangle(kerangka, (x1, y1), (x2, y2), color, 2)
        cv2.putText(kerangka, hasil, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

def predict(frame, kerangka, model):
    result = model.predict(frame)
    makeBox(result[0], kerangka, model)

while True:
    ret, kerangka = kameraVideo.read()
    frame_kecil = cv2.resize(kerangka, (640, 480))
    frame_count += 1
    if frame_count % 2 != 0:
        continue  
    
    result_1 = predict(frame_kecil, kerangka, model)
    result_2 = predict(frame_kecil, kerangka, facemodel)

    cv2.imshow('PENDETEKSI MANUSIA', kerangka)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

kameraVideo.release()
cv2.destroyAllWindows()
