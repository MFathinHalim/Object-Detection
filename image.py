import cv2
from ultralytics import YOLO
import random
import requests

def translate(url):
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
    if "result" in data:
        return data['result']
    else:
        return 


model = YOLO("yolo12n.pt")
facemodel = YOLO("model.pt")

colors = {}
num_classes = len(model.names)
colors = {i: tuple(random.randint(199, 255) for _ in range(3)) for i in range(num_classes)}

kerangka = cv2.imread("jkt.jpg")
if kerangka is None:
    print("Gambar tidak ditemukan!")
    exit()

# Resize dulu buat tau skala
desired_width = 640
height, width = kerangka.shape[:2]
aspect_ratio = height / width
new_height = int(desired_width * aspect_ratio)
scale = desired_width / width  # kita pakai buat skala font dan garis nanti

# Jalankan prediksi pakai gambar asli
result1 = model.predict(kerangka, conf=0.40)[0]
result2 = facemodel.predict(kerangka, conf=0.40)[0]

# Resize gambar buat tampilan
kerangka = cv2.resize(kerangka, (desired_width, new_height))

# Loop result 1
for box in result1.boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0] * scale)  # skalakan koordinat box
    class_id = int(box.cls[0])
    class_name = model.names[class_id]
    print(class_name)
    url = f'https://kamusrejang.vercel.app/api/word/translate/Indonesia?word={class_name}&lang=en'
    hasil = translate(url)
    color = colors.get(class_id, (255, 255, 255))

    cv2.rectangle(kerangka, (x1, y1), (x2, y2), color, thickness=2)
    cv2.putText(kerangka,hasil, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.7 * scale, color=color)

# Loop result 2
for box in result2.boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0] * scale)
    cv2.rectangle(kerangka, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)
    url = f'https://kamusrejang.vercel.app/api/word/translate/Indonesia?word={class_name}&lang=en'
    hasil = translate(url)
    cv2.putText(kerangka, hasil, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.7 * scale, color=(0, 255, 0))

# Tampilkan window
cv2.namedWindow("Deteksi Gabungan", cv2.WINDOW_NORMAL)
cv2.imshow("Deteksi Gabungan", kerangka)
cv2.resizeWindow("Deteksi Gabungan", desired_width, new_height)
cv2.waitKey(0)
cv2.destroyAllWindows()
