import cv2
import cvzone
from ultralytics import YOLO

kameraVideo = cv2.VideoCapture(0)
if not kameraVideo.isOpened():
    print("jir error")
    exit()

facemodel = YOLO('model.pt')

while (True):
    ret, kerangka = kameraVideo.read()

    face_result = facemodel.predict(kerangka, conf=0.40)
    for info in face_result:
        parameters = info.boxes
        for box in parameters:
            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2 = int(x1), int(y1), int(x2), int(y2)
            h,w = y2 - y1, x2 - x1
            cv2.rectangle(kerangka, (x1, y1), (x1 + w, y1 + h), (0,255,0), 2)
            cv2.putText(kerangka, str(info.names[0]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 5)

    cv2.imshow('Nyoba Image Detection', kerangka)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

kameraVideo.release()
cv2.destroyAllWindows()