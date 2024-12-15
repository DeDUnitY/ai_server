import time

tests = False
import cv2
import torch
import numpy as np

from ultralytics import YOLO

model = YOLO("../final/detector+rasp/yolov9.pt")
model2 = YOLO("../final/detector+rasp/yolov8s.pt")
device = torch.device('cuda')
model.to(device)
model2.to(device)

print(model2.names)
def detect(img):
    global tests
    objects_dict = {}
    img_arr = np.array([img.transpose(2, 0, 1)])
    img_arr = torch.from_numpy(img_arr).float().to(device)
    if tests:
        results = model(img_arr, verbose=False)
    else:
        results = model2(img_arr, verbose=False)

    for r in results:
        n = len(r.boxes.cls)
        for i in range(n):
            cls = int(r.boxes.cls[i].cpu())
            temp_obj = [r.boxes.conf[i].cpu(),
                        r.boxes.xyxy[i].cpu().numpy()]  # уверенность модели, координаты прямоугольника

            if cls not in objects_dict:
                objects_dict[cls] = [temp_obj]
            else:
                objects_dict[cls].append(temp_obj)
    return objects_dict


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame_height, frame_width, _ = frame.shape
    center_x = frame_width / 2
    center_y = frame_height / 2
    play = False
    frames_with_door = 0  # Счётчик кадров, на которых найдена дверь
    threshold_frames = 3  # Необходимое количество последовательных кадров с объектом
    center_margin = 0.3
    central_region = (
        center_x * (1 - center_margin),  # Левая граница центральной области
        center_x * (1 + center_margin),  # Правая граница центральной области
        center_y * (1 - center_margin),  # Верхняя граница центральной области
        center_y * (1 + center_margin),  # Нижняя граница центральной области
    )
    # Применяем функцию обнаружения объектов
    objects_dict = detect(frame)
    door_detected = False
    for cls, obj_list in objects_dict.items():
        for conf, box in obj_list:
            x1, y1, x2, y2 = map(int, box)
            object_center_x = x1 + x2 / 2
            object_center_y = y1 + y2 / 2

            # Проверяем, что объект находится в центре экрана и имеет разумные размеры
            if (central_region[0] <= object_center_x <= central_region[1] and
                    central_region[2] <= object_center_y <= central_region[
                        3] and conf >= 0.5):  # Ограничение по размерам
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                if tests:
                    cv2.putText(frame, f"{model.names[cls]} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                (0, 255, 0), 2)
                else:
                    cv2.putText(frame, f"{model2.names[cls]} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                (0, 255, 0), 2)
                door_detected = True
    if door_detected:
        frames_with_door += 1
        if frames_with_door >= threshold_frames:  # Если дверь найдена на нескольких кадрах
            start = True
            play = True
            frames_with_door = 0  # Сбрасываем счётчик
    else:
        frames_with_door = 0
    # Показываем результат
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if cv2.waitKey(1) & 0xFF == ord('w'):
        tests = not tests
# Освобождаем ресурсы
cap.release()
cv2.destroyAllWindows()
