import socket
import math
import logging
import time
import multiprocessing as mp
import numpy as np
import cv2
import os
from ultralytics import YOLO
from dotenv import load_dotenv
from classes import classNames


load_dotenv()

# Настройка логирования в файл
logging.basicConfig(
    filename='server_log.log',  # Имя файла для логов
    filemode='a',  # Режим записи: 'a' для добавления, 'w' для перезаписи
    level=logging.DEBUG,  # Уровень логирования
    format='%(asctime)s - %(levelname)s - %(message)s'  # Формат логов
)

logger = logging.getLogger(__name__)

model = YOLO("yolo-Weights/yolov8n.pt")


class Server:
    def __init__(self, host='0.0.0.0', port=8080):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((host, port))
        self.server_socket.listen(1)
        self.ctx = mp.get_context('spawn')
        self.frames_queue = self.ctx.Queue()
        self.results_queue = self.ctx.Queue()
        self.detect_process = self.ctx.Process(target=detection, args=(self.frames_queue, self.results_queue))
        self.detect_process.start()
        self.frame_processed = False
        self.boxes = list()
        logger.info(f"Сервер запущен на {host}:{port}")
        print(f"Сервер запущен на {host}:{port}")

    def run(self):
        while True:
            try:
                client_socket, addr = self.server_socket.accept()
                logger.info(f"Подключен клиент: {addr}")
                print(f"Подключен клиент: {addr}")
                self.handle_client(client_socket)
            except Exception as e:
                logger.error(f"Ошибка при подключении клиента: {e}")
                print(f"Ошибка при подключении клиента: {e}")

    def handle_client(self, client_socket):
        while True:
            try:
                # Получаем размер изображения
                data = client_socket.recv(4)
                if not data:
                    logger.warning("Соединение с клиентом разорвано.")
                    print("Соединение с клиентом разорвано.")
                    break

                img_size = int.from_bytes(data, byteorder='big')

                # Получаем само изображение
                img_data = b''
                while len(img_data) < img_size:
                    packet = client_socket.recv(img_size - len(img_data))
                    if not packet:
                        logger.warning("Соединение с клиентом разорвано.")
                        print("Соединение с клиентом разорвано.")
                        break
                    img_data += packet

                # Логируем время получения изображения
                receive_time = time.time()
                logger.info(f"Изображение получено в: "
                            f"{receive_time:.6f} секунд")

                # Преобразуем байты в изображение
                img_array = np.frombuffer(img_data, dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                # Обработка кадра
                logger.debug("Обработка кадра...")
                if not self.frame_processed:
                    self.frames_queue.put(img)
                    self.frame_processed = True
                if not self.results_queue.empty():
                    self.frame_processed = False
                    self.boxes = self.results_queue.get()
                for (x1, y1, x2, y2) in self.boxes:
                    # put box in cam
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                    # object details
                    org = [x1, y1]
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    fontScale = 1
                    color = (255, 0, 0)
                    thickness = 2

                    cv2.putText(img, 'person', org, font, fontScale, color, thickness)

                # self.detect_people(img)

                if img is not None:
                    cv2.imshow("Received Image", img)
                    cv2.waitKey(1)

                # Отправляем подтверждение клиенту
                client_socket.sendall(b'ACK')

            except Exception as e:
                logger.error(f"Ошибка при обработке данных от клиента: {e}")
                print(f"Ошибка при обработке данных от клиента: {e}")
                self.detect_process.join()
                break

        self.detect_process.join()
        client_socket.close()
        logger.info("Соединение с клиентом закрыто.")
        print("Соединение с клиентом закрыто.")
        cv2.destroyAllWindows()  # Закрываем все окна после разрыва соединения

def detection(frames_queue, results_queue):
    while True:
        if not frames_queue.empty():
            frame = frames_queue.get()
            boxes = detect_people(frame)
            results_queue.put(boxes)

def detect_people(img):
    boxes_ = list()
    results = model(img)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0])
            if classNames[cls] == "person":
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                boxes_.append((x1, y1, x2, y2))
    return boxes_


if __name__ == '__main__':
    host = os.getenv('CONNECT_CLIENT_HOST')
    port = os.getenv('CONNECT_CLIENT_PORT')
    # server = Server(host=host, port=port)
    server = Server()
    server.run()
