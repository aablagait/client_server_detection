import socket
import logging
import time
import os
import cv2
from dotenv import load_dotenv

load_dotenv()

# Настройка логирования в файл
logging.basicConfig(
    filename='client_log.log',  # Имя файла для логов клиента
    filemode='a',  # Режим записи: 'a' для добавления к существующему файлу
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

cap = cv2.VideoCapture(0)


class Client:
    def __init__(self, host='localhost', port=8080):
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect((host, port))

    def send_image(self):
        while True:
            success, img = cap.read()
            if not success:
                break

            _, img_encoded = cv2.imencode('.jpg', img)
            img_data = img_encoded.tobytes()
            img_size = len(img_data)

            # Отправляем размер изображения (4 байта)
            self.client_socket.sendall(img_size.to_bytes(4, byteorder='big'))

            # Логируем время отправки изображения
            send_time = time.time()
            logger.info(f"Изображение отправлено в: {send_time:.6f} секунд")

            # Отправляем само изображение
            self.client_socket.sendall(img_data)

            # Ожидаем подтверждение от сервера
            ack = self.client_socket.recv(3)  # Ожидаем 3 байта для ACK
            if ack != b'ACK':
                logger.error("Ошибка: подтверждение не получено.")
                print("Ошибка: подтверждение не получено.")
                break

            # Логируем время подтверждения получения изображения сервером
            ack_time = time.time()
            logger.info(f"Подтверждение получено в: {ack_time:.6f} секунд")

            latency = ack_time - send_time
            logger.info(f"Задержка на вычисление: {latency:.6f} секунд")


if __name__ == '__main__':
    host = os.getenv('CONNECT_SERVER_HOST')
    port = os.getenv('CONNECT_SERVER_PORT')
    client = Client(host=host, port=port)
    client.send_image()
