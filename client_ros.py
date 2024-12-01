import rospy
import socket
import logging
import time
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from clover import long_callback
from cv2 import long_callback


# Настройка логирования в файл
logging.basicConfig(
    filename='client_log.log',  # Имя файла для логов клиента
    filemode='a',  # Режим записи: 'a' для добавления к существующему файлу
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


rospy.init_node('cv')
bridge = CvBridge()

@long_callback
def send_image(data):
    host = 'localhost'
    port = 8080
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((host, port))
    while True:
        img = bridge.imgmsg_to_cv2(rospy.wait_for_message('main_camera/image_raw', Image), 'bgr8')

        _, img_encoded = cv2.imencode('.jpg', img)
        img_data = img_encoded.tobytes()
        img_size = len(img_data)

        # Отправляем размер изображения (4 байта)
        client_socket.sendall(img_size.to_bytes(4, byteorder='big'))

        # Логируем время отправки изображения
        send_time = time.time()
        logger.info(f"Изображение отправлено в: {send_time:.6f} секунд")

        # Отправляем само изображение
        client_socket.sendall(img_data)

        # Ожидаем подтверждение от сервера
        ack = client_socket.recv(3)  # Ожидаем 3 байта для ACK
        if ack != b'ACK':
            logger.error("Ошибка: подтверждение не получено.")
            print("Ошибка: подтверждение не получено.")
            break

        # Логируем время подтверждения получения изображения сервером
        ack_time = time.time()
        logger.info(f"Подтверждение получено в: {ack_time:.6f} секунд")

        latency = ack_time - send_time
        logger.info(f"Задержка на вычисление: {latency:.6f} секунд")


image_sub = rospy.Subscriber('main_camera/image_raw', Image, send_image)
rospy.spin()
