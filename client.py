import os
import requests
import numpy as np
from sshtunnel import SSHTunnelForwarder


# Создаем матрицу признаков размером количество объектов x количество признаков у объекта
matrix = np.random.rand(10, 5).tolist()

# Настройка туннеля: установим SSH-соединение с сервером
server = SSHTunnelForwarder(
    ssh_address_or_host=("176.109.74.200", 2223), # адрес SSH-сервера и порт
    ssh_username=os.getenv("SSH_USERNAME"),
    ssh_password=os.getenv("SSH_PASSWORD"),
    remote_bind_address=("172.17.0.1", 1337), # адрес и порт на удалённом сервере, к которому нужно подключаться
    local_bind_address=("127.0.0.1", 7777) # локальный адрес и порт, через который будет доступен сервис
)

server.start()
print("Туннель запущен, локальный порт: ", server.local_bind_port)

# Теперь можно обращаться к сервису через локальный адрес
response = requests.post(url=f"http://127.0.0.1:{server.local_bind_port}/predict", json={"features_matrix": matrix})
print("Полученные предсказания: ", response.json()["predictions"])

server.stop()
print("Туннель остановлен")
