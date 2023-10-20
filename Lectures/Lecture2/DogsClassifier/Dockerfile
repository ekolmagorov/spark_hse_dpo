# Используем в качестве базового образа образ Python
FROM python:3.10-slim

# Создадим внтури контейнера директорию app
WORKDIR /app
COPY ./requirements.txt /app/requirements.txt

# Установим в контейнере все необходимые зависимости
RUN pip3 install -r requirements.txt 

# Перекопируем всё содержимое текущей папки внутрь папки контейнера /app
COPY . /app

# Устанавливаем порт, на котором будет работать приложение
EXPOSE 5000

# Команда, которая запустится при docker run
ENTRYPOINT ["python3", "flask_dogs_classifier.py"]
