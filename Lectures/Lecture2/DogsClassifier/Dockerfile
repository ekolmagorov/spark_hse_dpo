FROM python:3.10-slim

WORKDIR /app
COPY ./requirements.txt /app/requirements.txt

RUN pip3 install -r requirements.txt 

COPY . /app

EXPOSE 5000

ENTRYPOINT ["python3", "flask_dogs_classifier.py"]
