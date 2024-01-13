FROM python:3.11-slim

RUN mkdir /app
RUN mkdir /root/data
WORKDIR /app

COPY ./test_program.py ./
COPY ./data.txt /root/data
RUN chmod +x test_program.py

ARG PROGRAMM_NAME
ENV SECRET_KEY hello

RUN echo Try build: $PROGRAMM_NAME

CMD ["python3", "test_program.py"]

