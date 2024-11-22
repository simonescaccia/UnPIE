FROM nvcr.io/nvidia/tensorflow:24.01-tf2-py3 AS env

WORKDIR /app

COPY requirements_tf2.txt .

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install -r requirements_tf2.txt

FROM env

COPY . .