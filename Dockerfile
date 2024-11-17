FROM tensorflow/tensorflow:2.15.0 AS env

WORKDIR /app

COPY requirements_tf2.txt .
COPY install.sh .

RUN bash install.sh

FROM env

COPY . .

CMD ["sh", "run_training.sh"]