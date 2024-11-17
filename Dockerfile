FROM tensorflow/tensorflow:2.15.0 AS env

WORKDIR /app

COPY requirements_tf2.txt .

RUN pip install -r requirements_tf2.txt

FROM env

COPY . .

CMD ["sh", "run_training.sh"]