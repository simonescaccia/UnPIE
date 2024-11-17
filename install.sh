#!/urs/bin/env bash

pip install --upgrade pip
apt-get update
apt-get install -y python3-pip
add-apt-repository ppa:deadsnakes/ppa &&
apt-get update &&
apt-get install python3.10 --assume-yes
apt-get install libpython3.10
python3.10 -m pip install --upgrade pip
python3.10 -m pip install -r requirements_tf2.txt
