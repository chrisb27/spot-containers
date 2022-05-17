FROM tensorflow/tensorflow:2.1.1-gpu

RUN apt-get install -y wget && apt-get install -y dpkg && apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb && dpkg -i cuda-keyring_1.0-1_all.deb
RUN apt-get update && apt-get install -y libsm6 libxext6 libxrender-dev libgl1-mesa-glx

COPY docker-requirements.txt prebuilt/*.whl ./
RUN python3 -m pip install -r docker-requirements.txt --find-links .

ENV LANG en_US.UTF-8 
ENV BOSDYN_CLIENT_USERNAME=Sasha
ENV BOSDYN_CLIENT_PASSWORD=olympicsasha1


COPY tensorflow_object_detection.py /app/
COPY spot_detect_and_follow.py /app/
COPY faster_rcnn_inception_v2_coco_2018_01_28 /app/
WORKDIR /app

ENTRYPOINT ["/usr/bin/python3", "/app/spot_detect_and_follow.py", "--model-path", "/app/frozen_inference_graph.pb", "--detection-class", "1","--sleep-between-capture", "0.05",  "192.168.80.3"]
