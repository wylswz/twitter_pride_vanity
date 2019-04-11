FROM wylswz/face_detection:0.2

ADD . /algorithms

ENTRYPOINT ["cd", "/algorithms"]
EXPOSE 80/tcp
EXPOSE 80/udp
RUN apt update
RUN apt install protobuf-compiler python3-pil python3-lxml python3-tk  python3-pip python-pip git vim -y
RUN pip install cython
RUN pip3 install Cython numpy
RUN pip3 install -r /algorithms/requirements.txt

