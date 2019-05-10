FROM wylswz/face_detection:0.2

ADD ./requirements.txt /algorithms/requirements.txt
ENTRYPOINT ["cd", "/algorithms"]
EXPOSE 80/tcp
EXPOSE 80/udp
EXPOSE 8000/tcp
RUN apt update
RUN apt install protobuf-compiler python3-pil python3-lxml python3-tk  python3-pip python-pip git vim curl -y
RUN pip install cython
RUN pip3 install Cython numpy
RUN pip3 install -r /algorithms/requirements.txt

