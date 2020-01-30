FROM tensorflow/tensorflow:1.13.1-py3

RUN apt-get update -y

RUN apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev

RUN pip install --upgrade pip

COPY requirements.txt /workspace/requirements.txt

RUN pip install -r /workspace/requirements.txt

COPY server.py /workspace/server.py

RUN chmod -x /workspace/server.py

COPY model/corrmodels.py /workspace/model/corrmodels.py

COPY inference /workspace/inference

COPY static /workspace/static

COPY templates /workspace/templates

WORKDIR /workspace

CMD ["python","server.py" ]

