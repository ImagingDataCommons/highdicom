FROM python:3.8-buster

RUN export DEBIAN_FRONTEND=noninteractive && \
    export DEBCONF_NONINTERACTIVE_SEEN=true && \
    apt-get update && \
    apt-get install -y --no-install-suggests --no-install-recommends \
        build-essential \
        libjpeg62-turbo-dev \
        libopenjp2-7-dev \
        software-properties-common && \
    apt-get clean

RUN python -m pip install --upgrade pip && \
    python -m pip install --prefix=/usr/local \
    dicomweb-client \
    dumb-init \
    highdicom \
    jupyterlab \
    numpy \
    matplotlib \
    Pillow \
    torch

RUN useradd -m -s /bin/bash jupyter

USER jupyter

EXPOSE 8888

COPY notebooks /usr/local/share/highdicom-examples

WORKDIR /usr/local/share/highdicom-examples

ENTRYPOINT ["/usr/local/bin/dumb-init", "--", \
           "/usr/local/bin/jupyter", "lab", "--ip", "0.0.0.0", "--no-browser"]
