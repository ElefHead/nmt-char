FROM nvcr.io/nvidia/pytorch:20.11-py3
LABEL maintainer="Ganesh Jagadeesan"

RUN apt-get update && \
    apt-get install -y libgomp1 unzip curl \
    cmake build-essential libpython3-dev \
    pkg-config libgoogle-perftools-dev

RUN python3 --version
RUN pip3 --version

RUN pip3 install -U pipenv

RUN mkdir -p /app/nmt-char

WORKDIR /app/nmt-char

COPY requirements.txt /app/nmt-char/
RUN pipenv install

COPY nmt/ /app/nmt-char/nmt/

COPY tasks/ /app/nmt-char/tasks/
RUN chmod +x /app/nmt-char/tasks/*.sh

## This is still work in progress,
## I work on this repo everyday.
