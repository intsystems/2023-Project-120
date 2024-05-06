FROM anibali/pytorch:1.13.0-cuda11.8-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN echo '[INFO] Start install dependencies' \
    && apt-get update \
    && apt-get install -y \
           ffmpeg \
           libsm6 \
           libxext6 \
           python3-dev \
           libasound2-dev \
           pulseaudio \
          alsa-utils \
    && echo '[INFO] Finish install dependencies'


COPY ./requirements.txt /tmp/requirements.txt 

RUN pip install -r /tmp/requirements.txt

COPY ./ /code

WORKDIR /code

RUN /bin/bash
