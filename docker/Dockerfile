FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04

USER root
RUN useradd -m -u 1119 ashida
RUN usermod -aG sudo ashida
RUN echo "ashida ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && \
    apt install -y tmux screen nano python3 python3-pip python3-opencv

RUN pip install numba tqdm notebook matplotlib scipy
RUN pip install cupy-cuda11x chainer

ENV LANG C.UTF-8
ENV LANGUAGE en_US
USER ashida

WORKDIR /home/ashida/ws/scripts