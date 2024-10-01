# FROM mathworks/matlab-deep-learning:r2023a
FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04

USER root

RUN useradd -m -u 1119 ashida
# uid 1119 ashidaにsudoを使えるようにする
RUN usermod -aG sudo ashida
RUN echo "ashida ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
#nointeractive
ENV DEBIAN_FRONTEND=noninteractive

#Python install
RUN apt update && \
    apt install -y python3 python3-pip python3-opencv

#-e MLM_LICENSE_FILE=/licenses/network.lic

# ENV MLM_LICENSE_FILE=/licenses/network.lic
# RUN cd /opt/matlab/R2023a/extern/engines/python/ && python setup.py install

RUN pip install scikit-image matplotlib 

RUN pip install numba h5py pandas tqdm notebook piexif rawpy


RUN apt-get install tmux screen -y
# RUN cd /opt/matlab/R2023a/extern/engines/python/ && pip install .


# COPY ./network.lic /licenses/network.lic

# RUN cd /home/ashida && wget https://www.mathworks.com/mpm/glnxa64/mpm && chmod +x mpm  && ./mpm install --release=R2023a --destination=/opt/matlab/R2023a/ --products Optimization_Toolbox

RUN pip install numba h5py pandas tqdm notebook piexif rawpy

RUN apt-get install nano -y
ENV LANG C.UTF-8
ENV LANGUAGE en_US
# RUN python3 -m cupyx.tools.install_library --cuda 11.x --library cutensor
USER ashida
RUN pip install -U setuptools pip &&  pip install cupy-cuda11x chainer

WORKDIR /home/ashida/
#python3 -m cupyx.tools.install_library --library cudnn --cuda 11.x
RUN pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
CMD ["/bin/bash -c 'python3 -m cupyx.tools.install_library --library cudnn --cuda 11.x'"]
