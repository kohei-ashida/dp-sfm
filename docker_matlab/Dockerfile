FROM mathworks/matlab:R2023a

USER root

RUN useradd -m -u 1119 ashida
# uid 1119 ashidaにsudoを使えるようにする
RUN usermod -aG sudo ashida
RUN echo "ashida ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

RUN apt update && \
    apt install -y python3-opencv 

#-e MLM_LICENSE_FILE=/licenses/network.lic

ENV MLM_LICENSE_FILE=/licenses/network.lic
# RUN cd /opt/matlab/R2023a/extern/engines/python/ && python setup.py install

RUN pip install scikit-image matplotlib 

COPY ./network.lic /licenses/network.lic

RUN pip install numba h5py pandas tqdm notebook piexif rawpy

#/opt/matlab/R2023a/bin/glnxa64をLD_LIBRARY_PATHに追加


RUN apt-get install tmux screen -y
RUN cd /opt/matlab/R2023a/extern/engines/python/ && pip install .

RUN cd /home/ashida && wget https://www.mathworks.com/mpm/glnxa64/mpm && chmod +x mpm  && ./mpm install --release=R2023a --destination=/opt/matlab/R2023a/ --products Optimization_Toolbox && ./mpm install --release=R2023a --destination=/opt/matlab/R2023a/ --products Image_Processing_Toolbox

RUN apt-get install nano -y
ENV LANG C.UTF-8
ENV LANGUAGE en_US
RUN apt-get install tmux screen -y


USER ashida

# ENV LD_LIBRARY_PATH=/opt/matlab/R2023a/bin/glnxa64:${LD_LIBRARY_PATH}
# RUN echo $LD_LIBRARY_PATH
# RUN pip install matlabengine==9.14.3
WORKDIR /home/ashida/
