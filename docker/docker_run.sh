#!/bin/bash
username=ashida
docker run  \
        --shm-size=12G \
        -itd --rm \
        --name dp-sfm-cupy \
        --gpus=all \
	-u $(id -u $username):$(id -g $username) \
        -v $PWD/../:/home/ashida/ws/\
        dp-sfm-cupy\
        /bin/bash
