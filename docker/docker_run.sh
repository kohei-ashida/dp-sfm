#!/bin/bash
username=ashida
docker run  \
        --shm-size=12G \
        -itd --rm \
        --gpus=all \
	-u $(id -u $username):$(id -g $username) \
        -v $PWD/../:/home/ashida/ws/\
        cupy\
        /bin/bash
