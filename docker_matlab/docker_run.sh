#!/bin/bash
username=ashida
docker run  \
        --shm-size=12G \
        -itd --rm \
	-u $(id -u $username):$(id -g $username) \
        -v $PWD/../:/home/${username}/matlab \
        matlab-python\
        /bin/bash
