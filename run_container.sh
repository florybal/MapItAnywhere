#!/bin/bash

docker run --rm -it \
--runtime=nvidia \
--gpus all \
--shm-size=16g \
-v "$(pwd)/datasets:/home/user/datasets/" \
mia:test
