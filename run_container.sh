#!/bin/bash

docker run --rm -it \
--runtime=nvidia \
--gpus all \
--shm-size=16g \
-v "$(pwd)/datasets:/home/user/datasets/" \
-v "$(pwd)/outputs:/home/user/output/" \
-v "$(pwd)/models:/home/user/input/" \
mia:test
