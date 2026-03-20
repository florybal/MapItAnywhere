#!/bin/bash

# script for starting/reusing a Docker container for the MIA mapper project.
# previously we always spawned a new container with --rm which meant the
# image was pulled and volumes re-mounted on every invocation. this version
# keeps a named container around so subsequent calls attach to it.

CONTAINER_NAME="mia-mapper"

# make a host-side logs directory that will be mounted, so the
# container can dump its output and we can inspect it later.
mkdir -p "$(pwd)/logs"

# timestamp for log file inside project (mounted back to host)
_ts=$(date +%Y%m%d_%H%M%S)

# the user can pass an arbitrary command to execute in the container;
# if none is supplied we default to launching a shell.
COMMAND="${*:-bash}"

# helper: execute a single command inside an existing container.
# we install requirements on-demand only when running a mapper command
# so that interactive shells are quick and don't hang in a pipeline.
run_in_container_command() {
    docker exec -it "$CONTAINER_NAME" bash -lc "cd /home/user/project && \
        python -m pip install -r requirements.txt && \
        $COMMAND 2>&1 | tee /home/user/project/logs/run_${_ts}.log"
}

# helper: open an interactive shell in running container
run_in_container_shell() {
    docker exec -it "$CONTAINER_NAME" bash
}

# check if container exists (running or stopped)
if docker ps -a --filter "name=^/${CONTAINER_NAME}$" --format '{{.Names}}' | grep -qw "$CONTAINER_NAME"; then
    # container is known; see if it's running
    if docker ps --filter "name=^/${CONTAINER_NAME}$" --format '{{.Names}}' | grep -qw "$CONTAINER_NAME"; then
        echo "Container '$CONTAINER_NAME' is already running, attaching..."
    else
        echo "Container '$CONTAINER_NAME' exists but is stopped; starting..."
        docker start "$CONTAINER_NAME" >/dev/null
    fi
    # after ensuring container is running, pick mode
    if [ "$COMMAND" = "bash" ]; then
        run_in_container_shell
    else
        run_in_container_command
    fi
    exit 0
fi

# no existing container, create a new one
# when creating we can preinstall requirements to speed later runs

# compute a large shared-memory size if not supplied via env
if [ -z "$SHM_SIZE" ]; then
    if mem_kb=$(awk '/MemTotal/ {print $2}' /proc/meminfo 2>/dev/null); then
        # convert kB -> GB and take 90%
        shm_g=$(awk -v m=$mem_kb 'BEGIN {printf "%dg", int(m/1024/1024*0.9)}')
        SHM_SIZE=${shm_g}
    else
        SHM_SIZE=16g
    fi
fi

docker run -it \
  --name "$CONTAINER_NAME" \
  --runtime=nvidia \
  --gpus all \
  --shm-size=${SHM_SIZE} \
  -v "$(pwd):/home/user/project" \
  -v "$(pwd)/datasets:/home/user/project/datasets" \
  -v "$(pwd)/outputs:/home/user/project/outputs" \
  -v "$(pwd)/scripts:/home/user/project/scripts" \
  mia-mapper:latest \
  bash -lc "cd /home/user/project && \
             python -m pip install -r requirements.txt && \
             if [ \"$COMMAND\" = \"bash\" ]; then
                 exec bash
             else
                 $COMMAND 2>&1 | tee /home/user/project/logs/run_${_ts}.log
             fi"
