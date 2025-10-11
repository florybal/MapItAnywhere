# Map It Anywhere (MIA)

## Overview
This project is a modified version of the original **Map It Anywhere (MIA)** project. The modifications have been made in compliance with the Creative Commons Attribution-ShareAlike 4.0 International License (CC BY-SA 4.0).

The original project can be found at: [Map It Anywhere GitHub Repository](https://github.com/MapItAnywhere/MapItAnywhere).

Map It Anywhere (MIA) is a project designed to empower Bird’s Eye View (BEV) mapping using large-scale public data. This README provides instructions on how to set up and run the project.

---

## Prerequisites

1. **Install Docker**:
   - Follow the instructions on the [Docker website](https://www.docker.com/get-started) to install Docker.

2. **Clone the Repository**:
   ```bash
   git clone https://github.com/florybal/MapItAnywhere.git
   cd MapItAnywhere
   ```

3. **Install Git LFS** (if not already installed):
   ```bash
   sudo apt-get install git-lfs
   git lfs install
   ```

---

## Running the Project

### 1. Build the Docker Image
Run the following command to build the Docker image:
```bash
bash run_container.sh build
```

### 2. Start the Docker Container
To start the container in detached mode:
```bash
bash run_container.sh up
```

### 3. Access the Container
To open an interactive shell inside the running container:
```bash
bash run_container.sh shell
```

### 4. Stop the Container
To stop and remove the running container:
```bash
bash run_container.sh down
```

---

## Directory Structure
- `docker-compose.yml`: Configuration for Docker Compose.
- `run_container.sh`: Script to manage Docker containers.
- `mapper/`: Contains the core code for data processing and mapping.
- `datasets/`: Directory for storing datasets (ignored by Git).
- `weights/`: Directory for storing model weights (ignored by Git).

---

## Notes
- Ensure that the `datasets/` and `weights/` directories are populated with the required data and model weights before running the project.
- For additional configuration, modify the YAML files in the `conf/` directory.

---

## Trained Weights
We have hosted trained weights for the Mapper model using the MIA dataset on Huggingface. [Click Here](https://huggingface.co/mapitanywhere/mapper) to download.

---

## License
This project is licensed under the CC-BY-NC license. See the LICENSE file for details.