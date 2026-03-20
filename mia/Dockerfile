# =========================================================
# CUDA 12.4 + Ubuntu 22.04 (recomendado p/ RTX 5090)
# =========================================================
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

# ---------------------------------------------------------
# Fix mirrors + force IPv4 (evita problemas de rede)
# ---------------------------------------------------------
RUN sed -i 's|http://archive.ubuntu.com|https://archive.ubuntu.com|g' /etc/apt/sources.list && \
    sed -i 's|http://security.ubuntu.com|https://security.ubuntu.com|g' /etc/apt/sources.list && \
    echo 'Acquire::ForceIPv4 "true";' > /etc/apt/apt.conf.d/99force-ipv4

# ---------------------------------------------------------
# Dependências do sistema
# ---------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    curl \
    wget \
    pkg-config \
    software-properties-common \
    libeigen3-dev \
    libopencv-dev \
    libceres-dev \
    libcairo2-dev \
    libgeos-dev \
    libgdal-dev \
    gdal-bin \
    python3.9 \
    python3.9-dev \
    python3.9-distutils \
    python3.9-venv \
    python3-pip \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------
# Criar ambiente virtual isolado (ESSENCIAL)
# ---------------------------------------------------------
RUN python3.9 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Atualizar pip no venv
RUN pip install --upgrade pip setuptools wheel

WORKDIR /workspace

# =========================================================
# OpenSfM
# =========================================================
RUN git clone https://github.com/mapillary/OpenSfM.git && \
    cd OpenSfM && \
    pip install .

# =========================================================
# OrienterNet
# =========================================================
RUN git clone https://github.com/facebookresearch/OrienterNet.git && \
    cd OrienterNet && \
    pip install -r requirements/full.txt

# =========================================================
# Dependências extras do MIA / MapPerceptionNet
# =========================================================
RUN pip install \
    geojson \
    shapely \
    geopandas \
    mercantile \
    turfpy \
    vt2geojson \
    folium \
    geopy \
    gradio_client \
    pyarrow \
    cloudpickle==2.0.0 \
    urllib3~=1.25.6 \
    scikit-image \
    filelock \
    hydra-core \
    pytorch-lightning \
    torchmetrics

# =========================================================
# Google Earth Engine (opcional)
# =========================================================
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \
    > /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | \
    gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg && \
    apt-get update && \
    apt-get install -y google-cloud-sdk && \
    rm -rf /var/lib/apt/lists/*

RUN pip install earthengine-api

# =========================================================
# PyTorch compatível com CUDA 12.4
# =========================================================
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# =========================================================
# Final
# =========================================================
WORKDIR /workspace
CMD ["bash"]
