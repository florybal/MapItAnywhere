FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel

# Install dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    unzip \
    vim \
    ffmpeg \
    libsm6 \
    libxext6

RUN apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install --pre torchvision --index-url https://download.pytorch.org/whl/nightly/cu124 \
    && pip install nuscenes-devkit

RUN useradd -m -u 1000 user

USER user

ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME

COPY --chown=user . $HOME 

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install "numpy<2" --force-reinstall

# Get Weights
#RUN bash get_weights.sh -> deu problema na construção do docker