# Builder
FROM nvidia/cuda:12.6.2-devel-ubuntu24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/opt/conda/bin:${PATH}"
ENV CONDA_PLUGINS_AUTO_ACCEPT_TOS=yes

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    ca-certificates \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh \
    && bash miniconda.sh -b -p /opt/conda \
    && rm miniconda.sh

# Get environment plugins
COPY environment.yml requirements.txt ./

RUN conda env create -f environment.yml -v && \
    conda run -n synth-speech-eval pip install -r requirements.txt -v && \
    conda run -n synth-speech-eval pip install espnet==202511 espnet-tts-frontend==0.0.3 -v && \
    conda clean -afy

# Runtime
FROM nvidia/cuda:12.6.2-base-ubuntu24.04

ENV PATH="/opt/conda/bin:${PATH}"
ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_PLUGINS_AUTO_ACCEPT_TOS=yes

COPY --from=builder /opt/conda /opt/conda

RUN apt-get update && apt-get install -y --no-install-recommends \
    espeak-ng \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

# Ensure the shell uses the conda environment
SHELL ["conda", "run", "-n", "synth-speech-eval", "/bin/bash", "-c"]
CMD ["conda", "run", "-n", "synth-speech-eval", "python3", "./eval.py", "audiomos25"]
