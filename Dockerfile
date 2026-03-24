FROM ubuntu:24.04

# Set non-interactive to avoid prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install an older Miniconda version compatible with Ubuntu 18.04 (glibc 2.27)
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh \
    && bash miniconda.sh -b -p /opt/conda \
    && rm miniconda.sh

ENV PATH="/opt/conda/bin:${PATH}"
ENV CONDA_PLUGINS_AUTO_ACCEPT_TOS=yes

COPY environment.yml ./ 
COPY requirements.txt ./

RUN conda env create -f environment.yml -v
RUN conda run -n synth-speech-eval pip install -r requirements.txt -v \
    && conda run -n synth-speech-eval pip install espnet==202511 espnet-tts-frontend==0.0.3 -v

RUN apt-get update && apt-get install -y espeak-ng

# Add all files from repo
COPY . ./

CMD ["conda", "run", "-n", "synth-speech-eval", "python3", "./eval.py", "audiomos25"]
