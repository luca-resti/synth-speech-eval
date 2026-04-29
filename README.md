# Multi-Scale Synthetic Speech Evaluation Framework

A multi-scale framework for synthetic speech quality evaluation for quality assurance.

## Overview

This tool produces utterance and system-level analysis for datasets of synthetic speech.
Using [SQ_AST](https://github.com/WafaaWardah/SQ-AST) and [WhisperX](https://github.com/m-bain/whisperX) as a backbone, interpretability methods are used to extract areas within the synthetic speech samples which are deemed troublesome on the sound quality metrics of Mean Opinion Score (MOS), Colouration, Discontinuity, Noisiness, and Loudness.

The following image shows an overview of the data flow within the tool.
![alt text](method_flow.png "Methodology Flow")

Before use, please refer to the repository "issues" page for any known current bugs or future development plans.

## Installation

### Cuda Drivers Installation:
[Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)<br>
[Windows](https://developer.nvidia.com/cuda-12-8-1-download-archive)

### SQ_AST Setup:
Download the SQ_AST model weights from [here](https://github.com/WafaaWardah/SQ-AST) and place them in the **./app/models/weights/** folder

### Conda Setup:
```bash
conda env create -f environment.yml
conda activate synth-speech-eval
conda install cuda-toolkit

pip install -r requirements.txt
pip install espnet==202511 espnet-tts-frontend==0.0.3
```

### Espeak setup:
**Linux:** run the following bash commands:
```bash
sudo apt-get update && sudo apt-get install espeak-ng
```

**Windows:** download and install the library from [here](https://espeak.sourceforge.net/download.html)

## Usage

Use the first arg after eval to select the config to run from the **./configs/** folder

```bash
python -m src.eval default
```

## Building Docker Images

Once downloading docker on your system and initiating the conda environment, run the following within that environment.
'''bash
create_image.bat
'''

## Licences

See the LICENSE file and third-party-licenses.txt for license information.

## Acknowledgements

**Project Co-Leads**: Luca Resti (University of York), James Walker (University of York)<br>
**Lead Developer**: Ben Heritage (University of York)

Project supported by the CoSTAR Network via the EPSRC IAA CoSTAR Live Lab Researcher Mobility fund.
