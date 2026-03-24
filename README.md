# Multi-Scale Synthetic Speech Evaluation Framework

A multi-scale framework for synthetic speech quality evaluation for quality assurance.

## Overview



## Installation

### Cuda Installation:
[Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)<br>
[Windows](https://developer.nvidia.com/cuda-12-8-1-download-archive)

### SQ_AST Setup:
Download the SQ_AST model weights from [here](https://github.com/WafaaWardah/SQ-AST) and place them in the **./models/weights/** folder

### Conda Setup:
```bash
conda env create -f environment.yml
conda activate synth-speech-eval

pip install -r requirements.txt
```

### Espeak setup:
**Linux:** run the following bash commands:
```bash
sudo apt-get update && sudo apt-get install espeak-ng
```

## Usage

Use the first arg after eval.py to select the config to run from the **./configs/** folder

```bash
python eval.py default
```

## Acknowledgements

**Project Co-Leads**: Luca Resti (University of York), James Walker (University of York)<br>
**Lead Developer**: Ben Heritage (University of York)

Project supported by the CoSTAR Network via the EPSRC IAA CoSTAR Live Lab Researcher Mobility fund.
