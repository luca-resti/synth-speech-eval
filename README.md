# Multi-Scale Synthetic Speech Evaluation Framework

A multi-scale framework for synthetic speech quality evaluation for quality assurance.

## Overview



## Installation

Download the SQ_AST model weights from: https://github.com/WafaaWardah/SQ-AST and place them in the ./models/weights/ folder

```bash
conda env create -f environment.yml
conda activate synth-speech-eval

pip install -r requirements.txt
```

## Usage

Use the first arg after eval.py to select the config to run from the ./configs/ folder

```bash
python eval.py default
```

## Acknowledgements

**Project Co-Leads**: Luca Resti (University of York), James Walker (University of York)<br>
**Lead Developer**: Ben Heritage (University of York)

Project supported by the CoSTAR Network via the EPSRC IAA CoSTAR Live Lab Researcher Mobility fund.
