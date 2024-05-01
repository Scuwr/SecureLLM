#!/bin/bash

conda create --name secureLLM -y

conda activate secureLLM
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia -y
conda install pip -y

pip install transformers
