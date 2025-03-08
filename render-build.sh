#!/bin/bash
apt-get update && apt-get install -y libgl1-mesa-glx
pip install -r requirements.txt
python -m spacy download en_core_web_sm
