#!/bin/bash
apt-get update && apt-get install -y tesseract-ocr libtesseract-dev
pip install -r requirements.txt
python -m spacy download en_core_web_sm
