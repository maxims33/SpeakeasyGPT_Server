#!/bin/sh

# Common exports across scripts
source ./common_env.sh

export FLASK_APP=serve.py
export ENABLE_DEBUG=False
export SD_URL=http://127.0.0.1:7860
export SD_STEPS=5
#export FACTORY_TYPE=GOOGLE
#export IMAGE_OUTPUT_PATH=./generated_images/output.png

pipenv run flask run --host 0.0.0.0
