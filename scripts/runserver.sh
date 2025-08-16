#!/bin/bash

# Common exports across scripts
source scripts/common_env.sh

export FLASK_APP=serve.py
export ENABLE_DEBUG=False
export FACTORY_TYPE=GOOGLE
#export SD_URL=http://127.0.0.1:7860
#export SD_STEPS=5
#export IMAGE_OUTPUT_PATH=./generated_images/output.png

#pipenv flask run --host 0.0.0.0 --port 5000
gunicorn --bind 0.0.0.0:5000 serve:app

# Alternative run command using main() method
#pipenv run python serve.py
