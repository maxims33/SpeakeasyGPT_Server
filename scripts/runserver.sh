#!/bin/bash

# Common exports across scripts
source scripts/common_env.sh

export FLASK_APP=serve.py
export ENABLE_DEBUG=False
export FACTORY_TYPE=GOOGLE

#flask run --host 0.0.0.0 --port 5000
gunicorn --bind 0.0.0.0:5000 serve:app

# Alternative run command using main() method
#python serve.py
