FROM python:3.10

ENV FLASK_APP serve.py
ENV ENABLE_DEBUG False
ENV OPENAI_API_KEY 
ENV HUGGINGFACEHUB_API_TOKEN 
ENV _BARD_API_KEY 
ENV GOOGLE_APPLICATION_CREDENTIALS 
ENV DOCS_PERSIST_DIRECTORY db_docs/
ENV IMAGES_PERSIST_DIRECTORY db_images/
# Assumes disk mounted at /mnt/disk
ENV HF_HOME /mnt/disk/hf_cache
# Allow statements and log messages to immediately appear in the Knative logs
ENV PYTHONUNBUFFERED True

# Copy requirements.txt to the docker image and install packages
COPY . /app
RUN pip install -r app/requirements.txt

EXPOSE 8080
ENV PORT 8080
WORKDIR /app

CMD exec flask run --host 0.0.0.0 --port $PORT
