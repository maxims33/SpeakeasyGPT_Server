import argparse
import langchain
from speakeasy.llmfactory import init_factory
from speakeasy.indexes import init_document_db, init_image_db
from env_params import parse_environment_variables

#TODO Consider multi threading to improve performance of ingestion

def parse_arguments():
    parser = argparse.ArgumentParser(description='SpeakEasyGPT - Ingest')
    parser.add_argument("--ingest-path", "-P", help='The file path to look for content to ingest', default='./filebox/')
    return parser.parse_args()

def params_from_path():
    cmdline_args = parse_arguments()
    params = {'file_path': cmdline_args.ingest_path}
    print(f"Ingest Path: {params['file_path']}")
    return params

def main():
    params = params_from_path()
    env_config = parse_environment_variables()
    langchain.debug = env_config['debug']

    factory = init_factory(env_config)
    init_document_db(factory, file_path=params['file_path'], persist_dir=env_config['docs_persist_directory'])
    init_image_db(factory, file_path=params['file_path'], persist_dir=env_config['images_persist_directory'])

if __name__ == "__main__":
    main()

