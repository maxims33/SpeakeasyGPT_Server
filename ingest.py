"""
#TODO Consider multi threading to improve performance of ingestion
"""

import argparse
import langchain
import nltk
from speakeasy.llmfactory import init_factory
from speakeasy.indexes import init_document_db, init_image_db
from env_params import env_config


def parse_arguments():
    """ Parse the command line arguments """
    parser = argparse.ArgumentParser(description='SpeakEasyGPT - Ingest')
    parser.add_argument("--ingest-path", "-P",
            help='The file path to look for content to ingest', default='./filebox/')
    return parser.parse_args()

def params_from_path():
    """ Parse the command line arguments """
    cmdline_args = parse_arguments()
    params = {'file_path': cmdline_args.ingest_path}
    print(f"Ingest Path: {params['file_path']}")
    return params

def main():
    """ Run from command line """
    params = params_from_path()
    langchain.debug = env_config['debug']

    factory = init_factory(env_config)
    init_document_db(factory,
        file_path=params['file_path'],
        persist_dir=env_config['docs_persist_directory']
    )
    init_image_db(factory,
        file_path=params['file_path'],
        persist_dir=env_config['images_persist_directory']
    )


def ensure_nltk_data(resource_path):
    """
    Checks if an NLTK resource is available and downloads it if missing.
    TODO: Check if works on windows
    Args:
        resource_path (str): The path of the NLTK resource to check/download
                             (e.g., 'tokenizers/punkt').
    """
    try:
        nltk.data.find(f'{resource_path}')
    except LookupError:
        print(f"NLTK resource '{resource_path}' not found. Downloading...")
        resource_name = resource_path.split('/')[-1]
        try:
            nltk.download(resource_name)
            print(f"NLTK resource '{resource_name}' downloaded successfully.")
        except Exception as e:
            print(f"Error downloading NLTK resource '{resource_name}': {e}")
            
if __name__ == "__main__":
    # Ensure the nltk data available, download it if not
    ensure_nltk_data('tokenizers/punkt_tab')
    ensure_nltk_data('taggers/averaged_perceptron_tagger_eng')
    
    main()
