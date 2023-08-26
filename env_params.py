"""
#TODO Additional params to potentially expose in future
#TODO Consider using dotenv to simplyfy handling of envrionment variables
"""

import os

def default_variables():
    """ Setting defaults for all environment variables """
    return {
        'debug': False,
        'docs_persist_directory': './db/',
        'images_persist_directory': './db2/',
        'factory_type': 'GOOGLE',
        'sd_url': None,
        'sd_steps': 10,
        'image_output_filename':'./generated_images/output.png',
        'google_llm_api_timeout': 30,
        'bard_experimental': False,

        #'local_model_name': 'google/flan-t5-large',
        #'device_id': 'cpu',
        #'max_iterations': 1,
        #'max_length': 512,
        #'embeddings_model_name': 'intfloat/e5-large-v2',
        #'embeddings_device_id': 'cuda',
        #'source_chunks': 1,
        #'text_split_size': 1000,
        #'text_split_overlap': 200,
    }

def parse_environment_variables():
    """ Setting the configured values, if defined """
    env_variables = default_variables()

    if os.environ.get("ENABLE_DEBUG") == 'True' or os.environ.get("ENABLE_DEBUG") == 'true':
        env_variables['debug'] = True

    if not os.environ.get("DOCS_PERSIST_DIRECTORY") is None:
        env_variables['docs_persist_directory'] = os.environ.get("DOCS_PERSIST_DIRECTORY")

    if not os.environ.get("IMAGES_PERSIST_DIRECTORY") is None:
        env_variables['images_persist_directory'] = os.environ.get("IMAGES_PERSIST_DIRECTORY")

    if not os.environ.get("FACTORY_TYPE") is None:
        env_variables['factory_type'] = os.environ.get("FACTORY_TYPE")

    if not os.environ.get("LOCAL_MODEL_NAME") is None:
        env_variables['local_model_name'] = os.environ.get("LOCAL_MODEL_NAME")

    if not os.environ.get("EMBEDDINGS_MODEL_NAME") is None:
        env_variables['embeddings_model_name'] = os.environ.get("EMBEDDINGS_MODEL_NAME")

    if not os.environ.get("GOOGLE_LLM_API_TIMEOUT") is None:
        env_variables['google_llm_api_timeout'] = int(os.environ.get("GOOGLE_LLM_API_TIMEOUT"))

    if not os.environ.get("SD_URL") is None:
        env_variables['sd_url'] = os.environ.get("SD_URL")

    if not os.environ.get("SD_STEPS") is None:
        env_variables['sd_steps'] = int(os.environ.get("SD_STEPS"))

    return env_variables
