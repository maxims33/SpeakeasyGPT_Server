# SpeakeasyGPT_Server
Repository for SpeakeasyGPT Server, built in Python

Prodies a JSON/REST API for use by SpeakeasyGPT client

Note that currently Voice Recognition and Text To Speah Capabilities are handled by client

## scripts

### runserver.sh

### ingest.sh

`Usage: scripts/ingest.sh -c TRUE_IF_DO_CLEAN_FIRST -f FILEPATH_TO_INGEST_FILES_FROM
`

Examples: 
`scripts/ingest.sh -c True -f ./filebox/
`

### cleandbs.sh

Clear out any ingested content

### common_envs.sh

Used by runserver.sh and ingest.sh to set the common environment variables

## Services

### query

Example usage:
`curl -X POST -H "Content-Type: application/json" -d '{"prompt": "Who is the current president"  }' http://127.0.0.1:5000/query
`
`curl -X POST -H "Content-Type: application/json" -d '{"prompt": "When did Andre Agassi win his last grand slam?", "llm_type":"LOCAL" }' http://127.0.0.1:5000/query
`
`{"response":"Andre Agassi won his last grand slam in 2003. He won the Australian Open in 2003, which was his eighth and final grand slam title."}
`

### run_agent

Example usage:
`curl -X POST -H "Content-Type: application/json" -d '{"prompt": "Who is the current president"  }' http://127.0.0.1:5000/run_agent
`

#### Tools

`
* Instruct_LLM: Use this tool to when instructed to generate content (like writing an email/letter/prompt), or for searching for information online.
* EvaluateExpression: Use this tool only for any mathematical calculations.
* Preview_Image: Use this tool to when instructed to preview image or a picture. The 'Action Input:' for this tool should be prompt optimized for stable diffusion
* Document_Query: Use this tool only to query about documents stored LOCALLY
* Image_Query: Use this tool to only query the captions of images stored LOCALLY
* Search_Image: Use this tool when asked to find or search for an image
* PAL_Logic: Use this tool only for programming logic to determine solution. DO NOT do the math yourself, or change the question wording when defining the 'Action Input:' for this tool.
* Run_Code: Use this tool when asked to create and execute code / script. Pass the original request as the 'Action Input:' for this tool
`

##### Custom Tools

##### Vectorstore Tools

##### Image Tools

### run_convo_agent

Example usage:
`curl -X POST -H "Content-Type: application/json" -d '{"prompt": "Who is the current president"  }' http://127.0.0.1:5000/run_convo_agent
`

## Environment Variable

`
FLASK_APP=
ENABLE_DEBUG=
DOCS_PERSIST_DIRECTORY= 
IMAGES_PERSIST_DIRECTORY= 
FACTORY_TYPE= 
SD_URL= 
SD_STEPS= 
IMAGE_OUTPUT_FILENAME=
GOOGLE_LLM_API_TIMEOUT=
BARD_EXPERIMENTAL=

OPENAI_API_KEY=
HUGGINGFACEHUB_API_TOKEN=
_BARD_API_KEY=
GOOGLE_APPLICATION_CREDENTIALS=
HF_HOME=
`

## Deployment

### Google Cloud App Engine

### Google Cloud Compute

