# SpeakeasyGPT_Server

The SpeakeasyGPT project is a Python-based application that provides a JSON/REST API for interacting with large language models (LLMs) and performing various tasks, including:
* **Querying LLMs:** Sending prompts to LLMs and receiving responses.
* **Searching Documents and Images:** Querying vector stores of documents and images using natural language.
* **Running Agents:** Utilizing agents that can interact with tools and external APIs to complete tasks.
 
## Key Components and Functionality:

SpeakeasyGPT project provides a flexible and extensible framework for interacting with LLMs and building AI-powered applications.

* **LLM Factories:** The llmfactory.py module provides factories for creating LLM instances from different providers. This allows you to easily switch between different LLMs without modifying the core application logic.
* **Vector Stores:** The project uses Chroma vector stores to store and search documents and images. This enables efficient retrieval of relevant information based on natural language queries.
* **Agents:** Custom LangChain agents are implemented to perform tasks by interacting with tools and external APIs. These agents can chain together multiple actions to achieve complex goals.
* **Tools:** The project defines various tools for tasks like math calculations, code execution, image generation, and vector store queries. These tools can be used by agents to complete tasks.
* **REST API:** The Flask REST API provides endpoints for interacting with the application's functionalities. This allows you to send prompts, search documents and images, run agents, and manage user accounts.

Note that currently Voice Recognition and Text To Speah Capabilities are handled by client

### Example Workflow:
1. A user sends a prompt to the `/query` endpoint.
2. The application uses the appropriate LLM factory to create an LLM instance.
3. The LLM processes the prompt and generates a response.
4. The response is returned to the user through the API.

## Installing dependencies with Pipenv

The project includes a Pipfile for managing dependencies using the Pipenv tool. 
Using Pipenv and the Pipfile helps manage dependencies effectively and ensures a consistent environment for running the SpeakeasyGPT project. Here's how to use it:

### 1. Install Pipenv:

If you don't have Pipenv installed, you can install it using pip:

```bash
pip install pipenv
```

### 2. Install Dependencies:

Run the following command to install the dependencies specified in the Pipfile:

```bash
pipenv install
```

This will create a virtual environment and install all the required packages.

### 3. (optional) Activate the Virtual Environment:

To activate the virtual environment, run:

```bash
pipenv shell
```

This will activate the virtual environment, and you'll see the environment name in your terminal prompt.
Note: This step is optional since the scripts prefix commands with `pipenv run` 

### Additional Notes:

**Pipfile.lock:** The Pipfile.lock file ensures that the exact versions of dependencies are installed, ensuring consistency across different environments.

**Updating Dependencies:** To update dependencies, you can use the following commands:

* `pipenv update:` - Updates all dependencies to their latest versions according to the version constraints in the Pipfile.
* `pipenv update [package_name]:` - Updates a specific package to its latest version.
* `pipenv install [package_name]` - This will add the package to the Pipfile and install it in the virtual environment.


## Scripts Usage Guide

This guide provides instructions on how to use the shell scripts located in the scripts folder of the SpeakeasyGPT_Server project. These scripts automate common tasks such as running the server, ingesting data, and cleaning databases.

### Prerequisites:
Ensure you have the required Python packages installed (refer to [Installing dependencies section](#installing-dependencies-with-pipenv)).
Set the necessary environment variables, including API keys and paths.
Make sure you have the appropriate permissions to execute shell scripts.

### Script Descriptions:
* **runserver.sh:** Starts the Flask development server for SpeakeasyGPT_Server.
* **ingest.sh:** Ingests data (documents and images) into the vector stores for querying.
* **cleandbs.sh:** Clears out all ingested content from the vector stores.
* **common_envs.sh:** Sets common environment variables used by other scripts.

### runserver.sh

Runs the Flask development server. The server will be accessible at http://127.0.0.1:5000 by default.

Usage:
```bash
./scripts/runserver.sh
```

### ingest.sh

This script ingests data from the specified directory into the document and image vector stores. It uses the ingest.py Python script to perform the actual ingestion process.

Usage: 
```bash
./scripts/ingest.sh -c <clean_flag> -p <ingest_path>
```

Option flags:
> **-c <clean_flag>** - Set to True to clean the databases before ingesting new data. Defaults to False. \
> **-p <ingest_path>** - Specifies the path to the directory containing the files to ingest. Defaults to ./filebox/.

Examples: 
```bash
# Ingest data from the ./filebox directory without clearing the database beforehand
./scripts/ingest.sh
```

```bash
# Ingest data from the ./data directory and clean databases beforehand
./scripts/ingest.sh -c True -p ./data/
```

### cleandbs.sh

This script removes all data from the document and image vector stores, effectively resetting them to an empty state.

Usage:
```bash 
./scripts/cleandbs.sh
```

### common_envs.sh

Used by runserver.sh and ingest.sh to set the common environment variables
You generally don't need to run it directly.

## REST Services API Reference

The SpeakeasyGPT project exposes a REST API using Flask, allowing you to interact with its functionalities through JSON requests and responses. The API endpoints are defined in `serve.py`, and the JSON schemas for input and output are defined using the Marshmallow library.

### Using the API

You can interact with the API using tools like curl, Postman, or any HTTP client library in your preferred programming language.

### Example Workflow

1. A user sends a prompt to the `/query` endpoint.
2. The application uses the appropriate LLM factory to create an LLM instance.
3. The LLM processes the prompt and generates a response.
4. The response is returned to the user through the API.

### API Endpoints:

#### / (GET)

Ping endpoint to check if the service is alive. Returns a simple JSON response with the message "Ping."

#### query (POST)

Sends a prompt to an LLM and returns the generated response.

Request Schema:

```json
{
  "prompt": "Your prompt here",
  "llm_type": "LLM type (optional)" 
}
```

Response Schema:

```json
{
  "response": "LLM generated response"
}
```

Example usage:
```bash
curl -X POST -H "Content-Type: application/json" -d '{"prompt": "Who is the current president"  }' http://127.0.0.1:5000/query
```
```bash
curl -X POST -H "Content-Type: application/json" -d '{"prompt": "When did Andre Agassi win his last grand slam?", "llm_type":"LOCAL" }' http://127.0.0.1:5000/query
```

Sample Response:
```json
{"response":"Andre Agassi won his last grand slam in 2003. He won the Australian Open in 2003, which was his eighth and final grand slam title."}
```

### search_docs (POST) 

Query the document vector store using a natural language query.

Request Schema:

```json
{
  "prompt": "Your prompt here",
  "llm_type": "LLM type (optional)" 
}
```

Response Schema:

```json
{
  "response": "LLM generated response"
}
```

### search_images (POST) 

Query the image vector store using a natural language query.

Request Schema:

```json
{
  "prompt": "Your prompt here",
  "llm_type": "LLM type (optional)" 
}
```

Response Schema:

```json
{
  "response": "LLM generated response"
}
```

### run_agent (POST)

Runs a zero-shot or few-shot agent to complete a task.

Request Schema:

```json
{
  "prompt": "Your instructions for the agent"
}
```

Response Schema:

```json
{
  "response": "Agent's response or result"
}
```

Example usage:
```bash
curl -X POST -H "Content-Type: application/json" -d '{"prompt": "Who is the current president"  }' http://127.0.0.1:5000/run_agent
```

### run_convo_agent (POST)

Runs a conversational agent that maintains context between interactions.

Request Schema:

```json
{
  "prompt": "Your message or query for the agent"
}
```

Response Schema:

```json
{
  "response": "Agent's response or result"
}
```

Example usage:
```bash
curl -X POST -H "Content-Type: application/json" -d '{"prompt": "Who is the current president"  }' http://127.0.0.1:5000/run_convo_agent
```

### authUser (POST)

Authenticates a user with username and password.

Request Schema:

```json
{
  "username": "Username",
  "password": "Password"
}
```

Response Schema:

```json
{
  "username": "Username",
  "fullname": "Full Name",
  "gender": "Gender",
  "orientation": "Orientation",
  "dob": "Date of Birth",
  "email": "Email Address"
}
```

### accountSettings (GET)

Retrieves account settings for a user.

Request Parameters:
* **username** - Username of the user.

Response Schema:

```json
{
  "username": "Username",
  "fullname": "Full Name",
  "gender": "Gender",
  "orientation": "Orientation",
  "dob": "Date of Birth",
  "email": "Email Address"
}
```

### accountSettings (POST)

Updates account settings for a user.

Request Schema:

```json
{
  "username": "Username",
  "password": "Password (optional)",
  "fullname": "Full Name (optional)",
  "gender": "Gender (optional)",
  "orientation": "Orientation (optional)",
  "dob": "Date of Birth (optional)",
  "email": "Email Address (optional)"
}
```


## Tools for Agents

* **Instruct_LLM:** Use this tool to when instructed to generate content (like writing an email/letter/prompt), or for searching for information online.
* **EvaluateExpression:** Use this tool only for any mathematical calculations.
* **Preview_Image:** Use this tool to when instructed to preview image or a picture. The 'Action Input:' for this tool should be prompt optimized for stable diffusion
* **Document_Query:** Use this tool only to query about documents stored LOCALLY
* **Image_Query:** Use this tool to only query the captions of images stored LOCALLY
* **Search_Image:** Use this tool when asked to find or search for an image
* **PAL_Logic:** Use this tool only for programming logic to determine solution. DO NOT do the math yourself, or change the question wording when defining the 'Action Input:' for this tool.
* **Run_Code:** Use this tool when asked to create and execute code / script. Pass the original request as the 'Action Input:' for this tool


## Environment Variable

Ensure that you set the necessary environment variables before running the SpeakeasyGPT application.

### Setting Environment Variables:

You can set environment variables in various ways depending on your operating system and deployment environment:
* Terminal: Use the export command in your terminal (e.g., export OPENAI_API_KEY=your_api_key).
* .env file: Create a .env file in your project directory and store the variables in the format VARIABLE_NAME=value. You can use a library like python-dotenv to load these variables into your application.
* Deployment Platform: Most cloud platforms provide ways to set environment variables for your application during deployment.

### Application Configuration:
```
FLASK_APP=Specifies the entry point Python file for the Flask application. Usually set to serve.py.

ENABLE_DEBUG=Enables or disables Flask`s debug mode. Set to True for development and debugging, False for production.

DOCS_PERSIST_DIRECTORY=Path to the directory where the document vector store is persisted.

IMAGES_PERSIST_DIRECTORY=Path to the directory where the image vector store is persisted.

FACTORY_TYPE=Specifies the type of LLM factory to use (e.g., OPENAI, HUGGINGFACE, GOOGLE, BARD).

SD_URL=URL of the Stable Diffusion API endpoint for image generation (e.g., http://127.0.0.1:7860).

SD_STEPS=Number of steps for Stable Diffusion image generation.

IMAGE_OUTPUT_FILENAME=Path to the output file for generated images.

GOOGLE_LLM_API_TIMEOUT=Timeout in seconds for Google LLM API requests.

BARD_EXPERIMENTAL=Enables or disables experimental features of the Bard API (e.g., image and code generation).
```
### API Keys and Credentials:
```
OPENAI_API_KEY=API key for OpenAI.

HUGGINGFACEHUB_API_TOKEN=API token for Hugging Face Hub.

_BARD_API_KEY=API key for Bard (Google AI).

GOOGLE_APPLICATION_CREDENTIALS=Path to the JSON file containing Google Cloud service account 
credentials.
```
### Other:
```
HF_HOME=Path to the directory where Hugging Face model files are cached.
```

## Code Structure and Functionality:

The project is organized into several modules and files:
* **ingest.py:** Ingests documents and images into vector stores.
* **serve.py:** Implements the Flask REST API and defines routes for various functionalities.
* **llmfactory.py:** Defines factories for creating LLM instances from different providers (e.g., OpenAI, HuggingFace, Google AI).
* **customagents.py:** Implements custom LangChain agents for task execution.
* **indexes.py:** Handles the creation and loading of vector stores for documents and images.
* **customloaders.py:** Defines custom loaders for handling specific file types (e.g., image captions).
* **vectorstore_tools.py:** Implements tools for querying vector stores.
* **image_tools.py:** Implements tools for generating and searching images.
* **custom_tools.py:** Defines custom tools for various tasks (e.g., math calculations, code execution).
* **models.py:** Defines database models for user accounts and settings using SQLAlchemy.

### Marshmallow Schemas:

The project uses Marshmallow schemas to define the structure and validation rules for JSON input and output. The schemas are defined in `serve.py` and ensure that the data exchanged through the API conforms to the expected format.

Example:

```python
# Request Schema for `/query` endpoint:
class RequestSchema(Schema):
  prompt = fields.Str(required=True)
  llm_type = EnumField(LLMType, required=False, missing=None, by_value=True, allow_none=True)

# Response Schema for /query endpoint:
class ResponseSchema(Schema):
  response = fields.Str()
  image = fields.Str()
```

## Running SpeakeasyGPT Unit Tests

The unit tests are written using the **unittest** framework
Here's how to run the unit tests for the SpeakeasyGPT project:

```bash
python -m unittest tests/*.py
```

## Deployment

### Google Cloud App Engine

Here's how to deploy the SpeakeasyGPT project to Google App Engine using the provided app.yaml file.

#### Prerequisites:
* Google Cloud Project: Ensure you have a Google Cloud project set up with billing enabled.
* gcloud CLI: Install and configure the Google Cloud SDK on your local machine.
* Configure app.yaml:
> * Scaling: Adjust the manual_scaling section if you need to change the number of instances.
> * Resources: Modify the resources section to specify the desired CPU, memory, and disk size for your application.
> * Environment Variables: Set the necessary environment variables in the env_variables section of app.yaml. This includes API keys, directory paths, and other configurations.
```
env_variables:
  FLASK_APP: serve.py
  ENABLE_DEBUG: False
  OPENAI_API_KEY: your_openai_api_key
  HUGGINGFACEHUB_API_TOKEN: your_huggingface_api_token
  _BARD_API_KEY: your_bard_api_key
  GOOGLE_APPLICATION_CREDENTIALS: path/to/your/credentials.json
  DOCS_PERSIST_DIRECTORY: db_docs/
  IMAGES_PERSIST_DIRECTORY: db_images/
```

#### Deploy to App Engine:

Run the following command to deploy the application:

```bash
gcloud app deploy
```

Once the deployment is complete, you can access your application using the provided URL in the deployment output.

#### Troubleshooting:
If you encounter errors during deployment, check the deployment logs for details.
Ensure your environment variables are set correctly and that your service account has the necessary permissions.


### Building and deploying docker image to a GCP Compute Engine Instance

Here's how to build and deploy SpeakeasyGPT to Google Cloud Platform using the provided Dockerfile and a VM instance with a persistent disk

#### Prerequisites:
* Google Cloud Project: Ensure you have a Google Cloud project set up with billing enabled.
* gcloud CLI: Install and configure the Google Cloud SDK on your local machine.

#### 1. Build the Docker Image with gcloud build:

Open a terminal and navigate to the project's root directory.
Build the Docker image using gcloud build and push it to Google Container Registry (GCR):

```bash
gcloud builds submit --tag gcr.io/[YOUR_PROJECT_ID]/speakeasygpt .
```

For deploying and running the build simply, but without a persisten disk attached, use the `cloud run` command. E.g.:

```bash
gcloud run deploy --image gcr.io/[YOUR_PROJECT_ID]/speakeasygpt --platform managed
```

Otherwise, read on.

#### 2. Create a VM Instance:

* Go to the Google Cloud Console and navigate to the "Compute Engine" -> "VM instances" section.
* Click "Create Instance".
* Choose a machine type and configuration that meets your requirements.
* Under "Boot disk", select "Change" and choose an operating system image that supports Docker (e.g., Ubuntu).
* Under "Disks", click "Add new disk" and create a persistent disk with sufficient size for your data (e.g., 100 GB). Name this disk appropriately (e.g., "speakeasydisk-2").
* Under "Firewall", ensure that the "Allow HTTP traffic" and "Allow HTTPS traffic" options are checked.
* Click "Create" to create the VM instance. Name this instance appropriately (e.g., "speakeasygpt-vm").

#### 3. Update VM Instance with Container Image and Mount Disk:

Run the following command to update the VM instance with the container image and mount the persistent disk:
```bash
gcloud compute instances update-container speakeasygpt-vm \
--container-image gcr.io/[YOUR_PROJECT_ID]/speakeasygpt:latest \
--container-mount-disk mount-path="/mnt/disk",name="speakeasydisk-2",mode=rw
```

#### 4. Expose Port and Start Container (if not automatically started):

If the container doesn't start automatically, you might need to expose the port and start it manually:

```bash
gcloud compute instances add-tags speakeasygpt-vm --tags http-server
gcloud compute instances update-container speakeasygpt-vm --port 8080
gcloud compute instances start-container speakeasygpt-vm
```

#### 5. Access the Application:

Get the external IP address of your VM instance from the Cloud Console.
Access the application using the following URL: http://[EXTERNAL_IP_ADDRESS]:8080


### Building and deploying docker image to a GCP Compute Engine Instance (Alternate approach using docker)

#### Prerequisites:
* Same as previous approach, apart from:
* Docker: Install Docker on your local machine.

#### 1. Build the Docker Image:

Build the Docker image using the following command:

```bash
docker build -t speakeasygpt .
```

#### 2. Push the Image to Google Container Registry (GCR):

Tag the image with your GCP project ID and a name:

```bash
docker tag speakeasygpt gcr.io/[YOUR_PROJECT_ID]/speakeasygpt
```

Push the image to GCR:

```bash
docker push gcr.io/[YOUR_PROJECT_ID]/speakeasygpt
```

#### 3. Create a VM Instance:

* Same as previous approach

#### 4. Connect to the VM Instance:

Once the instance is created, click on the SSH button to connect to it.

#### 5. Install Docker and Mount Persistent Disk:

In the SSH terminal, update the package lists and install Docker:

```bash
sudo apt update
sudo apt install docker.io
```

Mount the persistent disk:

```bash
# Get the device name of the persistent disk (e.g., /dev/sdb)
lsblk

# Create a mount point
sudo mkdir /mnt/disk

# Mount the disk
sudo mount /dev/sdb /mnt/disk
```

Make the mount permanent (optional)
Edit /etc/fstab and add a line like this:

```/dev/sdb /mnt/disk ext4 defaults 0 0```

#### 6. Pull and Run the Docker Image:

Pull the image from GCR:

```bash
sudo docker pull gcr.io/[YOUR_PROJECT_ID]/speakeasygpt
```

Run the image, mapping the persistent disk to the appropriate directories inside the container:

```bash
sudo docker run -d \
  -p 8080:8080 \
  -v /mnt/disk/db_docs:/app/db_docs \
  -v /mnt/disk/db_images:/app/db_images \
  -v /mnt/disk/hf_cache:/app/hf_cache \
  gcr.io/[YOUR_PROJECT_ID]/speakeasygpt
```

#### 7. Access the Application:

Same as previous approach
