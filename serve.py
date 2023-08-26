"""
Flask rest service implementation
"""

import langchain
from flask import Flask, jsonify, request
from marshmallow import Schema, fields, post_load
from marshmallow_enum import EnumField
from langchain.chains import RetrievalQA
from speakeasy.llmfactory import LLMType, init_factory, init_factory_from_type
from speakeasy.customagents import init_agent, init_conversational_agent
from speakeasy.indexes import load_document_db, load_image_db
from env_params import parse_environment_variables

# Flask app
app = Flask(__name__)

# Collect environment variables or defaults
env_config = parse_environment_variables()

# Set langchain's debug flag
langchain.debug = env_config['debug']

# Initialise the factory
factory = init_factory(env_config)

# Setup the VectorStore DB
doc_db = load_document_db(factory, persist_dir=env_config['docs_persist_directory'])
img_db = load_image_db(factory, persist_dir=env_config['images_persist_directory'])

# Setup the Agents
agent = init_agent(factory, doc_db, img_db, max_iterations=1)
convo_agent = init_conversational_agent(factory, doc_db, img_db)


# ------- Serializing / Deserializing ------------------

class Request():
    """ Request object and handlers """
    def __init__(self, prompt, llm_type = None):
        self.prompt = prompt
        self.llm_type = llm_type # if None should then use whichever factory instantiated at start

class RequestSchema(Schema):
    """ Request schema """
    prompt = fields.Str(required=True)
    llm_type = EnumField(LLMType,
        required=False,
        missing=None,
        by_value=True,
        allow_none=True) # Not sure exactly which did the trick
    @post_load()
    def make_request(self, data):
        """ Instantiate the Request object """
        return Request(**data)

def deserialize_request(req_json):
    """ Deserielze the request json """
    return RequestSchema(partial=True).load(req_json.get_json())

# Response class and hanlders
class Response():
    """ Response object """
    def __init__(self, resp, image = None):
        self.response = resp
        if not image is None:
            self.image = image

class ResponseSchema(Schema):
    """ The response schema"""
    response = fields.Str()
    image = fields.Str()

def format_response(respstr):
    """ format the response json """
    schema = ResponseSchema(many=False, partial=True)
    respobj = Response(respstr)

    # Basic image handling
    image_tag = 'BASE64ENCODED:'
    if respstr.startswith(imageTag):
        respobj.response = 'Here is the image.'
        respobj.image = respstr[len(image_tag):] # Strip BASE64ENCODED:

    return schema.dump(respobj)

# Setup factory dict with defaults
# Retrieve from a dict if already instansitated
factory_dict = {
        'LLMType.GOOGLE': None,
        'LLMType.BARD': None,
        'LLMType.HUGGINGFACE': None,
        'LLMType.LOCAL': None,
        'LLMType.OPENAI': None
}
factory_dict[str(LLMType(env_config['factory_type']))] = factory

def choose_factory(req):
    """ Not applicable for agents since these were initialised at start up """
    if req.llm_type is None:
        return factory
    if not factory_dict[str(req.llm_type)] is None:
        return factory_dict[str(req.llm_type)]

    # Instansiate new factory since not found in dict
    factory_dict[str(req.llm_type)] = init_factory_from_type(req.llm_type, env_config)
    return factory_dict[str(req.llm_type)]

# --------- Define Routes ------------------------------
# Consider default exception / error handler

@app.route("/")
def ping():
    """ endpoint to verify services are alive """
    return format_response("Ping.")

@app.route("/query", methods=['GET', 'POST'])
def query_llm():
    """ endpoint for basic LLM prompt """
    try:
        req = deserialize_request(request)
        fact = choose_factory(req)
        print(f"Factory: {fact}")
        return format_response(fact.llm(req.prompt))
    except Exception as exc:
        print(f"Caught exception: {exc}")
        return format_response('Oops sorry an error occured.')

@app.route("/search_docs", methods=['POST'])
def search_docs():
    """ endpoint for querying the document vectorstore """
    try:
        req = deserialize_request(request)
        fact = choose_factory(req)
        chain = RetrievalQA.from_chain_type(llm=fact.llm,
            chain_type="stuff",
            retriever=doc_db.as_retriever(search_kwargs={"k":factory.max_k}),
            input_key="question",
            return_source_documents=True)
        return format_response(chain(req.prompt)['result'])
    except Exception as exc:
        print(f"Caught exception: {exc}")
        return format_response('Oops sorry an error occured.')

@app.route("/search_images", methods=['POST'])
def search_images():
    """ endpoint for querying the image vectorstore """
    try:
        req = deserialize_request(request)
        fact = choose_factory(req)
        chain = RetrievalQA.from_chain_type(llm=fact.llm,
            chain_type="stuff",
            retriever=img_db.as_retriever(search_kwargs={"k":factory.max_k}),
            input_key="question",
            return_source_documents=True)
        return format_response(chain(req.prompt)['result'])
    except Exception as exc:
        print(f"Caught exception: {exc}")
        return format_response('Oops sorry an error occured.')

@app.route("/run_agent", methods=['POST'])
def run_agent():
    """ endpoint for zero / few shot agent """
    try:
        req = deserialize_request(request)
        return format_response(agent.run(req.prompt))
    except Exception as exc:
        print(f"Caught exception: {exc}")
        return format_response('Oops sorry an error occured.')

@app.route("/run_convo_agent", methods=['POST'])
def run_convo_agent():
    """ endpoint for conversational agent"""
    try:
        req = deserialize_request(request)
        return format_response(convo_agent.run(input=req.prompt))
    except Exception as exc:
        print(f"Caught exception: {exc}")
        return format_response('Oops sorry an error occured.')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
