import torch
from typing import List, Optional
from enum import Enum
from langchain import HuggingFaceHub
from langchain.llms.base import LLM
from langchain.llms import OpenAI, HuggingFacePipeline, VertexAI
from langchain.chat_models import ChatOpenAI, ChatVertexAI
from langchain.embeddings import HuggingFaceEmbeddings, VertexAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from bardapi import Bard
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from langchain.schema import AIMessage, HumanMessage, SystemMessage

#TODO Exploring ChatBard and session/conversation management. Also ChatOpenAI
#TODO Exploring the Callback functionality - E.g.: StdOut Callback Streaming
#TODO Exploring the Async functionality
#TODO Add Factory for using text-generation, AutoModelForCausalLM as opposed to text2text-generation, Seq2SeqLLMs
#TODO Add Factory for Using LLama specific classes when dealing with Llama, GPT4All models 
#TODO How to get load_in_8bit=True working in windows? Can it only work with quantized models?

# Setup the LLM Types
class LLMType(Enum):
    LOCAL = "LOCAL"
    HUGGINGFACE = "HUGGINGFACE"
    OPENAI = "OPENAI"
    GOOGLE = "GOOGLE"
    BARD = "BARD"

# Initialise a LLM Factory
def init_factory_from_type(llm_type, env_config):
    match llm_type:
        case LLMType.LOCAL:
            fa = LocalLLMFactory(env_config = env_config, device_id = "cpu") # Make configurable via environment vars
        case LLMType.HUGGINGFACE:
            fa = HuggingFaceFactory(env_config = env_config)
        case LLMType.OPENAI:
            fa = OpenAIFactory(env_config = env_config)
        case LLMType.GOOGLE:
            fa = GoogleLLMFactory(env_config = env_config)
        case LLMType.BARD:
            fa = BardLLMFactory(env_config = env_config, api_timeout = env_config['google_llm_api_timeout'])
        case _:
            fa = LocalLLMFactory(env_config = env_config)
    return fa

def init_factory(env_config):
    return init_factory_from_type(LLMType(env_config['factory_type']), env_config)


# ----------------------------------------------------------------------------------

# Abstract factory for LLMs
class LLMFactory(object):
    temperature = 0
    model_name = ""
    max_length = 512
    max_k = 1

    def __init__(self, model_name = "", temperature = 0, max_length = 512, max_k = 1, env_config = {}):
        self.env_config = env_config
        self.model_name = model_name 
        self.temperature = temperature
        self.max_length = max_length
        self.max_k = max_k
        self.construct_llm()

    def __repr__(self):
        return '<LLMFactory(model_name={self.model_name})>'.format(self=self)

    def construct_llm(self):
        """Construct the LLM component for the given model""" 
        raise NotImplementedError("Must be implemented in subclass")

    # Added to attempt add support for Bardapi Experimental Features
    def image_support(self) -> bool:
        return False

    def code_support(self) -> bool:
        return False
    
# Abstract factory for LLM components and embeddings
class LLMAndEmbeddingsFactory(LLMFactory):
    def __init__(self, 
            model_name = LLMFactory.model_name, 
            temperature = LLMFactory.temperature, 
            max_length = LLMFactory.max_length, 
            max_k = LLMFactory.max_k, 
            embedding_model_name = 'intfloat/e5-large-v2', 
            embedding_device_id = 'cuda', 
            env_config = {}
        ):
        super(LLMAndEmbeddingsFactory, self).__init__(model_name, temperature, max_length, max_k, env_config = env_config)
        self.embedding_model_name = embedding_model_name
        self.embedding_device_id = embedding_device_id
        self.construct_embeddings()

    def __repr__(self):
        return '<LLMFAndEmbeddingFactory(model_name={self.model_name}, embedding_model_name={self.embedding_model_name})>'.format(self=self)

    def construct_embeddings(self):
        """Construct the relevant Embeddings"""
        self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model_name,  model_kwargs={"device": self.embedding_device_id})

# ------------------------------------------------------------------------------------------------------------

# Factory for OpenAI LLM components
class OpenAIFactory(LLMAndEmbeddingsFactory):
    max_k = 4

    def __init__(self, env_config = {}):
        super(OpenAIFactory, self).__init__(max_k = OpenAIFactory.max_k, env_config = env_config)

    def __repr__(self):
        return '<OpenAIFactory(model_name=NA)>'.format(self=self)

    def construct_llm(self):    
        self.llm = OpenAI()   # ChatOpenAI()
    
    def construct_embeddings(self):
        self.embeddings = OpenAIEmbeddings()

# Factory for HuggingFace Hub LLM components
class HuggingFaceFactory(LLMAndEmbeddingsFactory):
    temperature = 0.5
    model_name = "google/flan-t5-xxl"
    
    def __init__(self, env_config = {}):
        super(HuggingFaceFactory, self).__init__(
                model_name = HuggingFaceFactory.model_name, 
                temperature = HuggingFaceFactory.temperature, 
                env_config = env_config
            )
    
    def __repr__(self):
        return '<HuggingFaceFactory(model_name={self.model_name}), embedding_model_name={self.embedding_model_name}>'.format(self=self)

    def construct_llm(self):
        self.llm=HuggingFaceHub(repo_id=self.model_name, model_kwargs={"temperature":self.temperature, "max_length":self.max_length})
    
# Factory for LLM components for running on local hardware
# Seq2Seq Models: "google/flan-t5-large", "lmsys/fastchat-t5-3b-v1.0"
class LocalLLMFactory(LLMAndEmbeddingsFactory):

    def __init__(self, env_config = {}, model_name = "google/flan-t5-large", device_id = "cpu", load_in_8bit = False):
        self.device_id = device_id
        self.load_in_8bit = load_in_8bit
        super(LocalLLMFactory, self).__init__(env_config = env_config, model_name = model_name)

    def __repr__(self):
        return '<LocalLLMFactory(model_name={self.model_name}, embedding_model_name={self.embedding_model_name})>'.format(self=self)

    def construct_llm_from_id(self):
        # Handle device mapping
        self.llm = HuggingFacePipeline.from_model_id(
                model_id=self.model_name, 
                task="text2text-generation", 
                device=-1, 
                model_kwargs={"temperature":self.temperature, "max_length":self.max_length}
            )

    def construct_llm(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.device_id == 'cuda':
            model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name, 
                load_in_8bit=self.load_in_8bit,
                torch_dtype=torch.float16, 
                low_cpu_mem_usage=True, 
                device_map="auto",
                trust_remote_code=True
            )
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            model.to(torch.device(self.device_id))
        
        pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=self.max_length,
                temperature=self.temperature, top_p=0.95, repetition_penalty=1.15
            )
        self.llm = HuggingFacePipeline(pipeline=pipe)

    # Uncomment if want to use the instructor embeddings for local factory, instead of the hugging face embeddings
    #def construct_embeddings(self):
        #self.embedding_model_name="hkunlp/instructor-xl"
        #self.embeddings = HuggingFaceInstructEmbeddings(model_name=self.embedding_model_name, model_kwargs={"device": self.embeddings_device_id}) 

# Factory for Google LLM components, I.e.: Vertex AI
# Models: "text-bison@001" , "code-bison@001" , "chat-bison@001"
class GoogleLLMFactory(LLMAndEmbeddingsFactory):

    def __init__(self, env_config = {}, model_name = "code-bison@001", max_k = 4):
        super(GoogleLLMFactory, self).__init__(max_k = max_k, model_name = model_name, env_config = env_config)

    def __repr__(self):
        return '<GoogleLLMFactory(model_name={self.model_name})>'.format(self=self)

    def construct_llm(self):
        self.llm = VertexAI( # ChatVertexAI
                model_name = self.model_name,
                max_output_tokens = self.max_length,
                temperature = self.temperature,
                top_p=0.8,
                top_k=40,
                verbose=True
            )

    def construct_embeddings(self):
        self.embeddings = VertexAIEmbeddings()

# Factory for Bard
# Assumes API key is already set as environment variable, for example using os.environ['_BARD_API_KEY']="xxxxxxxx"
# Set environment variable BARD_EXPERIMENTAL to True for using Bardapi Experimental features (assumes using github branch of Bardapi)
class BardLLMFactory(LLMAndEmbeddingsFactory):
    def __init__(self, env_config = {}, api_timeout = 30):
        self.api_timeout = api_timeout
        super(BardLLMFactory, self).__init__(env_config = env_config)

    def __repr__(self):
        return '<BardLLMFactory(model_name=Bard), embedding_model_name={self.embedding_model_name}>'.format(self=self)

    def construct_llm(self):
        self.llm = BardLLMWrapper(timeout = self.api_timeout) 

    def image_support(self) -> bool:
        return self.env_config['bard_experimental']

    def code_support(self) -> bool:
        return self.env_config['bard_experimental']


# ----------------------- LLM Wrappers ----------------------------------------------------------

# Simple wrapper which operates similar to langchain LLM, as Bard is Bardapi is not supported by langchain
# Attempted add support for Bard API Experimental fetures such as: retrieveing images, links and code, as well as running code.
class BardLLMWrapper(LLM):
    googlellm : Bard = None
    code_runner : Bard = None
    last_response_dict = { 'content': '', 'images':[], 'links':[], 'code':'' } 
    response_element = 'content'

    def __init__(self, timeout = None):
        super(BardLLMWrapper, self).__init__()
        self.googlellm = Bard(timeout = timeout)
        self.code_runner = Bard(timeout = timeout, run_code=True)

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager = None,
    ) -> str:
        #print(f"\nInput Prompt::::::::::::::::::::::::\n{prompt}") # For debugging
        response_dict = self.googlellm.get_answer(prompt)
        self.last_response_dict = response_dict
        full_resp =  response_dict[self.response_element]
        resp_frag = full_resp.replace('**', '') # Replace the chars Bard sometimes adds
        if stop != None:
            if not isinstance(stop, List):
                stop = stop()
            for s in stop:
                tmp = full_resp.split(s) # Hacked stop sequence handling
                if len(tmp) > 1:
                    resp_frag = tmp[0]
        #print(f"\nFull Response Dict::::::::::::::::::::::\n{response_dict}")
        #print(f"\nFull Response until stop sequence:::::::::\n{resp_frag}") # For debugging
        self.response_element = 'content' # Set the next response back to content
        return resp_frag

    @property
    def _llm_type(self) -> str:
        return "google-bard"

    # Methods for interacting with Bard API Experimental features

    def get_last_images(self) -> List:
        return self.last_response_dict['images']

    def get_last_image_links(self) -> List:
        return self.last_response_dict['links']

    def get_last_code_fragment(self) -> str:
        return self.last_response_dict['code']

    def set_next_response_for_images(self):
        self.response_element = 'images'

    def set_next_response_for_links(self):
        self.response_element = 'links'

    def set_next_response_for_code(self):
        self.response_element = 'code'

    def run_code(self, prompt):
        resp = self.code_runner.get_answer(prompt) # How to get the output of the code execution?
        return resp

