import torch
from typing import List, Optional
from enum import Enum
from langchain import HuggingFaceHub
from langchain.llms.base import LLM
from langchain.llms import OpenAI, HuggingFacePipeline
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import HuggingFaceEmbeddings, HuggingFaceInstructEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from bardapi import Bard

#TODO Exploring Bard specific - How to handle temperature? Also trying out summerizers, translators, retrievers
#TODO Exploring ChatBard by using the dev branch of bardapi, Also ChatOpenAI
#TODO Exploring the Callback functionality - E.g.: StdOut Callback Stream
#TODO Add Factory for using text-generation, CausalLLMs as opposed to text2text-generation, Seq2SeqLLMs
#TODO Add Factory for Using LLama specific classes when dealing with Llama, GPT4All models 
#TODO How to get load_in_8bit=True working? Can it only work with quantized models? Try different version of bitsandbytes?

# Setup the LLM Types
class LLMType(Enum):
    LOCAL = "LOCAL"
    HUGGINGFACE = "HUGGINGFACE"
    OPENAI = "OPENAI"
    GOOGLE = "GOOGLE"

# Initialise a LLM Factory
def init_factory_from_type(llm_type, env_config):
    match llm_type:
        case LLMType.LOCAL:
            fa = LocalLLMFactory(env_config = env_config)
        case LLMType.HUGGINGFACE:
            fa = HuggingFaceFactory(env_config = env_config)
        case LLMType.OPENAI:
            fa = oi_factory = OpenAIFactory(env_config = env_config)
        case LLMType.GOOGLE:
            fa = GoogleLLMFactory(env_config = env_config, api_timeout = env_config['google_llm_api_timeout'])
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
    device_id = "cpu" # Device for Local LLM - can be 'cpu' or 'cuda'
    max_k = 1
    env_config = {}

    def __init__(self, model_name, temperature = 0, max_length = 512, device_id = 'cpu', max_k = 1, env_config = {}):
        self.env_config = env_config
        self.model_name = model_name 
        self.temperature = temperature
        self.max_length = max_length
        self.max_k = max_k
        self.device_id = device_id
        self.llm=None
        self.construct_llm()

    def __repr__(self):
        return '<LLMFactory(model_name={self.model_name})>'.format(self=self)

    def construct_llm(self):
        """Construct the LLM component for the given model""" 
        raise NotImplementedError("Must be implemented in subclass")
    
# Abstract factory for LLM components and embeddings
class LLMAndEmbeddingsFactory(LLMFactory):
    embedding_model_name = "intfloat/e5-large-v2"
    embedding_device_id = "cuda" 

    def __init__(self, 
            model_name = LLMFactory.model_name, 
            temperature = LLMFactory.temperature, 
            max_length = LLMFactory.max_length, 
            device_id = LLMFactory.device_id, 
            max_k = LLMFactory.max_k, 
            embedding_model_name = 'intfloat/e5-large-v2', 
            embedding_device_id = 'cuda', 
            env_config = {}
        ):
        super(LLMAndEmbeddingsFactory, self).__init__(model_name, temperature, max_length, device_id, max_k, env_config = env_config)
        self.embedding_model_name = embedding_model_name
        self.embedding_device_id = embedding_device_id
        self.embeddings=None
        self.construct_embeddings()

    def __repr__(self):
        return '<LLMFAndEmbeddingFactory(model_name={self.model_name, embedding_model_name={self.embedding_model_name}})>'.format(self=self)

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
        return '<HuggingFaceFactory(model_name={self.model_name}})>'.format(self=self)

    def construct_llm(self):
        self.llm=HuggingFaceHub(repo_id=self.model_name, model_kwargs={"temperature":self.temperature, "max_length":self.max_length})
    
    #def construct_embeddings(self):
        #self.embeddings = HuggingFaceEmbeddings() 

# Factory for LLM components for running on local hardware
class LocalLLMFactory(LLMAndEmbeddingsFactory):
    model_name = "google/flan-t5-large"
    #model_name = "lmsys/fastchat-t5-3b-v1.0"
    #embedding_model_name="hkunlp/instructor-xl"

    def __init__(self, env_config = {}):
        super(LocalLLMFactory, self).__init__(
                model_name = LocalLLMFactory.model_name, 
                device_id = LocalLLMFactory.device_id, 
                env_config = env_config
            )

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
        model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name, 
                #load_in_8bit=True, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="auto",
                )
        model.to(torch.device(self.device_id))
        pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=self.max_length,
                temperature=self.temperature, top_p=0.95, repetition_penalty=1.15
                )
        self.llm = HuggingFacePipeline(pipeline=pipe)

    #def construct_embeddings(self):
        #self.embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl", model_kwargs={"device": self.device_id}) 

# Factory for Google LLM APIs. E.g. Bard
class GoogleLLMFactory(LLMAndEmbeddingsFactory):
    api_timeout : int

    def __init__(self, env_config = {}, api_timeout = 30):
        #os.environ['_BARD_API_KEY']="xxxxxxxx"
        self.api_timeout = api_timeout
        super(GoogleLLMFactory, self).__init__(env_config = env_config)

    def __repr__(self):
        return '<GoogleLLMFactory(model_name=Bard), embedding_model_name={self.embedding_model_name}>'.format(self=self)

    def construct_llm(self):
        self.llm = GoogleLLMWrapper(timeout = self.api_timeout) 


# ----------------------- LLM Wrappers ----------------------------------------------------------

# Simple wrapper which operates similar to langchain LLM
class GoogleLLMWrapper(LLM):
    googlellm : Bard = None

    def __init__(self, timeout = None):
        super(GoogleLLMWrapper, self).__init__()
        self.googlellm = Bard(timeout = timeout)

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager = None,
    ) -> str:
        #print(f"\nInput Prompt:::::::::::\n{prompt}") # For debugging
        full_resp =  self.googlellm.get_answer(prompt)['content']
        resp_frag = full_resp.replace('**', '') # Replace the chars Bard sometimes adds
        if stop != None:
            if not isinstance(stop, List):
                stop = stop()
            for s in stop:
                tmp = full_resp.split(s) # Hacked stop sequence handling
                if len(tmp) > 1:
                    resp_frag = tmp[0]
        #print(f"\nFull Response until stop sequence:::::::::\n{resp_frag}") # For debugging
        return resp_frag

    @property
    def _llm_type(self) -> str:
        return "google-bard"

