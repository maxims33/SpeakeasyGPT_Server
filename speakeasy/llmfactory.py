"""
Module for various types of LLM Factory
#TODO Exploring ChatBard and session/conversation management. Also ChatOpenAI
#TODO Exploring the Callback functionality - E.g.: StdOut Callback Streaming
#TODO Exploring the Async functionality
#TODO Add Factory for using text-generation, AutoModelForCausalLM
#TODO Add Factory for Using LLama specific classes when dealing with Llama, GPT4All models
#TODO How to get load_in_8bit=True working in windows? Can it only work with quantized models?
"""
from typing import List, Optional
from enum import Enum
import torch
import os
from bardapi import Bard
from langchain_community.llms import HuggingFaceHub
from langchain.llms.base import LLM
from langchain_community.llms import OpenAI, HuggingFacePipeline
from langchain_google_vertexai import VertexAI, VertexAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
#from langchain.chat_models import ChatVertexAI
from langchain_community.embeddings import (
    HuggingFaceEmbeddings,
    #    HuggingFaceInstructEmbeddings,
)
from langchain_community.embeddings.openai import OpenAIEmbeddings
#from langchain.schema import AIMessage, HumanMessage, SystemMessage
from transformers import (
    pipeline,
    AutoTokenizer,
    #    AutoModelForCausalLM
    AutoModelForSeq2SeqLM)
#from langchain.prompts.chat import (
#    ChatPromptTemplate,
#    SystemMessagePromptTemplate,
#    AIMessagePromptTemplate,
#    HumanMessagePromptTemplate
#)


class LLMType(Enum):
  """ Setup the LLM Types """
  LOCAL = "LOCAL"
  HUGGINGFACE = "HUGGINGFACE"
  OPENAI = "OPENAI"
  GOOGLE = "GOOGLE"
  GOOGLEAISTUDIO = "GOOGLEAISTUDIO"
  BARD = "BARD"


def init_factory_from_type(llm_type, env_config):
  """ Initialise a LLM Factory for a specific type """
  match llm_type:
    case LLMType.LOCAL:
      fact = LocalLLMFactory(env_config=env_config, device_id="cpu")
    case LLMType.HUGGINGFACE:
      fact = HuggingFaceFactory(env_config=env_config)
    case LLMType.OPENAI:
      fact = OpenAIFactory(env_config=env_config)
    case LLMType.GOOGLE:
      fact = GoogleLLMFactory(env_config=env_config)
    case LLMType.GOOGLEAISTUDIO:
      fact = GoogleAIstudioFactory(env_config=env_config)
    case LLMType.BARD:
      fact = BardLLMFactory(env_config=env_config,
                            api_timeout=env_config['google_llm_api_timeout'])
    case _:
      fact = LocalLLMFactory(env_config=env_config)
  return fact


def init_factory(env_config):
  """ Initialise a LLM Factory """
  return init_factory_from_type(LLMType(env_config['factory_type']),
                                env_config)


# ----------------------------------------------------------------------------------


class LLMFactory():
  """
    Abstract factory for LLMs
    """
  temperature = 0
  model_name = ""
  max_length = 512
  max_k = 1

  def __init__(self,
               model_name="",
               temperature=0,
               max_length=512,
               max_k=1,
               env_config=None):
    self.env_config = env_config
    self.model_name = model_name
    self.temperature = temperature
    self.max_length = max_length
    self.max_k = max_k
    self.llm = None
    self.construct_llm()

  def __repr__(self):
    return f"<LLMFactory(model_name={self.model_name})>"

  def construct_llm(self):
    """Construct the LLM component for the given model"""
    raise NotImplementedError("Must be implemented in subclass")

  def image_support(self) -> bool:
    """ Added to attempt add support for Bardapi Experimental Features """
    return False

  def code_support(self) -> bool:
    """ Added to attempt add support for Bardapi Experimental Features """
    return False


class LLMAndEmbeddingsFactory(LLMFactory):
  """ Abstract factory for LLM components and embeddings """

  def __init__(self,
               model_name=LLMFactory.model_name,
               temperature=LLMFactory.temperature,
               max_length=LLMFactory.max_length,
               max_k=LLMFactory.max_k,
               embedding_model_name='intfloat/e5-large-v2',
               embedding_device_id='cuda',
               env_config=None):
    super().__init__(model_name,
                     temperature,
                     max_length,
                     max_k,
                     env_config=env_config)
    self.embedding_model_name = embedding_model_name
    self.embedding_device_id = embedding_device_id
    self.embeddings = None
    self.construct_embeddings()

  def __repr__(self):
    return f"""<LLMFAndEmbeddingFactory(model_name={self.model_name},
 embedding_model_name={self.embedding_model_name})>"""

  def construct_llm(self):
    """ Construct the LLM """
    raise NotImplementedError("Must be implemented in subclass")

  def construct_embeddings(self):
    """Construct the relevant Embeddings"""
    self.embeddings = None  #HuggingFaceEmbeddings(


#            model_name=self.embedding_model_name,
#            model_kwargs={"device": self.embedding_device_id
#        })

# -----------------------------------------------------------------------------------


class OpenAIFactory(LLMAndEmbeddingsFactory):
  """ Factory for OpenAI LLM components """
  max_k = 4

  def __init__(self, env_config=None):
    super().__init__(max_k=OpenAIFactory.max_k, env_config=env_config)

  def __repr__(self):
    return "<OpenAIFactory(model_name=NA)>"

  def construct_llm(self):
    """ LLM constructor method """
    self.llm = OpenAI()  # ChatOpenAI()

  def construct_embeddings(self):
    """ Embeddings constructor method """
    self.embeddings = OpenAIEmbeddings()


class HuggingFaceFactory(LLMAndEmbeddingsFactory):
  """ Factory for HuggingFace Hub LLM components """
  temperature = 0.5
  model_name = "google/flan-t5-xxl"

  def __init__(self, env_config=None):
    super().__init__(model_name=HuggingFaceFactory.model_name,
                     temperature=HuggingFaceFactory.temperature,
                     env_config=env_config)

  def __repr__(self):
    return f"""<HuggingFaceFactory(model_name={self.model_name}),
 embedding_model_name={self.embedding_model_name}>"""

  def construct_llm(self):
    """ LLM constructor method """
    self.llm = HuggingFaceHub(repo_id=self.model_name,
                              model_kwargs={
                                  "temperature": self.temperature,
                                  "max_length": self.max_length
                              })


# Seq2Seq Models: "google/flan-t5-large", "lmsys/fastchat-t5-3b-v1.0"
class LocalLLMFactory(LLMAndEmbeddingsFactory):
  """
    Factory for LLM components for running on local hardware
    """

  def __init__(self,
               env_config=None,
               model_name="google/flan-t5-large",
               device_id="cpu",
               load_in_8bit=False):
    self.device_id = device_id
    self.load_in_8bit = load_in_8bit
    super().__init__(env_config=env_config, model_name=model_name)

  def __repr__(self):
    return f"""<LocalLLMFactory(model_name={self.model_name},
 embedding_model_name={self.embedding_model_name})>"""

  def construct_llm_from_id(self):
    """ LLM constructor method """
    # Handle device mapping
    self.llm = HuggingFacePipeline.from_model_id(  #pylint: disable=attribute-defined-outside-init
        model_id=self.model_name,
        task="text2text-generation",
        device=-1,
        model_kwargs={
            "temperature": self.temperature,
            "max_length": self.max_length
        })

  def construct_llm(self):
    """ LLM constructor method """
    tokenizer = AutoTokenizer.from_pretrained(self.model_name)
    if self.device_id == 'cuda':
      model = AutoModelForSeq2SeqLM.from_pretrained(
          self.model_name,
          load_in_8bit=self.load_in_8bit,
          torch_dtype=torch.float16,
          low_cpu_mem_usage=True,
          device_map="auto",
          trust_remote_code=True)
    else:
      model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
      model.to(torch.device(self.device_id))

    pipe = pipeline("text2text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    max_length=self.max_length,
                    temperature=self.temperature,
                    top_p=0.95,
                    repetition_penalty=1.15)
    self.llm = HuggingFacePipeline(pipeline=pipe)

  # Uncomment if want to use the instructor embeddings for local factory,
  # instead of the hugging face embeddings
  #def construct_embeddings(self):
  #    """ Embdings constructor method """
  #    self.embedding_model_name="hkunlp/instructor-xl"
  #    self.embeddings = HuggingFaceInstructEmbeddings(
  #        model_name=self.embedding_model_name,
  #        model_kwargs={"device": self.embeddings_device_id}
  #    )


class GoogleLLMFactory(LLMAndEmbeddingsFactory):
  """
    Factory for Google LLM components, I.e.: Vertex AI
    """

  def __init__(self, env_config=None, model_name="gemini-1.0-pro", embedding_model_name="textembedding-gecko", max_k=4):
    super().__init__(max_k=max_k, model_name=model_name, embedding_model_name=embedding_model_name, env_config=env_config)

  def __repr__(self):
    return f"<GoogleLLMFactory(model_name={self.model_name}) embedding_model_name={self.embedding_model_name}>"

  def construct_llm(self):
    """ LLM constructor method """
    self.llm = VertexAI(  # ChatVertexAI
        model_name=self.model_name,
        max_output_tokens=self.max_length,
        temperature=self.temperature,
        top_p=0.8,
        top_k=40,
        verbose=True)

  def construct_embeddings(self):
    """ Embdings constructor method """
    self.embeddings = VertexAIEmbeddings(self.embedding_model_name)


class GoogleAIstudioFactory(LLMAndEmbeddingsFactory):
  """
  Factory for Google AI Studio LLM components.
  """

  def __init__(self, env_config=None, model_name="gemini-pro", max_k=4):
    super().__init__(max_k=max_k, model_name=model_name, env_config=env_config)

  def __repr__(self):
    return f"<GoogleAIstudioFactory(model_name={self.model_name})>"

  def construct_llm(self):
    """LLM constructor method."""
    #TODO: AI studio - Add embeddings
    #TODO: AI studio - Check if need the key handling here
    #TOOD: AI studio - Agents not able to be instansiated - May need wrapper?
    self.llm = ChatGoogleGenerativeAI(
        model=self.model_name, google_api_key=os.environ["GOOGLE_API_KEY"])


class BardLLMFactory(LLMAndEmbeddingsFactory):
  """
    Factory for Bard
    Assumes API key is already set as environment variable,
    for example using os.environ['_BARD_API_KEY']="xxxxxxxx"
    Set environment variable BARD_EXPERIMENTAL to True for using Bardapi Experimental features
    (assumes using github branch of Bardapi)
    """

  def __init__(self, env_config=None, api_timeout=30):
    self.api_timeout = api_timeout
    super().__init__(env_config=env_config)

  def __repr__(self):
    return f"""<BardLLMFactory(model_name=Bard),
 embedding_model_name={self.embedding_model_name}>"""

  def construct_llm(self):
    """ LLM constructor method """
    self.llm = BardLLMWrapper(timeout=self.api_timeout)

  def image_support(self) -> bool:
    """ Added to attempt add support for Bardapi Experimental Features """
    return self.env_config['bard_experimental']

  def code_support(self) -> bool:
    """ Added to attempt add support for Bardapi Experimental Features """
    return self.env_config['bard_experimental']


# ----------------------- LLM Wrappers ------------------------------------------------

from bardapi import BardCookies


class BardLLMWrapper(LLM):
  """
    Simple wrapper which operates similar to langchain LLM,
    as Bard is Bardapi is not supported by langchain
    Attempted add support for Bard API Experimental fetures
    such as: retrieveing images, links and code, as well as running code.
    """
  googlellm: Bard = None
  code_runner: Bard = None
  last_response_dict = {'content': '', 'images': [], 'links': [], 'code': ''}
  response_element = 'content'

  def __init__(self, timeout=None):
    super().__init__()
    cookie_dict = {

        # Any cookie values you want to pass session object.
    }
    self.googlellm = BardCookies(cookie_dict=cookie_dict,
                                 timeout=timeout)  # Bard()
    self.code_runner = BardCookies(cookie_dict=cookie_dict,
                                   timeout=timeout,
                                   run_code=True)

  def _call(  #pylint: disable=unused-argument
      self,
      prompt: str,
      stop: Optional[List[str]] = None,
      run_manager=None,
  ) -> str:
    print(f"\nInput Prompt::::::::::::::::::::::::\n{prompt}")  # For debugging
    response_dict = self.googlellm.get_answer(prompt)
    print(f"\nresponse_dict::::::::::::::::::::::::\n{response_dict}"
          )  # For debugging
    self.last_response_dict = response_dict
    full_resp = response_dict[self.response_element]
    resp_frag = full_resp.replace('**',
                                  '')  # Replace the chars Bard sometimes adds
    if stop is not None:
      if not isinstance(stop, List):
        stop = stop()
      for stp in stop:
        tmp = full_resp.split(stp)  # Hacked stop sequence handling
        if len(tmp) > 1:
          resp_frag = tmp[0]
    print(f"\nFull Response Dict::::::::::::::::::::::\n{response_dict}")
    #print(f"\nFull Response until stop sequence:::::::::\n{resp_frag}") # For debugging
    self.response_element = 'content'  # Set the next response back to content
    return resp_frag

  @property
  def _llm_type(self) -> str:
    return "google-bard"

  def get_last_images(self) -> List:
    """ Methods for interacting with Bard API Experimental features """
    return self.last_response_dict['images']

  def get_last_image_links(self) -> List:
    """ Methods for interacting with Bard API Experimental features """
    return self.last_response_dict['links']

  def get_last_code_fragment(self) -> str:
    """ Methods for interacting with Bard API Experimental features """
    return self.last_response_dict['code']

  def set_next_response_for_images(self):
    """ Methods for interacting with Bard API Experimental features """
    self.response_element = 'images'

  def set_next_response_for_links(self):
    """ Methods for interacting with Bard API Experimental features """
    self.response_element = 'links'

  def set_next_response_for_code(self):
    """ Methods for interacting with Bard API Experimental features """
    self.response_element = 'code'

  def run_code(self, prompt):
    """ Methods for interacting with Bard API Experimental features """
    resp = self.code_runner.get_answer(
        prompt)  # How to get the output of the code execution?
    return resp
