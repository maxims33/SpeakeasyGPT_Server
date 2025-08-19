"""
Module for various types of LLM Factory
#TODO Exploring Chat Models and session/conversation management.
#TODO Exploring the Callback functionality - E.g.: Logging LLM stats
#TODO Exploring the Async functionality?
#TODO HF Pipelines - Factory for using text-generation, AutoModelForCausalLM
#TODO How to get load_in_8bit=True working in windows? quantized models only?
#TODO Removing unused code and imports
"""
from typing import List, Optional
from enum import Enum
import torch
import os
from langchain_community.llms import HuggingFaceHub
from langchain_huggingface import HuggingFacePipeline
from langchain.llms.base import LLM
from langchain_community.llms import OpenAI
from langchain_google_vertexai import VertexAI, VertexAIEmbeddings, ChatVertexAI
from langchain_google_genai import ChatGoogleGenerativeAI
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

  def __init__(self, env_config=None, model_name="gemini-2.0-flash", embedding_model_name="text-embedding-005", max_k=4):
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

  def __init__(self, env_config=None, model_name="gemini-2.0-flash", max_k=4):
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



