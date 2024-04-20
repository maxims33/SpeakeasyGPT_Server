import unittest
from unittest.mock import patch
from speakeasy.llmfactory import init_factory_from_type, LLMType


class TestOpenAIFactory(unittest.TestCase):

  @patch('speakeasy.llmfactory.OpenAIFactory')
  def test_construct_llm(self, mock_openai_factory):
    factory = init_factory_from_type(LLMType.OPENAI, {})
    mock_openai_factory.assert_called_once()
