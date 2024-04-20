import unittest
from unittest.mock import patch
from env_params import parse_environment_variables

class TestEnvParams(unittest.TestCase):

    @patch.dict('os.environ', {'ENABLE_DEBUG': 'True'})
    def test_debug_enabled(self):
        env_vars = parse_environment_variables()
        self.assertTrue(env_vars['debug'])

    @patch.dict('os.environ', {'DOCS_PERSIST_DIRECTORY': '/path/to/docs'})
    def test_docs_directory(self):
        env_vars = parse_environment_variables()
        self.assertEqual(env_vars['docs_persist_directory'], '/path/to/docs')