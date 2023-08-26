"""
Module for custom langchain tools
"""

from langchain.tools import Tool
from langchain.tools.base import ToolException
from langchain.chains import LLMMathChain
from langchain.chains.llm_math.prompt import PROMPT
from langchain.vectorstores.base import VectorStore
from speakeasy.llmfactory import LLMFactory


class CustomBaseTool(Tool):
    """
    Base class for custom tools
    """
    factory : LLMFactory = None
    vector_db : VectorStore = None

    def __init__(self, fact, name, description,
            vdb = None, return_direct = False): #pylint: disable=too-many-arguments
        super().__init__(
                return_direct = return_direct,
                name = name,
                description = description,
                func = self._run,
                handle_tool_error = self._handle_error
        )
        self.factory = fact
        self.vector_db = vdb

    def _run(self, query: str) -> str:
        """Use the tool."""
        raise NotImplementedError("does not support async")

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("does not support async")

    # Subclasses to override to customise error handling
    def _handle_error(self, error : ToolException) -> str:
        print("TOOL ERROR ENCOUNTERED!")
        return  ("The following errors occurred during tool execution:" + error.args[0] + 
            "Please try another tool!")


# ------------------------------------------------------------------------


class RunCodeTool(CustomBaseTool):
    """
     Class representing tool for executing (python) code - Only supported with Bard Experimental
    """
    def __init__(self, fact, return_direct = False):
        super().__init__(fact,
                name="Run_Code",
                description="Use this tool when asked to create and execute code / script. "
                "Pass the original request as the 'Action Input:' for this tool",
                return_direct = return_direct
            )

    def _run(self, query: str) -> str:
        if self.factory.code_support() == True:
            return self.factory.llm.run_code(query)['code']
        return "Tool not supported. Check Bard_Experimental setting."

# Class representing tool for math operations using expression evaluation
class CustomMathTool(CustomBaseTool):
    def __init__(self, fact, return_direct = False):
        super().__init__(fact,
                name="EvaluateExpression",
                description="Use this tool only for any mathematical calculations.",
                return_direct = return_direct
            )

    def _run(self, query: str) -> str:
        pp = PROMPT
        math_chain = LLMMathChain.from_llm(self.factory.llm, verbose=True, prompt=pp)
        return math_chain.run(query) 

# Class representing a tool used for generating content - Simply queries the LLM
class CustomInstructLLMTool(CustomBaseTool):
    def __init__(self, fact, return_direct = False):
        super().__init__(fact,
                name="Instruct_LLM",
                description="Use this tool to when instructed to generate content "
                "(like writing an email/letter/prompt), or for searching for information online.",
                return_direct = return_direct
            )

    def _run(self, query: str) -> str:
        return self.factory.llm(query)
