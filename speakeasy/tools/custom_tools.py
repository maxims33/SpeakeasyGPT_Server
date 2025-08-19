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
            vdb = None, return_direct = False):
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
        return  "Errors during tool execution: " + error.args[0] + " Please try another tool!"


# ------------------------------------------------------------------------


class CustomMathTool(CustomBaseTool):
    """
    Class representing tool for math operations using expression evaluation
    """
    def __init__(self, fact, return_direct = False):
        super().__init__(fact,
                name="EvaluateExpression",
                description="Use this tool only for any mathematical calculations.",
                return_direct = return_direct
            )

    def _run(self, query: str) -> str:
        p_t = PROMPT
        math_chain = LLMMathChain.from_llm(self.factory.llm, verbose=True, prompt=p_t)
        return math_chain.run(query)

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("does not support async")

class CustomInstructLLMTool(CustomBaseTool):
    """
    Class representing a tool used for generating content - Simply queries the LLM
    """
    def __init__(self, fact, return_direct = False):
        super().__init__(fact,
                name="Instruct_LLM",
                description="Use this tool to when instructed to generate content "
                "(like writing an email/letter/prompt), or for searching for information online.",
                return_direct = return_direct
            )

    def _run(self, query: str) -> str:
        return self.factory.llm.invoke(query)

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("does not support async")
