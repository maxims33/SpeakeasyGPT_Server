from langchain.tools import Tool
from langchain.tools.base import ToolException
from langchain.chains import PALChain, LLMMathChain
from langchain.chains.pal.math_prompt import MATH_PROMPT
from langchain.chains.llm_math.prompt import PROMPT
from langchain.vectorstores import Chroma
from speakeasy.llmfactory import LLMFactory

#TODO More tools to create - E.g.: Summerization, Transalation, etc
#TODO PAL Tool not working with Bard (Maybe need experimental Bard for code, and also possibly issue with the stop used).
#TODO Consider further for which tools return_direct should be utilised to bypass agent handling the response from a tool
#TODO Consider using decorators or other options for code simplification.

class CustomBaseTool(Tool):
    factory : LLMFactory = None
    vector_db : Chroma = None

    def __init__(self, fa, name, description, db = None, return_direct = False):
        super(CustomBaseTool, self).__init__(
                return_direct = return_direct, 
                name = name, 
                description = description, 
                func = self._run, 
                handle_tool_error = self._handle_error
        )
        self.factory = fa
        self.vector_db = db

    def _run(self, query: str) -> str:
        """Use the tool."""
        raise NotImplementedError("does not support async")

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("does not support async")

    # Subclasses to override to customise error handling
    def _handle_error(self, error : ToolException) -> str:
        print("TOOL ERROR ENCOUNTERED!")
        return  "The following errors occurred during tool execution:" + error.args[0]+ "Please try another tool!"


# ------------------------------------------------------------------------

# Class representing tool for logic operations using Program Aided LLM
class CustomPALTool(CustomBaseTool):
    def __init__(self, fa, return_direct = False):
        super(CustomPALTool, self).__init__(fa,
                name="PAL_Logic",
                description="Use this tool only for programming logic to determine solution." 
                " DO NOT do the math yourself, or change the question wording when defining the 'Action Input:' for this tool.",
                return_direct = return_direct
            )

    def _run(self, query: str) -> str:
        pp = MATH_PROMPT
        pp.template = pp.template + "ONLY output the code as per the below examples. DO NOT respond by telling me Sure!"
        pal_chain = PALChain.from_math_prompt(self.factory.llm, verbose=True, prompt=pp)
        return pal_chain(query)['result']

# Class representing tool for math operations using expression evaluation
class CustomMathTool(CustomBaseTool):
    def __init__(self, fa, return_direct = False):
        super(CustomMathTool, self).__init__(fa,
                name="EvaluateExpression",
                description="Use this tool only for any mathematical calculations.",
                return_direct = return_direct
            )

    def _run(self, query: str) -> str:
        pp = PROMPT
        math_chain = LLMMathChain.from_llm(self.factory.llm, verbose=True, prompt=pp)
        return math_chain.run(query) 

# Class representing a tool used for generating content
class CustomInstructLLMTool(CustomBaseTool):
    def __init__(self, fa, return_direct = False):
        super(CustomInstructLLMTool, self).__init__(fa,
                name="Instruct_LLM",
                description="Use this tool to when instructed to generate content (like writing an email/letter/prompt)," 
                " or for searching for information online.",
                return_direct = return_direct
            )

    def _run(self, query: str) -> str:
        return self.factory.llm(query)

