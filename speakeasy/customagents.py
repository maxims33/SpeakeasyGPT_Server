"""
Module defining custom langchain agents
#TODO Exploring adding memory backed by a Vectorstore. Explicit memory 'management'
"""

from typing import Any, List, Tuple
from langchain.agents import AgentExecutor
from langchain.schema import AgentAction, AgentFinish
from langchain.agents.mrkl.base import ZeroShotAgent
from langchain.agents.conversational.base import ConversationalAgent
from langchain.agents.agent_toolkits import VectorStoreToolkit, VectorStoreInfo
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from .tools.custom_tools import CustomInstructLLMTool, CustomMathTool
from .tools.vectorstore_tools import CustomDocumentQueryTool, CustomImageQueryTool

class CustomBaseAgent():
    """ Base class for custom agents """
    def handle_agent_error(self, error) -> str:
        """ Error handler method """
        print("THE OUTPUT FORMAT IS INCORRECT!")
        return f"CHECK YOUR OUTPUT FORMAT! {str(error)[:200]}"


class CustomSingleActionAgent(CustomBaseAgent, ZeroShotAgent):
    """ Custom single action agents """
    def __init__(self, llm_chain, allowed_tools, verbose):
        super().__init__(llm_chain=llm_chain,
                allowed_tools=allowed_tools,
                verbose=verbose
        )

class CustomConversationalAgent(CustomBaseAgent, ConversationalAgent):
    """ Custom conversation agent """
    def __init__(self, llm_chain, allowed_tools, verbose):
        super().__init__(llm_chain=llm_chain,
            allowed_tools=allowed_tools,
            verbose=verbose
        )


# ------------------------------------------------------------------------------------

def base_tools(factory, doc_db, img_db) -> List:
    """ Some issues handling documents with sources - reverting to custom tools """
    vectorstore_info = VectorStoreInfo(
        name="Local_Documents",
        description="Documents and PDFs",
        vectorstore=doc_db
    )
    # Need to pass in max_k
    toolkit = VectorStoreToolkit(llm=factory.llm, vectorstore_info=vectorstore_info)
    b_t = toolkit.get_tools()

    vectorstore_info = VectorStoreInfo(
        name="Local_Images",
        description="Photos and Pictures and Images",
        vectorstore=img_db
    )
    toolkit = VectorStoreToolkit(llm=factory.llm, vectorstore_info=vectorstore_info)
    b_t = b_t + toolkit.get_tools()
    return b_t

def custom_tools(factory, doc_db, img_db, include_base_tools = False) -> List:
    """ Collect the custom tools, and base tools if desired """
    b_t = []
    if include_base_tools is True:
        b_t = base_tools(factory, doc_db, img_db)

    c_t = [
        CustomInstructLLMTool(factory),
        CustomMathTool(factory),
        CustomDocumentQueryTool(factory, doc_db),
        CustomImageQueryTool(factory, img_db)
    ]
    return c_t + b_t

def default_agent_executor(agent,
        tools,
        memory=None,
        max_iterations=1,
        verbose=True,
        early_stopping_method='generate'
    ):
    """ Create a deafult agent executor """
    return AgentExecutor.from_agent_and_tools(
        agent=agent,
        memory=memory,
        tools=tools,
        verbose=verbose,
        max_iterations=max_iterations,
        early_stopping_method=early_stopping_method,
        handle_parsing_errors=agent.handle_agent_error
    )

def init_agent(factory,
        doc_db, img_db,
        max_iterations=1,
        verbose=True,
        early_stopping_method='generate'
    ):
    """ Initialize Agent - Without memory """
    prefix = """Answer the following questions or tasks as best you can.
Ensure to choose one of the following tools as the 'Action':"""
    suffix = """You MUST to stick to the format provided above,
but do not output the 'Final Answer:' until instructed. Begin!\n
Question: {input}
{agent_scratchpad}"""
    tools = custom_tools(factory, doc_db, img_db)
    tool_names = [tool.name for tool in tools]
    prompt = CustomSingleActionAgent.create_prompt(tools,
        prefix=prefix,
        suffix=suffix,
        input_variables=["input", "agent_scratchpad"]
    )
    agent = CustomSingleActionAgent(llm_chain = LLMChain(llm=factory.llm, prompt=prompt),
        allowed_tools=tool_names, verbose=verbose
    )
    agent_executor = default_agent_executor(agent, tools,
            max_iterations=max_iterations,
            verbose=verbose,
            early_stopping_method=early_stopping_method
        )
    return agent_executor

def init_conversational_agent(factory,
        doc_db,
        img_db,
        max_iterations=1,
        verbose=True,
        early_stopping_method='generate'
    ):
    """ Initialize Conversational Agent - With memory """
    prefix = """You are Assistant, a large language model.
Assistant is designed to be able to assist with a wide range of tasks, 
from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. 
As a language model, Assistant is able to generate human-like text based on the input it receives, 
allowing it to engage in natural-sounding conversations and provide responses 
that are coherent and relevant to the topic at hand.

Assistant is constantly learning and improving, and its capabilities are constantly evolving. 
It is able to process and understand large amounts of text, and can use this knowledge to provide accurate 
and informative responses to a wide range of questions. 
Additionally, Assistant is able to generate its own text based on the input it receives, 
allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

Overall, Assistant is a powerful tool that can help with a wide range of tasks and provide valuable insights
and information on a wide range of topics. 
Whether you need help with a specific question or just want to have a conversation about a particular topic, 
Assistant is here to assist.

TOOLS:
------

Assistant has access to the following tools:"""
    suffix = """You MUST to stick to the format provided above, Begin!

Previous conversation history:
{chat_history}

New input: {input}
{agent_scratchpad}"""
    instruction_format = """To use a tool, please use the following format:

```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
```

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

```
Thought: Do I need to use a tool? No
{ai_prefix}: [your response here]
```"""
    tools = custom_tools(factory, doc_db, img_db)
    tool_names = [tool.name for tool in tools]
    prompt = CustomConversationalAgent.create_prompt(tools,
        prefix=prefix,
        suffix=suffix,
        format_instructions=instruction_format,
        input_variables=["chat_history", "input", "agent_scratchpad"]
    )
    agent = CustomConversationalAgent(llm_chain = LLMChain(llm=factory.llm, prompt=prompt),
        allowed_tools=tool_names,
        verbose=verbose
    )
    agent_executor = default_agent_executor(agent, tools,
            memory=ConversationBufferMemory(memory_key="chat_history"),
            max_iterations=max_iterations,
            verbose=verbose,
            early_stopping_method=early_stopping_method
        )
    return agent_executor
