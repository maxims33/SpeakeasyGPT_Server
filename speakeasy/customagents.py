"""
Module defining custom langchain agents
"""

from typing import Any, List, Tuple, Sequence, Callable
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from .tools.custom_tools import CustomInstructLLMTool, CustomMathTool
from .tools.vectorstore_tools import CustomDocumentQueryTool, CustomImageQueryTool

def handle_agent_error(error) -> str:
    """ Error handler method """
    print("THE OUTPUT FORMAT IS INCORRECT!")
    return f"CHECK YOUR OUTPUT FORMAT! {str(error)[:200]}"

# ------------------------------------------------------------------------------------

def custom_tools(factory, doc_db, img_db) -> List:
    """ Collect the custom tools """
    c_t = [
        CustomInstructLLMTool(factory),
        CustomMathTool(factory),
        CustomDocumentQueryTool(factory, doc_db),
        CustomImageQueryTool(factory, img_db)
    ]
    return c_t

def init_agent(factory,
        doc_db, img_db,
        max_iterations=1,
        verbose=True,
        early_stopping_method='generate'
    ):
    """ Initialize Agent - Without memory """
    system_prompt = """Answer the following questions or tasks as best you can."""
    
    tools = custom_tools(factory, doc_db, img_db)
    
    # create_agent from langchain.agents returns a CompiledStateGraph (LangGraph)
    agent = create_agent(
        model=factory.llm,
        tools=tools,
        system_prompt=system_prompt,
        debug=verbose
    )
    
    return agent

def init_conversational_agent(factory,
        doc_db,
        img_db,
        max_iterations=1,
        verbose=True,
        early_stopping_method='generate'
    ):
    """ Initialize Conversational Agent - With memory (LangGraph Persistence) """
    system_prompt = """You are Assistant, a large language model.
Assistant is designed to be able to assist with a wide range of tasks, 
from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. 
As a language model, Assistant is able to generate human-like text based on the input it receives, 
allowing it to engage in natural-sounding conversations and provide responses 
that are coherent and relevant to the topic at hand."""
    
    tools = custom_tools(factory, doc_db, img_db)

    # LangGraph uses a checkpointer for memory
    memory = MemorySaver()

    agent = create_agent(
        model=factory.llm,
        tools=tools,
        system_prompt=system_prompt,
        checkpointer=memory,
        debug=verbose
    )
    
    return agent
