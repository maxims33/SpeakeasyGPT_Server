from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
from langchain.agents import initialize_agent, AgentExecutor
from langchain.schema import AgentAction, AgentFinish
from langchain.agents.mrkl.base import ZeroShotAgent
from langchain.agents.conversational.base import ConversationalAgent
from langchain.agents.agent_toolkits import VectorStoreToolkit, VectorStoreInfo
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from .tools.custom_tools import CustomInstructLLMTool, CustomMathTool, RunCodeTool
from .tools.vectorstore_tools import CustomDocumentQueryTool, CustomImageQueryTool
from .tools.image_tools import CustomGenerateImageTool, SearchImageTool

#TODO Exploring other agents and output parsers E.g.: Structured/Pydantic Chat? OpenAI functions?
#TODO Exploring adding memory backed by a Vectorstore. Explicit memory 'management' 

class CustomBaseAgent(object):
    def _handle_agent_error(self, error) -> str:
        print("THE OUTPUT FORMAT IS INCORRECT!")
        return f"CHECK YOUR OUTPUT FORMAT! {str(error)[:200]}"

# Use only with Bard 
class CustomBardAgent(CustomBaseAgent):
    @property
    def observation_prefix(self) -> str:
        """Prefix to append the observation with."""
        return "Observation:"

    def _stop(self):
        return [f"\n{self.observation_prefix}"]

    # Overriding method from base class - main reason is to avoid sending stop on the last pass
    def return_stopped_response(
        self,
        early_stopping_method: str,
        intermediate_steps: List[Tuple[AgentAction, str]],
        **kwargs: Any,
    ) -> AgentFinish:
        """Return response when agent has been stopped due to max iterations."""
        if early_stopping_method == "force":
            # `force` just returns a constant string - Not currently using
            return AgentFinish(
                {"output": "Agent stopped due to iteration limit or time limit."}, ""
            )
        elif early_stopping_method == "generate":
            # Generate does one final forward pass
            thoughts = ""
            for action, observation in intermediate_steps:
                thoughts += action.log
                thoughts += (
                    f"\n{self.observation_prefix}{observation}\n{self.llm_prefix}"
                )
            # Adding to the previous steps, we now tell the LLM to make a final pred
            thoughts += (
                    "\n\nI now need to return ONLY a final answer based on the previous steps. The answer MUST be prefixed with the following exact label 'Final Anwer: '. DO NOT OUTPUT ACTION TAG!"
            )
            new_inputs = {"agent_scratchpad": thoughts, "stop":None} #Setting stop to None
            full_inputs = {**kwargs, **new_inputs}
            full_output = self.llm_chain.predict(**full_inputs)
            # We try to extract a final answer
            parsed_output = self.output_parser.parse(full_output)
            if isinstance(parsed_output, AgentFinish):
                # If we can extract, we send the correct stuff
                return parsed_output
            else:
                # If we can extract, but the tool is not the final tool,
                # we just return the full output
                return AgentFinish({"output": full_output}, full_output)
        else:
            raise ValueError(
                "early_stopping_method should be one of `force` or `generate`, "
                f"got {early_stopping_method}"
            )

class CustomSingleActionAgent(CustomBaseAgent, ZeroShotAgent):
    def __init__(self, llm_chain, allowed_tools, verbose):
        super(CustomSingleActionAgent, self).__init__(llm_chain=llm_chain, allowed_tools=allowed_tools, verbose=verbose)

class CustomConversationalAgent(CustomBaseAgent, ConversationalAgent):
    def __init__(self, llm_chain, allowed_tools, verbose):
        super(CustomConversationalAgent, self).__init__(llm_chain=llm_chain, allowed_tools=allowed_tools, verbose=verbose)


# ------------------------------------------------------------------------------------

# Some issues handling documents with sources - reverting to custom tools
def base_tools(factory, doc_db, img_db) -> List:
    vectorstore_info = VectorStoreInfo(
        name="Local_Documents",
        description="Documents and PDFs",
        vectorstore=doc_db
    )
    # Need to pass in max_k
    toolkit = VectorStoreToolkit(llm=factory.llm, vectorstore_info=vectorstore_info)
    bt = toolkit.get_tools()

    vectorstore_info = VectorStoreInfo(
        name="Local_Images",
        description="Photos and Pictures and Images",
        vectorstore=img_db
    )
    toolkit = VectorStoreToolkit(llm=factory.llm, vectorstore_info=vectorstore_info)
    bt = bt + toolkit.get_tools()
    return bt

def custom_tools(factory, doc_db, img_db, include_base_tools = False) -> List:
    bt = []
    if include_base_tools == True:
        bt = base_tools(factory, doc_db, img_db)
    
    ct = [
        CustomInstructLLMTool(factory),
        CustomMathTool(factory),
        CustomGenerateImageTool(factory, 
            return_direct=True, 
            api_url=factory.env_config['sd_url'], 
            generation_steps=factory.env_config['sd_steps'],
            output_filename=factory.env_config['image_output_filename']
        ),
        CustomDocumentQueryTool(factory, doc_db), 
        CustomImageQueryTool(factory, img_db), 
        SearchImageTool(factory, return_direct=True),
        RunCodeTool(factory, return_direct=True)
    ]
    return ct + bt

def default_agent_executor(agent, tools, memory=None, max_iterations=1, verbose=True, early_stopping_method='generate'):
    return AgentExecutor.from_agent_and_tools(
        agent=agent,
        memory=memory,
        tools=tools,
        verbose=verbose,
        max_iterations=max_iterations,
        early_stopping_method=early_stopping_method,
        handle_parsing_errors=agent._handle_agent_error
    )

# Initialize Agent - Without memory
def init_agent(factory, doc_db, img_db, max_iterations=1, verbose=True, early_stopping_method='generate'):
    prefix = """Answer the following questions or tasks as best you can. Ensure to choose one of the following tools as the 'Action':"""
    suffix = """You MUST to stick to the format provided above, but do not output the 'Final Answer:' until instructed. Begin!\n
Question: {input}
{agent_scratchpad}"""
    tools = custom_tools(factory, doc_db, img_db)
    tool_names = [tool.name for tool in tools]
    prompt = CustomSingleActionAgent.create_prompt(tools, input_variables=["input", "agent_scratchpad"]) #prefix=prefix, suffix=suffix,
    llm_chain = LLMChain(llm=factory.llm, prompt=prompt)
    agent = CustomSingleActionAgent(llm_chain=llm_chain, allowed_tools=tool_names, verbose=verbose)
    agent_executor = default_agent_executor(agent, tools, 
            max_iterations=max_iterations, 
            verbose=verbose, 
            early_stopping_method=early_stopping_method
        ) 
    return agent_executor

# Initialize Conversational Agent - With memory
def init_conversational_agent(factory, doc_db, img_db, max_iterations=1, verbose=True, early_stopping_method='generate'): 
    prefix = """You are Assistant, a large language model. Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

Overall, Assistant is a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.

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
    prompt = CustomConversationalAgent.create_prompt(tools, prefix=prefix, suffix=suffix, format_instructions=instruction_format, input_variables=["chat_history", "input", "agent_scratchpad"]) 
    llm_chain = LLMChain(llm=factory.llm, prompt=prompt)
    agent = CustomConversationalAgent(llm_chain=llm_chain, allowed_tools=tool_names, verbose=verbose)
    mem = memory=ConversationBufferMemory(memory_key="chat_history")
    agent_executor = default_agent_executor(agent, tools, 
            memory=mem, 
            max_iterations=max_iterations, 
            verbose=verbose, 
            early_stopping_method=early_stopping_method
        ) 
    return agent_executor

