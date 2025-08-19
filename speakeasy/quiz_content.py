from typing import List
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# Define structure of a single question using Pydantic
class QuizQuestion(BaseModel):
    question: str = Field(..., description="The quiz question.")
    options: List[str] = Field(..., description="List of answer options.", min_items=4, max_items=4, example=["A) text", "B) text", "C) text", "D) text"])
    answer: str = Field(..., description="The correct answer. MUST be just a single letter (A,B,C or D)", min_length=1)

# Define structure of a list of quiz questions
class QuizQuestions(BaseModel):
  questions: List[QuizQuestion] = Field(..., description="List of quiz questions.")
  
def generate_quiz_questions(llm, topic):
  # Create the prompt template  
  prompt_template = PromptTemplate.from_template("The topic of the quiz is {topic}. Generate 5  multiple-choice questions. {format_instructions}")

  # Create the output parser
  output_parser = PydanticOutputParser(pydantic_object=QuizQuestions)

  # Create the chain
  quiz_chain = prompt_template | llm | output_parser

  # Generate the quiz question
  result =  quiz_chain.invoke({'topic': topic, 
    'format_instructions': output_parser.get_format_instructions()
  })

  return result
