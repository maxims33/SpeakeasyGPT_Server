import json
import base64
from typing import List
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from speakeasy.orm.models import find_recipes

# Define structure of a single recipe using Pydantic
class RecipeModel(BaseModel):
    """ A Recipe """
    name: str = Field(..., description="The name of the dish.")
    ingredients: List[str] = Field(..., 
      description="List of ingredients needed to make the dish.", 
      min_items=1, 
      example=["1 teaspoon of sugar", "half cup of water", "2 kilos of potatoes", "flour"])
    instructions: str = Field(..., 
      description="The steps to make the dish", 
      min_length=1)
  
# Define structure of the nutrition content
class Menu(BaseModel):
  """ The Menu consisting of a list of Recipes """
  menu: List[RecipeModel] = Field(..., description="List of receipes.")

#-----------------------------------------

def generate_menu(llm, category):
  # Create the prompt template  
  prompt_template = PromptTemplate.from_template("""
  The category of the menu is {category}. 
  Generate 5 receipes for this category. {format_instructions}""")

  # Create the output parser
  output_parser = PydanticOutputParser(pydantic_object=Menu)

  # Create the chain
  chain = prompt_template | llm | output_parser

  # Generate the content for the given category
  result =  chain.invoke({'category': category, 
    'format_instructions': output_parser.get_format_instructions()
  })

  return result

def load_image_as_base64(image_path: str):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return encoded_string

# Retrieve the menu from the database
def retrieve_menu(category: str, image_folder: str):
  content_list = []
  for recipe in find_recipes(category = category):
    try:
      content_list.append({
        "id": recipe.id, #type: ignore
        "name": str(recipe.name), 
        "category": str(recipe.category),
        "ingredients": json.loads(str(recipe.ingredients)),
        "instructions": str(recipe.instructions), 
        #"image_file": load_image_as_base64(f"{image_folder}/{str(recipe.image_file)}")
        "image_file": str(recipe.image_file)
      })
    except Exception as exc:
      print(f"Caught exception parsing recipes (id={str(recipe.id)}): {exc}")
      pass
  content = { "menu": content_list }
  return content
