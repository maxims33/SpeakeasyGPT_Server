from typing import List
from pydantic import BaseModel, Field

# NOT USED YET

# Define structure of a single question using Pydantic

class RecipeModel(BaseModel):
  name: str = Field(..., description="The name of the dish.")
  ingredients: List[str] = Field(..., 
    description="List of ingredients needed to make the dish.")
  instructions: str = Field(..., description="The steps to make the dish.")
  filename: str = Field(...,
    description="The the path to the file where the image was saved.")

class ImagePromptAndFilename(BaseModel):
  image_prompt: str = Field(..., 
    description="A promot for generating an image for a recipe.")
  filename: str = Field(..., 
    description="The name of the image file includng png extension")