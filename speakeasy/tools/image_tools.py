from .custom_tools import CustomBaseTool
import requests
import json
import base64
import io
from PIL import Image, PngImagePlugin

#TODO Consider loading and storing image into Vectorstore? E.g.: Loading image caption into vectorstore

# Class representing a tool used for generating image using stable diffusion API
# Note that Automatic1111 should be run using -api flag
class CustomGenerateImageTool(CustomBaseTool):
    generation_steps : int = None
    api_url : str = None
    output_filename : str = None
    save_image = True

    def __init__(self, fa, 
            db = None, 
            return_direct = False, 
            api_url = None, 
            generation_steps = 10, 
            output_filename = './generated_images/output.png', 
            save_image = True
        ):
        super(CustomGenerateImageTool, self).__init__(fa,
                name="Preview_Image",
                description="Use this tool to when instructed to preview image or a picture." 
                " The 'Action Input:' for this tool should be prompt optimized for stable diffusion",
                db = db,
                return_direct = return_direct
            )
        self.generation_steps = generation_steps
        self.api_url = api_url
        self.output_filename = output_filename
        self.save_image = save_image

    # Saving as a PNG. Getting PNG info via API also - Is there a good reason to (re-)save it to file? Maybe just temporary
    def saveImageAsPNG(self, imageBase64 : str, imageStr : str):
        image = Image.open(io.BytesIO(base64.b64decode(imageBase64)))
        png_payload = {"image": "data:image/png;base64," + imageStr}
        response2 = requests.post(url=f'{self.api_url}/sdapi/v1/png-info', json=png_payload)
        pnginfo = PngImagePlugin.PngInfo()
        pnginfo.add_text("parameters", response2.json().get("info"))
        file_name = f'{self.output_filename}' # Add timestamps in filename
        print( f'Generated image and saved at {file_name}')
        image.save(file_name, pnginfo=pnginfo)

    # Would be nice to implement an async version since does take a while
    def _run(self, query: str) -> str:
        payload = { "prompt": query, "steps": self.generation_steps } # 
        response = requests.post(url=f'{self.api_url}/sdapi/v1/txt2img', json=payload) 
        # Check response codes, handle errors
        r = response.json()
        for i in r['images']: # Assumes single image
            imageBase64 = i.split(",",1)[0]
            if self.save_image == True:
                self.saveImageAsPNG(imageBase64, i)
            return f'BASE64ENCODED:{imageBase64}'
        return 'Ooops, No image generated.'

