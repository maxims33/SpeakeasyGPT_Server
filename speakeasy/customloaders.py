from PIL import Image
from typing import List, Union
from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader

class CustomCaptionLoader(BaseLoader):
    def __init__(
        self,
        path_images: Union[str, List[str]],
    ):
        if isinstance(path_images, str):
            self.image_paths = [path_images]
        else:
            self.image_paths = path_images
 
    def load(self) -> List[Document]:
        """
        Load from a list of image files. Assumes only single item in list I think
        """
        try:
            from clip_interrogator import Config, Interrogator
        except ImportError:
            raise ValueError(
                "Raising Exception. Missing clip_interrogator moddule"
            )

        image = Image.open(self.image_paths[0]).convert('RGB')
        config = Config(clip_model_name="ViT-L-14/openai", blip_model_type='blip-large', device='cuda', cache_path="./img_embeddings")
        config.apply_low_vram_defaults()
        ci = Interrogator(config)
        #caption = ci.interrogate(image)
        caption = ci.generate_caption(image)
        doc = Document(page_content=caption, metadata={"source": self.image_paths[0]})
        return [doc]

