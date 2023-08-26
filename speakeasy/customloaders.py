"""
Module for custom handling of loading files
"""

from typing import List, Union
from PIL import Image
from langchain.docstore.document import Document
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders.base import BaseLoader

class CustomCaptionLoader(BaseLoader): #pylint: disable=too-few-public-methods
    """ Load captions of images """
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

        config = Config(clip_model_name="ViT-L-14/openai", 
            blip_model_type='blip-large', 
            device='cuda', 
            cache_path="./img_embeddings"
        )
        config.apply_low_vram_defaults()
        cap_int = Interrogator(config)
        doc_list : List[Document] = []
        for it in self.image_paths: #pylint: disable=invalid-name
            image = Image.open(it).convert('RGB')
            #caption = cap_int.interrogate(image) # Reduced memory required by not running clip model
            caption = cap_int.generate_caption(image)
            print(f"Image Caption for {it}: {caption}")
            doc = Document(page_content=caption, metadata={"source": str(it)})
            doc_list.append(doc)
        return docc_list

class DirectoryCaptionLoader(DirectoryLoader): #pylint: disable=too-few-public-methods
    """ Load directoy of captions of images """
    def __init__(self, file_path, glob="./*"):
        super().__init__(file_path, glob = glob, loader_cls = None)
        self.image_paths = []

    def load_file(self, item, path, docs, pbar): #pylint: disable=unused-argument
        if item.is_file():
            self.image_paths.append(item)
    
    def load(self) -> List[Document]:
        super().load()
        captions = CustomCaptionLoader(self.image_paths).load()
        return captions
