"""
Module for custom handling of loading files
"""

from typing import List, Union
from PIL import Image
from langchain.docstore.document import Document
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders.base import BaseLoader

class CustomCaptionLoader(BaseLoader):
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
            from clip_interrogator import Config, Interrogator #pylint: disable=import-outside-toplevel
        except ImportError as exec1:
            raise ValueError(
                "Raising Exception. Missing clip_interrogator moddule"
            ) from exec1

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
            #caption = cap_int.interrogate(image) # Reduced memory needed by not running clip model
            caption = cap_int.generate_caption(image)
            print(f"Image Caption for {it}: {caption}")
            doc = Document(page_content=caption, metadata={"source": str(it)})
            doc_list.append(doc)
        return doc_list

class DirectoryCaptionLoader(DirectoryLoader):
    """ Load directoy of captions of images """
    def __init__(self, file_path, glob="./*"):
        super().__init__(file_path, glob = glob, loader_cls = None)
        self.image_paths = []

    def load_file(self, item, path, docs, pbar): #pylint: disable=unused-argument
        """ override the load_files method """
        if item.is_file():
            self.image_paths.append(item)

    def load(self) -> List[Document]:
        """ override the load method """
        super().load()
        captions = CustomCaptionLoader(self.image_paths).load()
        return captions
