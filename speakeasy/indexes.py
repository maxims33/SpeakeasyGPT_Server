from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from .customloaders import CustomCaptionLoader, DirectoryCaptionLoader
from langchain.document_loaders import (
        DirectoryLoader, 
        TextLoader, 
        CSVLoader,
        PyPDFLoader,
        UnstructuredPDFLoader,
        UnstructuredHTMLLoader,
        UnstructuredPowerPointLoader,
        UnstructuredWordDocumentLoader,
        ImageCaptionLoader
)


def load_document_helper(folder_path, file_ext, loader_class, text_splitter = None):
    loader = DirectoryLoader(folder_path, glob=file_ext, loader_cls=loader_class)
    docs = loader.load()
    if text_splitter == None:
        print(f"Number of {file_ext}: {len(docs)}")
        return docs
    texts = text_splitter.split_documents(docs)
    print(f"Number of {file_ext}: {len(docs)}, Number of texts split: {len(texts)}")
    return texts

# Initialize the Document VectorStore
def init_document_db(factory, file_path, persist_dir, chunk_size = 1500, chunk_overlap = 200):
    # Create Text Splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap  = chunk_overlap
    )
    texts = load_document_helper(file_path, "./*.pdf", UnstructuredPDFLoader, text_splitter) #PydPDFLoader loads a doc per page
    texts += load_document_helper(file_path, "./*.html", UnstructuredHTMLLoader, text_splitter) 
    texts += load_document_helper(file_path, "./*.pptx", UnstructuredPowerPointLoader, text_splitter) 
    texts += load_document_helper(file_path, "./*.docx", UnstructuredWordDocumentLoader, text_splitter) 
    texts += load_document_helper(file_path, "./*.csv", CSVLoader, text_splitter)
    texts += load_document_helper(file_path, "./*.txt", TextLoader, text_splitter)
    return create_document_db(factory, texts, persist_dir = persist_dir)

def create_document_db(factory, texts, persist_dir):
    # Create the VectorStore DB
    if len(texts) > 0:
        return Chroma.from_documents(texts, factory.embeddings, persist_directory=persist_dir)

def load_document_db(factory, persist_dir):
    # Load the VectorStore DB
    db = Chroma(persist_directory=persist_dir, embedding_function=factory.embeddings)    
    return db

# Initialize the Image VectorStore
def init_image_db(factory,  file_path, persist_dir):
    docs = DirectoryCaptionLoader(file_path = file_path, glob = "./*.jpg").load()
    docs += DirectoryCaptionLoader(file_path = file_path, glob = "./*.png").load()
    return create_image_db(factory, docs, persist_dir = persist_dir)

def create_image_db(factory, docs, persist_dir):
    # Create the VectorStore DB
    if len(docs) > 0:
        return Chroma.from_documents(docs, factory.embeddings, persist_directory=persist_dir)

def load_image_db(factory, persist_dir):
    # Load the VectorStore DB
    db = Chroma(persist_directory=persist_dir, embedding_function=factory.embeddings)
    return db

