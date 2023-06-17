from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader, PyPDFLoader, DirectoryLoader, UnstructuredPDFLoader
from langchain.vectorstores import Chroma
from .customloaders import CustomCaptionLoader

#TODO Support multiple image types like PrivateGPT
#TODO Chromadb in pipenv giving warning about C functions, unlike the one installed with pip

# Initialize the Document VectorStore
def init_document_db(factory, file_path, persist_dir, chunk_size = 1500, chunk_overlap = 200):
    # Create Text Splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap  = chunk_overlap
    )
    # Load the PDFs
    pdf_folder_path = file_path
    loader = DirectoryLoader(pdf_folder_path, glob="./*.pdf", loader_cls=UnstructuredPDFLoader) #PydPDFLoader loads a doc per page
    docs = loader.load()
    texts = text_splitter.split_documents(docs)
    print(f"Number of Docs: {len(docs)}, Number of Texst: {len(texts)}") 
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
    # Load the Images
    img_folder_path = file_path
    loader = DirectoryLoader(img_folder_path, glob="*.jpg", loader_cls=CustomCaptionLoader) 
    docs = loader.load()
    print(f"Number of Images: {len(docs)}")
    return create_image_db(factory, docs, persist_dir = persist_dir)

def create_image_db(factory, docs, persist_dir):
    # Create the VectorStore DB
    if len(docs) > 0:
        return Chroma.from_documents(docs, factory.embeddings, persist_directory=persist_dir)

def load_image_db(factory, persist_dir):
    # Load the VectorStore DB
    db = Chroma(persist_directory=persist_dir, embedding_function=factory.embeddings)
    return db

