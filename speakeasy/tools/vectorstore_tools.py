from langchain.chains import RetrievalQA
from .custom_tools import CustomBaseTool

#TODO Consider merging Image and Document query into a single Vectorstore query tool using the Vectorstore router

# Class representing tools for document Q & A. 
class CustomDocumentQueryTool(CustomBaseTool):
    def __init__(self, fa, db, return_direct = False):
        super(CustomDocumentQueryTool, self).__init__(fa,
                name="Document_Query", 
                description="Use this tool only to query about documents stored LOCALLY",
                db = db,
                return_direct = return_direct
            )

    def _run(self, query: str) -> str:
        """Use the tool."""
        chain = RetrievalQA.from_chain_type(llm=self.factory.llm,
            chain_type="stuff",
            retriever=self.vector_db.as_retriever(search_kwargs={"k":self.factory.max_k}),
            input_key="question",
            return_source_documents=True)
        return chain(query)['result'] # Should enhance to return the source document paths

# Class representing tools for image Q & A (based on captions previously generated)
class CustomImageQueryTool(CustomBaseTool):
    def __init__(self, fa, db , return_direct = False):
        super(CustomImageQueryTool, self).__init__(fa,
                name="Image_Query", 
                description="Use this tool to only query the captions of images stored LOCALLY",
                db = db,
                return_direct = return_direct
            )

    def _run(self, query: str) -> str:
        chain = RetrievalQA.from_chain_type(llm=self.factory.llm,
            chain_type="stuff",
            retriever=self.vector_db.as_retriever(search_kwargs={"k":self.factory.max_k}),
            input_key="question",
            return_source_documents=True)
        return chain(query)['result'] 

