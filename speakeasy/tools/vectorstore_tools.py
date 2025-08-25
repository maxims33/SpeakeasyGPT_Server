"""
Module with tools for document Q & A.
#TODO Consider merging Image and Document query into Vectorstore router
"""

from langchain.chains import RetrievalQA
from .custom_tools import CustomBaseTool

class CustomDocumentQueryTool(CustomBaseTool):
    """
    Class representing tools for document Q & A.
    """
    def __init__(self, fact, vdb, return_direct = False):
        super().__init__(fact,
                name="Document_Query",
                description="Use this tool only to query about documents stored LOCALLY",
                vdb = vdb,
                return_direct = return_direct
            )

    def _run(self, query: str) -> str:
        """Use the tool."""
        chain = RetrievalQA.from_chain_type(llm=self.factory.llm,
            chain_type="stuff",
            retriever=self.vector_db.as_retriever(search_kwargs={"k":self.factory.max_k}),
            input_key="question",
            return_source_documents=True)
        return chain.invoke(query)['result'] # Should enhance to return the source document paths

    async def _arun(self, query: str) -> str:
        raise NotImplementedError("does not support async")

class CustomImageQueryTool(CustomBaseTool):
    """
    Class representing tools for image Q & A (based on captions previously generated)
    """
    def __init__(self, fact, vdb , return_direct = False):
        super().__init__(fact,
                name="Image_Query",
                description="Use this tool to only query the captions of images stored LOCALLY",
                vdb = vdb,
                return_direct = return_direct
            )

    def _run(self, query: str) -> str:
        chain = RetrievalQA.from_chain_type(llm=self.factory.llm,
            chain_type="stuff",
            retriever=self.vector_db.as_retriever(search_kwargs={"k":self.factory.max_k}),
            input_key="question",
            return_source_documents=True)
        return chain.invoke(query)['result']

    async def _arun(self, query: str) -> str:
        raise NotImplementedError("does not support async")
