import numpy as np
import chromadb 
from chromadb import Settings 
import uuid
from langchain_chroma import Chroma
from typing import List,Dict,Any,Tuple,Optional
from src.retrieval.embedding_manager import get_embeddings_model
from src.config import settings
import os
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever


class VectorStore:
    """ Manages document embeddings in chromaDB vector store"""
    
    def __init__(self,collection_name:str="pdf_document",persistent_directory:Optional[str]=None):
        """
        Args:
            collection_name : Name of chromaDB
            persistent_directory: Directory to persist the vector store 
        """
        self.collection_name = collection_name
        self.persistent_directory = persistent_directory or settings.CHROMA_PERSISTENT_DIRECTORY
        self.embedding_model = get_embeddings_model()
        self.vector_store = self._initialize_store()
        
        
    def _initialize_store(self):
        """ Intialize the chromaDB client and collection"""
        
        os.makedirs(self.persistent_directory,exist_ok=True)
        return Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embedding_model,
            persist_directory=self.persistent_directory
        )
    
    def index_documents(self,documents:List[Document])->List[str]:
        """
        Add documents and their embeddings to the vector store 
        
        Args:
            documents: List of langchain Documents
        """
        
        
        
        print(f"Adding {len(documents)} documents to vector store...")
        
        ids = []
        
        
        for i,doc in enumerate(documents):
            
            #Generate ids
            doc_id = f"{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)
            
            
            #adding metadata
            
            doc.metadata['doc_index'] = i
            doc.metadata['content_length'] = len(doc.page_content)
            
        
        try:
            added_ids =  self.vector_store.add_documents(
                documents=documents,
                ids=ids
            )
            print(f"Successfully add {len(documents)} documents into the vector store")
            
        except Exception as e:
            print(f"Error adding documents to vector store {e}")
            raise
        
        return added_ids
    
    def similarity_search(self,query:str,k: int = 4) ->List[Document]:
        
        try:
            
            results = self.vector_store.similarity_search(query=query,k=k)
            return results
        except Exception as e:
            print(f"Error featching from VectorDB {e}")
            raise
        
    def similarity_search_with_score(self,query:str,k:int =4) ->List[Tuple[Document,float]]:
        try:
            
            results = self.vector_store.similarity_search_with_score(query=query,k=k)
            return results
        
        except Exception as e:
            print(f"Error featching from VectorDB {e}")
            raise
        
    def as_retriever(self,search_kwargs:Optional[dict] = None)->BaseRetriever:
        
        try:
            return self.vector_store.as_retriever(search_kwargs=search_kwargs or {"k":4})
        except Exception as e:
            print(f"Error {e}")
            raise