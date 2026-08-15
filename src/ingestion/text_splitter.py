from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
from langchain_core.documents import Document
from src.config import settings

def split_documents(documents,chunk_size:int = settings.CHUNK_SIZE,chunk_overlap:int = settings.CHUNK_OVERLAP)-> List[Document]:
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap,
        length_function = len,
        separators = ['\n\n','\n',' ','']
    )
    
    split_docs = text_splitter.split_documents(documents=documents)
    
    print(f"Split {len(documents)} into {len(split_docs)} chunks\n")
    
    if split_docs:
        print(f"Example chunk")
        print(f"Content {split_docs[0].page_content[:200]}")
        print(f"Meatadata {split_docs[0].metadata}")
        
    return split_docs
    