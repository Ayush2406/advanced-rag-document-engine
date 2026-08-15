from langchain_community.document_loaders import PyMuPDFLoader
from pathlib import Path
from typing import List
from langchain_core.documents import Document

def pdf_loader(pdf_directory)-> List[Document]:
    
    documents = []
    pdf_dir = Path(pdf_directory)
    
    # Finding all pdf files Recursively
    
    pdf_files = list(pdf_dir.glob("**/*.pdf"))
    
    print(f"Found {len(pdf_files)} PDF documents.")
    
    for pdf_file in pdf_files:
        
        print(f"Processing {pdf_file.name}")
        
        try:
            loader = PyMuPDFLoader(str(pdf_file))
            document = loader.load()
            
            # adding metadata
            for doc in document:
                doc.metadata['source_file'] = pdf_file.name
                doc.metadata['type'] = "pdf"
                
            documents.extend(document)
            print(f" ✔ Loaded {len(documents)} pages")
        except Exception as e:
            print(f"Error {e}")
    
    print(f"\nTotal documents loaded {len(documents)}")
    
    return documents

