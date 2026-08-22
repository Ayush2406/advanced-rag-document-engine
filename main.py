from src.ingestion.document_loader import pdf_loader
from src.ingestion.text_splitter import split_documents
from src.retrieval.vector_store import VectorStore
from src.config import settings
from langchain_core.documents import Document
from src.pipeline import rag_pipeline
import argparse

def run_ingestion(vector_store:VectorStore):
    """"Loading PDFs and storing them in VectorDB"""
    print("---Ingestion Started---\n")
    documents = pdf_loader(settings.RAW_DOCS_DIR)
    chunks = split_documents(documents=documents)
    vector_store.index_documents(chunks)
    print("---Ingestion Completed---\n")

def main():
    
    parser = argparse.ArgumentParser(description="Advanced RAG Document Engine")
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Run Document ingestion (load, split, embed, store) before quering"
    )
    parser.add_argument(
        "--query",
        type=str,
        default="",
        help="The query to ask the RAG pipeline"
    )

    args = parser.parse_args()
    
    vector_store = VectorStore()
    
    if args.ingest:
        run_ingestion(vector_store=vector_store)
        
    if args.query:
        chain = rag_pipeline(vector_store=vector_store)
        print(f"\nAsking question: {args.query}")
        answer = chain.invoke(args.query)
        print("\n ===Final Answer===")
        print(answer)
    
if __name__ == "__main__":
    main()
