from src.ingestion.document_loader import pdf_loader
from src.ingestion.text_splitter import split_documents
from src.retrieval.vector_store import VectorStore
from src.config import settings
from langchain_core.documents import Document
from src.retrieval.retriever import RAGRetriever
from src.generation.llm_service import get_llm
from src.generation.prompts import get_rag_prompt

# Loading and converting pdf to langchain documents -> List[Document]
documents = pdf_loader(settings.RAW_DOCS_DIR)

# Splitting documents into smaller chunks -> List[Document]
chunks = split_documents(documents=documents)

# Storing in Vector DB
vector_store = VectorStore()

vector_store.index_documents(documents=chunks)

# Retrieve
retriever = RAGRetriever(vector_store=vector_store)
query = "What is probability"
results= retriever.retrieve(query=query)

formated_context = "\n\n".join([doc['content']for doc in results]) if results else ""
prompt_template = get_rag_prompt()
formatted_prompt = prompt_template.invoke({
    "context":formated_context,
    "question":query
})

llm = get_llm()
response = llm.invoke(formatted_prompt)
print(response.content)


