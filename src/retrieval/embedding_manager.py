import torch
from src.config import settings
from langchain_huggingface import HuggingFaceEmbeddings

def get_embeddings_model() -> HuggingFaceEmbeddings:
    
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
        
    embeddings = HuggingFaceEmbeddings(
        model_name = settings.EMBEDDING_MODEL_NAME,
        model_kwargs={"device":device},
        encode_kwargs = {"normalize_embeddings":True}
    )
    
    return embeddings