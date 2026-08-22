from langchain_groq import ChatGroq
from src.config import settings
from langchain_google_genai import ChatGoogleGenerativeAI

def get_llm(temperature:float = 0.0)->ChatGroq:
    return ChatGroq(
        groq_api_key = settings.GROQ_API_KEY,
        model_name = settings.GROQ_MODEL_NAME,
        temperature=temperature,
        max_tokens=4096,
        max_retries=2
    )
    
def get_judge_llm(temperature:float =0.0) ->ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model = "gemini-3.1-flash-lite",
        api_key = settings.GOOGLE_GEMINI_KEY,
        temperature = temperature,
        transport = "rest",
        max_retries = 2,
    )