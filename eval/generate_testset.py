import os
os.environ.setdefault("OPENSSL_CONF", "/dev/null")

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.testset import TestsetGenerator
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings 
import numpy as np
import pandas as pd
from app.ingestion import load_directory
from app.core import settings
import os 

def generate_testset():
    """Generate a test set using the specified LLM and Embeddings models."""
    
    # Set the Mistral API key in the environment variables
    os.environ["MISTRAL_API_KEY"] = settings.MISTRAL_API_KEY

    # Set the data path
    data_path = "data/docs/Nueral networks"

    # Load the documents from the specified directory
    documents = load_directory(data_path)[:10]

    # Initialize the LLM and Embeddings wrappers with the specified models and API key  
    generator_llm = LangchainLLMWrapper(ChatMistralAI(model=settings.LARGE_MODEL_NAME, api_key=settings.MISTRAL_API_KEY, temperature=0.7, max_tokens=2048))
    generator_embeddings = LangchainEmbeddingsWrapper(MistralAIEmbeddings(model=settings.EMBEDDING_MODEL_NAME, api_key=settings.MISTRAL_API_KEY))

    # Generate the test set using the loaded documents and the initialized LLM and Embeddings wrappers
    generator = TestsetGenerator(llm=generator_llm, embedding_model=generator_embeddings)
    dataset = generator.generate_with_langchain_docs(documents, testset_size=20)
    
    # Convert the generated dataset to a pandas DataFrame and save it as a CSV file
    df = dataset.to_pandas()
    df.head()
    df.to_csv("dataset/testset.csv", index=False)
if __name__ == "__main__":
    generate_testset()