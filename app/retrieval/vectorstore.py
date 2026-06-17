"""ChromaDB vector store helpers — initialization, chunk ingestion, and collection inspection."""

from langchain_chroma import Chroma
from langchain_mistralai import MistralAIEmbeddings
from langchain_core.documents import Document
from typing import List

from app.core import settings
from app.ingestion import choose_loader
from app.ingestion import split_documents
import os 

# Set the API KEY for the MistralAI model 
os.environ["MISTRAL_API_KEY"] = settings.MISTRAL_API_KEY

def init_vectorStore(collection_name: str = settings.COLLECTION_NAME
                    , persist_directory: str = settings.VECTOR_DATABASE_PATH) -> Chroma:
    """ Initialize the Chroma vector store. """
    # The vector store will be created automatic in the persist_directory if it does not exist.
    try:
        return Chroma(collection_name = collection_name,
                persist_directory = persist_directory, 
                embedding_function = MistralAIEmbeddings(model= settings.EMBEDDING_MODEL_NAME))
    except Exception as e:
            raise Exception(f"Failed to initialize vector store — check Mistral API key and ChromaDB path: {e}")
# Delete the vector store collection if it exists
def _delete_vectorStore_collection(vectorStore: Chroma) -> None:
    """ Delete the Chroma vector store. """
    vectorStore.delete_collection()

# Add chunked documents to the vector store
def add_chunks(vector_store: Chroma, 
            documents: List[Document]) -> None:
    """ Add chunked documents to the vector store. """
    result = vector_store._collection.get(ids = [doc.metadata.get("chunk_id") for doc in documents])
    existed_ids = set(result['ids'])
    
    # Filter out the documents that already exist in the vector store
    new_docs = [docs for docs in documents if docs.metadata.get("chunk_id") not in existed_ids]        
    
    # Early return if there are no new documents to add
    if not new_docs:
        print("No new documents to add.")
        return
    # Add new documents to the vector store
    vector_store.add_documents(documents = new_docs,
                            ids = [doc.metadata.get("chunk_id") for doc in new_docs],
                        # Metadata is already included in the Document objects
                        # and extracted via the add_documents method 
                        )

# View the collection information
def view_collection(vector_store: Chroma, collection_name: str = settings.COLLECTION_NAME) -> dict:
    """ View the information of the default Chroma vector store collection., or set a specific collection name. """
    return {"collection name": collection_name,
            "collection chunks available": vector_store._collection.count()}