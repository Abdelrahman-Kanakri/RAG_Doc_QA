from fastapi import APIRouter, UploadFile, HTTPException
import shutil

from app.generation.answerer import generate_answer
from app.retrieval.vectorstore import init_vectorStore, add_chunks, view_collection
from app.retrieval.multiquery import multi_query_retrieve
from app.ingestion.loaders import choose_loader
from app.ingestion.chunking import split_documents
from pathlib import Path

router = APIRouter()

@router.get("/health")
async def check_health():
    """ Check the health of the API and the connection to the vector store. """
    try:
        # Initialize the vector store to check if it's connected
        init_vectorStore()
        vector_store_status = "connected"
    except:
        vector_store_status = "disconnected"
    return {
        "status": "healthy",
        "vector_store": vector_store_status}

@router.post("/ingest")
async def ingest_document(file: UploadFile):
    """  Ingest a document by uploading a file, splitting it into chunks, and adding the chunks to the vector store.
    
    flow: 
    
        1. Upload a file (PDF or Markdown) to the /ingest endpoint.
        
        2. The API validates the file type and saves it to the server if no, return 415 error code.
        
        3. The file is loaded using the appropriate loader based on its type.
        
        4. The loaded document is split into smaller chunks using the split_documents function.
        
        5. The chunked documents are added to the Chroma vector store using the add_chunks function.
        
        6. The API returns a response indicating the success of the ingestion process, including the number of chunks created and the document ID.
    
    """
    # Set the directory path for uploaded files
    DIR_PATH = "data/docs/uploads"
    # Load the document using the appropriate loader
    file_extension = file.filename.split(".")[-1].lower()
    file_location = f"{DIR_PATH}/{file.filename}"
    
    # Check file type. Only PDF and Markdown files are supported.
    if not file_extension in ("pdf", "md"):
        raise HTTPException(status_code = 415, detail="Unsupported file type. Only PDF and Markdown files are supported.")
    
    # Check if the directory exists, if not create it.
    Path(DIR_PATH).mkdir(parents=True, exist_ok=True)
    
    # store the file in the disk/server
    with open(file_location, "wb") as buffer: 
            shutil.copyfileobj(file.file, buffer)
        
    try:
        # Ingestion Process: Load the document, split it into chunks, and add the chunks to the vector store.
        file_content = choose_loader(file_location) # This is to validate the file before saving it to disk and ingesting it into the vector store.
        chunks = split_documents(file_content)
        # A chunk guard to ensure that the document was split into chunks successfully. If not, raise an HTTPException with a 422 status code and a relevant error message.
        if not chunks:
            raise HTTPException(status_code = 422, detail="Failed to split the document into chunks. Please check the file content.")
        vector_store = init_vectorStore()
        add_chunks(vector_store, chunks)
    except HTTPException:
            Path(file_location).unlink(missing_ok=True)
            raise
    except Exception as e:
        Path(file_location).unlink(missing_ok=True)
        raise HTTPException(status_code = 500, detail=f"An error occurred during the ingestion process: {str(e)}")
    
    return {
        "document_id": chunks[0].metadata.get("document_id", "unknown"),
        "chunks": len(chunks),
        "message": f"Document '{file.filename}' ingested successfully with {len(chunks)} chunks."
    }

@router.post("/query")
async def query_document(query: str):
    """ Query the knowledge base with a user query and retrieve relevant documents. """ 
    retrieved_docs = multi_query_retrieve(query, init_vectorStore())
    
    # Check if there is a relevant docs for the given query. If not, raise an HTTPException with a 404 status code and a relevant error message.
    if not retrieved_docs:
        raise HTTPException(status_code = 404, detail="No relevant documents found for the given query.")
    response = generate_answer(query, retrieved_docs)
    return { 
            "answer": response["answer"],
            "sources": response["sources"]
        }