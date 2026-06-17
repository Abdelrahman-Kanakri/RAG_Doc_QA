"""FastAPI application entry point — creates the app instance and mounts the API router."""

from fastapi import FastAPI
from app.api import router
import uvicorn


app = FastAPI(title = "RAG Document QA", version = "v1.0",
            description = "A Retrieval-Augmented Generation (RAG) system for document question answering, leveraging Mistral AI for query expansion and Chroma for vector storage.")

app.include_router(router, prefix = "/api/v1")

if __name__ == "__main__":
    uvicorn.run(app, host="128.127.0.1", port=8080)
