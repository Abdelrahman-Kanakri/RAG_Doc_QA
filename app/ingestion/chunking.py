"""Splits documents into overlapping chunks and enriches each chunk with a unique hash ID and source metadata."""

import hashlib
from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.core import settings


def split_documents(documents: List[Document],
                    chunk_size: int = settings.CHUNK_SIZE,
                    chunk_overlap: int = settings.CHUNK_OVERLAP) -> List[Document]:
    """ Split documents into smaller chunks using RecursiveCharacterTextSplitter. """

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunked_documents = text_splitter.split_documents(documents)
    if not chunked_documents:
        raise ValueError("No chunks were created from the documents. Please check the input documents and the chunking parameters.")
    for idx, chunk in enumerate(chunked_documents):
        # Generate a unique hash for the Chunk content
        chunk_hash = hashlib.md5((chunk.page_content + str(idx)).encode('utf-8')).hexdigest()
        
        # Get the document ID from the original document's metadata for the chunk metadata
        doc_id =  Path(chunk.metadata.get("source", "")).name
        
        # Update the metadata with the hash and chunk index
        chunk.metadata.update({
            "chunk_id": chunk_hash,
            "chunk_index": idx,
            "document_id": doc_id,
            # Placeholder for keywords, can be populated later
            "keywords": "", 
            "chunk_headers": "",
        })
    
    return chunked_documents