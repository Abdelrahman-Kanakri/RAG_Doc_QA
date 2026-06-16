from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from pathlib import Path

def load_pdf(file_path: str) -> list:
    """ Load a PDF file and return a list of documents. """
    # Initialize the PyPDFLoader with the file path and extraction mode
    loader = PyPDFLoader(
        file_path=file_path,
        mode = 'page',
        extraction_mode = 'plain',
    )
    content_pdf = loader.load()
    return content_pdf 

def load_markdown(file_path: str) -> list:
    """ Load a markdown file and return a list of documents."""
    
    # Read the content of the markdown file
    doc = Path(file_path).read_text()
    
    return [Document(page_content=doc, metadata={"source": file_path})]
def choose_loader(file_path: str)-> list: 
    """ Choose the appropriate loader based on the file extension."""
    if file_path.endswith(".pdf"):
        return load_pdf(file_path)
    elif file_path.endswith(".md"):
        return load_markdown(file_path)
    else:
        raise ValueError("Unsupported file type. Only PDF and Markdown files are supported.")

