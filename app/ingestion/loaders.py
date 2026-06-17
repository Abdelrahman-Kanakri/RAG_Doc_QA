"""Document loaders for PDF and Markdown files, with a dispatcher that selects the right loader by extension."""

from langchain_community.document_loaders import PyPDFLoader
from pypdf import PdfReader
from pypdf.errors import PdfReadError 
from langchain_core.documents import Document
from pathlib import Path
from typing import List

def load_pdf(file_path: str) -> list:
    """ Load a PDF file and return a list of documents. """
    try: 
        reader = PdfReader(file_path)    
        if reader.is_encrypted():
            raise PdfReadError("PDF is encrypted and cannot be read.")
        
        # Initialize the PyPDFLoader with the file path and extraction mode
        loader = PyPDFLoader(
        file_path=file_path,
        mode = 'page',
        extraction_mode = 'plain')
        content_pdf = loader.load()
    except PdfReadError as e:
        raise PdfReadError(f"Error loading PDF: {e}")
    except Exception as e:
        raise Exception(f"An unexpected error occurred while loading the PDF: {e}")
    return content_pdf 

def load_markdown(file_path: str) -> list:
    """ Load a markdown file and return a list of documents."""
    # Read the content of the markdown file
    if not Path(file_path).exists():
        raise FileNotFoundError(f"File not found: {file_path}")
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

def load_directory(root_path: str = "data/docs/") -> List[Document]:
    """ Load all supported files from a directory and return a list of documents. """
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    DATA_DIR = PROJECT_ROOT / root_path
    
    # print(f"Scanning: {DATA_DIR}, exists: {DATA_DIR.exists()}")
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")
    
    data_paths = [subdir for subdir in Path(DATA_DIR).rglob("*") if subdir.is_file()]
    # Even the if there is DATA_DIR exists,
    # there might not be any files in the directory, 
    # so we need to check if there are any files in the directory.
    # If not, raise a FileNotFoundError with a relevant error message.
    if not data_paths:
        raise FileNotFoundError(f"No files found in the data directory: {DATA_DIR}")
    documents = []
    for path in data_paths: 
            documents.extend(choose_loader(str(path)))
            print(f"Loaded {len(documents)} documents from {path}")
    return documents

