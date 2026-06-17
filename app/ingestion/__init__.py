"""Ingestion pipeline — document loading and chunking."""

from .loaders import choose_loader, load_pdf, load_markdown, load_directory
from .chunking import split_documents

__all__ = ["choose_loader", "load_pdf", "load_markdown", "load_directory", "split_documents"]
