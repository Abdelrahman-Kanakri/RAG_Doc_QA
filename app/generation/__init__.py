"""Generation layer — LLM-based answer synthesis over retrieved document chunks."""

from .answerer import generate_answer

__all__ = ["generate_answer"]
