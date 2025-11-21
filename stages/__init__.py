"""
Stages package for Unstructured ParseQuery pipeline

Contains individual stage modules:
- ingest: UC volume upload
- parse: Document parsing with ai_parse_document
- categorize: Document categorization with ai_query
- extract: Entity extraction with ai_query
- deidentify: PII removal with ai_query
"""

from .ingest import ingest_file
from .parse import parse_document
from .categorize import categorize_document
from .extract import extract_entities
from .deidentify import deidentify_document

__all__ = [
    "ingest_file",
    "parse_document",
    "categorize_document",
    "extract_entities",
    "deidentify_document"
]
