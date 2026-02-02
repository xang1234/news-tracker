"""Storage layer for document persistence."""

from src.storage.database import Database, get_database
from src.storage.repository import DocumentRepository

__all__ = ["Database", "get_database", "DocumentRepository"]
