"""Bayesian authority scoring for content authors.

Components:
- AuthorityConfig: Pydantic settings for scoring parameters
- AuthorityProfile: Dataclass mapping to the authority_profiles table
- AuthorTier: Enum for base weight classification (anonymous/verified/research)
- AuthorityService: Bayesian scoring with time decay and probation ramp
- AuthorityRepository: CRUD operations for authority profile persistence
"""

from src.authority.config import AuthorityConfig
from src.authority.repository import AuthorityRepository
from src.authority.schemas import AuthorityProfile, AuthorTier
from src.authority.service import AuthorityService

__all__ = [
    "AuthorityConfig",
    "AuthorityProfile",
    "AuthorityRepository",
    "AuthorityService",
    "AuthorTier",
]
