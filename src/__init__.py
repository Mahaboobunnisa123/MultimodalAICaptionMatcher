"""
Source code initialization file for the Multimodal AI Caption Matcher project.
"""

__version__ = "1.0.0"
__author__ = "AI Research Team"
__description__ = "Multimodal AI Caption Matcher using CLIP embeddings"

# Import main classes for easy access
from .image_processor import ImageProcessor
from .embedding_generator import EmbeddingGenerator  
from .caption_matcher import CaptionMatcher

__all__ = [
    "ImageProcessor",
    "EmbeddingGenerator", 
    "CaptionMatcher"
]