"""
Caption Matcher Module for Multimodal AI Caption Matcher

This module handles similarity calculation and caption ranking using cosine similarity.
Paraphrased implementation based on the original multimodal AI project.
"""

import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import CLIPProcessor, CLIPModel
from typing import List, Tuple, Dict, Optional
import logging
from configs.config import PREDEFINED_CAPTIONS, DEFAULT_TOP_K

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CaptionMatcher:
    """
    Handles caption-to-image matching using semantic similarity in CLIP embedding space.
    """
    
    def __init__(self, caption_database: Optional[List[str]] = None):
        """
        Initialize the caption matcher with a caption database.
        
        Args:
            caption_database (Optional[List[str]]): List of candidate captions
        """
        self.caption_database = caption_database or PREDEFINED_CAPTIONS
        self.caption_count = len(self.caption_database)
        logger.info(f"Initialized caption matcher with {self.caption_count} captions")
    
    def compute_text_embeddings(self, caption_list: List[str], clip_model: CLIPModel, 
                               text_processor: CLIPProcessor) -> torch.Tensor:
        """
        Compute embeddings for a list of text captions.
        
        Args:
            caption_list (List[str]): List of text captions
            clip_model (CLIPModel): CLIP model instance
            text_processor (CLIPProcessor): CLIP text processor
            
        Returns:
            torch.Tensor: Text feature embeddings
        """
        try:
            # Process text captions
            text_inputs = text_processor(
                text=caption_list, 
                return_tensors="pt", 
                padding=True,
                truncation=True
            )
            
            # Move to same device as model
            device = next(clip_model.parameters()).device
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
            
            # Generate text embeddings
            with torch.no_grad():
                text_embeddings = clip_model.get_text_features(**text_inputs)
            
            # Normalize embeddings
            normalized_embeddings = torch.nn.functional.normalize(text_embeddings, dim=-1)
            
            logger.info(f"Generated text embeddings for {len(caption_list)} captions")
            return normalized_embeddings
            
        except Exception as e:
            logger.error(f"Error computing text embeddings: {e}")
            raise
    
    def calculate_similarity_scores(self, image_embeddings: np.ndarray, 
                                  text_embeddings: np.ndarray) -> np.ndarray:
        """
        Calculate cosine similarity between image and text embeddings.
        
        Args:
            image_embeddings (np.ndarray): Image feature embeddings
            text_embeddings (np.ndarray): Text feature embeddings
            
        Returns:
            np.ndarray: Similarity scores matrix
        """
        try:
            # Compute cosine similarity matrix
            similarity_matrix = cosine_similarity(image_embeddings, text_embeddings)
            
            logger.info(f"Computed similarity matrix with shape: {similarity_matrix.shape}")
            return similarity_matrix
            
        except Exception as e:
            logger.error(f"Error calculating similarity scores: {e}")
            raise
    
    def rank_captions_by_similarity(self, similarity_scores: np.ndarray, 
                                   caption_list: List[str], 
                                   top_k: int = DEFAULT_TOP_K) -> Tuple[List[str], List[float]]:
        """
        Rank captions based on similarity scores and return top-k results.
        
        Args:
            similarity_scores (np.ndarray): Similarity scores for each caption
            caption_list (List[str]): List of candidate captions
            top_k (int): Number of top captions to return
            
        Returns:
            Tuple[List[str], List[float]]: Top captions and their similarity scores
        """
        try:
            # Get similarity scores for the first (and only) image
            scores = similarity_scores[0]
            
            # Get indices sorted by similarity (descending order)
            ranked_indices = np.argsort(scores)[::-1]
            
            # Select top-k captions and scores
            top_k = min(top_k, len(caption_list))
            top_indices = ranked_indices[:top_k]
            
            top_captions = [caption_list[idx] for idx in top_indices]
            top_scores = [float(scores[idx]) for idx in top_indices]
            
            logger.info(f"Ranked top {top_k} captions by similarity")
            return top_captions, top_scores
            
        except Exception as e:
            logger.error(f"Error ranking captions: {e}")
            raise
    
    def find_best_matches(self, image_embeddings: torch.Tensor, 
                         candidate_captions: List[str],
                         clip_model: CLIPModel, 
                         text_processor: CLIPProcessor,
                         top_k: int = DEFAULT_TOP_K) -> Tuple[List[str], List[float]]:
        """
        Find the best matching captions for given image embeddings.
        
        Args:
            image_embeddings (torch.Tensor): Image feature embeddings
            candidate_captions (List[str]): List of candidate captions
            clip_model (CLIPModel): CLIP model instance
            text_processor (CLIPProcessor): CLIP text processor
            top_k (int): Number of top matches to return
            
        Returns:
            Tuple[List[str], List[float]]: Best matching captions and scores
        """
        # Compute text embeddings
        text_embeddings = self.compute_text_embeddings(
            candidate_captions, clip_model, text_processor
        )
        
        # Convert to numpy arrays for similarity computation
        image_numpy = image_embeddings.detach().cpu().numpy()
        text_numpy = text_embeddings.detach().cpu().numpy()
        
        # Calculate similarity scores
        similarity_scores = self.calculate_similarity_scores(image_numpy, text_numpy)
        
        # Rank and return top matches
        best_captions, scores = self.rank_captions_by_similarity(
            similarity_scores, candidate_captions, top_k
        )
        
        return best_captions, scores
    
    def match_image_captions(self, image_embeddings: torch.Tensor, 
                           clip_model: CLIPModel, 
                           text_processor: CLIPProcessor,
                           custom_captions: Optional[List[str]] = None,
                           top_k: int = DEFAULT_TOP_K) -> Tuple[List[str], List[float]]:
        """
        Main method to match image with captions from the database.
        
        Args:
            image_embeddings (torch.Tensor): Image feature embeddings
            clip_model (CLIPModel): CLIP model instance
            text_processor (CLIPProcessor): CLIP text processor
            custom_captions (Optional[List[str]]): Custom caption list (overrides database)
            top_k (int): Number of top matches to return
            
        Returns:
            Tuple[List[str], List[float]]: Best matching captions and their scores
        """
        # Use custom captions or default database
        captions_to_use = custom_captions or self.caption_database
        
        logger.info(f"Matching image against {len(captions_to_use)} captions")
        
        # Find best matches
        best_matches, similarity_scores = self.find_best_matches(
            image_embeddings, captions_to_use, clip_model, text_processor, top_k
        )
        
        return best_matches, similarity_scores
    
    def update_caption_database(self, new_captions: List[str], append: bool = True) -> None:
        """
        Update the caption database with new captions.
        
        Args:
            new_captions (List[str]): New captions to add
            append (bool): Whether to append or replace existing captions
        """
        if append:
            self.caption_database.extend(new_captions)
        else:
            self.caption_database = new_captions
        
        self.caption_count = len(self.caption_database)
        logger.info(f"Updated caption database. New count: {self.caption_count}")
    
    def get_database_stats(self) -> Dict[str, int]:
        """
        Get statistics about the caption database.
        
        Returns:
            Dict[str, int]: Database statistics
        """
        return {
            "total_captions": self.caption_count,
            "avg_caption_length": sum(len(cap.split()) for cap in self.caption_database) // self.caption_count,
            "min_caption_length": min(len(cap.split()) for cap in self.caption_database),
            "max_caption_length": max(len(cap.split()) for cap in self.caption_database)
        }