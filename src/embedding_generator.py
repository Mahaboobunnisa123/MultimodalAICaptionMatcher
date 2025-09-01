"""
Embedding Generator Module for Multimodal AI Caption Matcher

This module handles feature embedding extraction from images using the CLIP model.
"""

import torch
from transformers import CLIPModel
import logging
from typing import Tuple
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    """
    Generates semantic embeddings from images using the CLIP vision encoder.
    """
    
    def __init__(self, model_identifier: str = "openai/clip-vit-base-patch32", device: str = "auto"):
        """
        Initialize the embedding generator with CLIP model.
        
        Args:
            model_identifier (str): Hugging Face model identifier for CLIP
            device (str): Device to run the model on ("auto", "cuda", "cpu")
        """
        self.model_identifier = model_identifier
        self.device = self._determine_device(device)
        self.clip_model = None
        self._load_clip_model()
    
    def _determine_device(self, device_preference: str) -> str:
        """
        Determine the best available device for model inference.
        
        Args:
            device_preference (str): User's device preference
            
        Returns:
            str: Selected device ("cuda" or "cpu")
        """
        if device_preference == "auto":
            selected_device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            selected_device = device_preference
        
        logger.info(f"Using device: {selected_device}")
        return selected_device
    
    def _load_clip_model(self) -> None:
        """Load and initialize the CLIP model."""
        try:
            self.clip_model = CLIPModel.from_pretrained(self.model_identifier)
            self.clip_model.to(self.device)
            self.clip_model.eval()  # Set to evaluation mode
            
            logger.info(f"Successfully loaded CLIP model: {self.model_identifier}")
            logger.info(f"Model loaded on device: {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
            raise
    
    def extract_visual_features(self, processed_image_data: dict) -> torch.Tensor:
        """
        Extract visual feature embeddings from preprocessed image data.
        
        Args:
            processed_image_data (dict): Preprocessed image tensors from CLIPProcessor
            
        Returns:
            torch.Tensor: Image feature embeddings
        """
        try:
            # Move input tensors to the same device as model
            device_inputs = {
                key: value.to(self.device) if isinstance(value, torch.Tensor) else value
                for key, value in processed_image_data.items()
            }
            
            # Extract image features without gradient computation
            with torch.no_grad():
                visual_features = self.clip_model.get_image_features(**device_inputs)
            
            logger.info(f"Extracted visual features with shape: {visual_features.shape}")
            return visual_features
            
        except Exception as e:
            logger.error(f"Error extracting visual features: {e}")
            raise
    
    def extract_textual_features(self, processed_text_data: dict) -> torch.Tensor:
        """
        Extract textual feature embeddings from preprocessed text data.
        
        Args:
            processed_text_data (dict): Preprocessed text tensors from CLIPProcessor
            
        Returns:
            torch.Tensor: Text feature embeddings
        """
        try:
            # Move input tensors to the same device as model
            device_inputs = {
                key: value.to(self.device) if isinstance(value, torch.Tensor) else value
                for key, value in processed_text_data.items()
            }
            
            # Extract text features without gradient computation
            with torch.no_grad():
                textual_features = self.clip_model.get_text_features(**device_inputs)
            
            logger.info(f"Extracted textual features with shape: {textual_features.shape}")
            return textual_features
            
        except Exception as e:
            logger.error(f"Error extracting textual features: {e}")
            raise
    
    def generate_image_embeddings(self, processed_image_data: dict) -> Tuple[torch.Tensor, CLIPModel]:
        """
        Generate normalized image embeddings ready for similarity comparison.
        
        Args:
            processed_image_data (dict): Preprocessed image tensors
            
        Returns:
            Tuple[torch.Tensor, CLIPModel]: Normalized image features and model instance
        """
        # Extract raw features
        raw_features = self.extract_visual_features(processed_image_data)
        
        # Normalize features for cosine similarity
        normalized_features = torch.nn.functional.normalize(raw_features, dim=-1)
        
        logger.info("Image embeddings generated and normalized successfully")
        return normalized_features, self.clip_model
    
    def convert_to_numpy(self, tensor_features: torch.Tensor) -> np.ndarray:
        """
        Convert tensor features to numpy array for further processing.
        
        Args:
            tensor_features (torch.Tensor): Feature tensors
            
        Returns:
            np.ndarray: Features as numpy array
        """
        try:
            numpy_features = tensor_features.detach().cpu().numpy()
            logger.info(f"Converted features to numpy array with shape: {numpy_features.shape}")
            return numpy_features
            
        except Exception as e:
            logger.error(f"Error converting tensor to numpy: {e}")
            raise
    
    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.
        
        Returns:
            dict: Model configuration and device information
        """
        return {
            "model_name": self.model_identifier,
            "device": self.device,
            "num_parameters": sum(p.numel() for p in self.clip_model.parameters()),
            "model_dtype": next(self.clip_model.parameters()).dtype,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
        }