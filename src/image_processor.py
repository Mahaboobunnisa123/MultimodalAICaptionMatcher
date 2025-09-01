"""
Image Processor Module for Multimodal AI Caption Matcher

This module handles image loading, preprocessing, and preparation for CLIP model inference.
Paraphrased implementation based on the original multimodal AI project.
"""

import torch
from PIL import Image
from transformers import CLIPProcessor
from pathlib import Path
import logging
from typing import Union, Tuple, Optional
from configs.config import IMAGE_EXTENSIONS, MAX_IMAGE_SIZE

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageProcessor:
    """
    Handles image loading, validation, and preprocessing for the CLIP model.
    """
    
    def __init__(self, model_identifier: str = "openai/clip-vit-base-patch32"):
        """
        Initialize the image processor with CLIP processor.
        
        Args:
            model_identifier (str): Hugging Face model identifier for CLIP
        """
        self.model_identifier = model_identifier
        self.clip_processor = None
        self._initialize_processor()
    
    def _initialize_processor(self) -> None:
        """Load and initialize the CLIP processor."""
        try:
            self.clip_processor = CLIPProcessor.from_pretrained(self.model_identifier)
            logger.info(f"Successfully initialized CLIP processor: {self.model_identifier}")
        except Exception as e:
            logger.error(f"Failed to initialize CLIP processor: {e}")
            raise
    
    def validate_image_path(self, img_path: Union[str, Path]) -> Path:
        """
        Validate image file path and extension.
        
        Args:
            img_path (Union[str, Path]): Path to the image file
            
        Returns:
            Path: Validated image path
            
        Raises:
            FileNotFoundError: If image file doesn't exist
            ValueError: If image format is not supported
        """
        image_path = Path(img_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image format: {image_path.suffix}")
        
        return image_path
    
    def load_image_data(self, img_path: Union[str, Path]) -> Image.Image:
        """
        Load and validate image from file path.
        
        Args:
            img_path (Union[str, Path]): Path to the image file
            
        Returns:
            Image.Image: Loaded PIL image in RGB format
        """
        validated_path = self.validate_image_path(img_path)
        
        try:
            # Load image and convert to RGB
            pil_image = Image.open(validated_path).convert("RGB")
            
            # Resize if image is too large
            if pil_image.size[0] > MAX_IMAGE_SIZE[0] or pil_image.size[1] > MAX_IMAGE_SIZE[1]:
                pil_image.thumbnail(MAX_IMAGE_SIZE, Image.Resampling.LANCZOS)
                logger.info(f"Resized image to: {pil_image.size}")
            
            logger.info(f"Successfully loaded image: {validated_path.name}")
            return pil_image
            
        except Exception as e:
            logger.error(f"Error loading image {validated_path}: {e}")
            raise
    
    def preprocess_image_tensor(self, pil_image: Image.Image) -> Tuple[dict, CLIPProcessor]:
        """
        Preprocess image for CLIP model inference.
        
        Args:
            pil_image (Image.Image): PIL image to preprocess
            
        Returns:
            Tuple[dict, CLIPProcessor]: Processed image tensors and processor
        """
        try:
            # Process image using CLIP processor
            processed_inputs = self.clip_processor(
                images=pil_image, 
                return_tensors="pt"
            )
            
            logger.info("Image preprocessing completed successfully")
            return processed_inputs, self.clip_processor
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            raise
    
    def process_image_pipeline(self, img_path: Union[str, Path]) -> Tuple[dict, CLIPProcessor]:
        """
        Complete image processing pipeline from path to tensor.
        
        Args:
            img_path (Union[str, Path]): Path to the image file
            
        Returns:
            Tuple[dict, CLIPProcessor]: Processed image tensors and processor
        """
        # Load image
        pil_image = self.load_image_data(img_path)
        
        # Preprocess for model
        processed_data, processor = self.preprocess_image_tensor(pil_image)
        
        return processed_data, processor
    
    def get_image_info(self, img_path: Union[str, Path]) -> dict:
        """
        Get metadata information about an image.
        
        Args:
            img_path (Union[str, Path]): Path to the image file
            
        Returns:
            dict: Image metadata information
        """
        validated_path = self.validate_image_path(img_path)
        pil_image = Image.open(validated_path)
        
        return {
            "filename": validated_path.name,
            "format": pil_image.format,
            "mode": pil_image.mode,
            "size": pil_image.size,
            "width": pil_image.width,
            "height": pil_image.height
        }