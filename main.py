"""
Main Multimodal AI Caption Matcher

This is the main entry point for the multimodal AI caption matching system.
Integrates all components to provide a complete image-to-caption matching solution.
"""

import argparse
import time
from pathlib import Path
from typing import List, Tuple, Optional
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.image_processor import ImageProcessor
from src.embedding_generator import EmbeddingGenerator
from src.caption_matcher import CaptionMatcher
from src.utils import (
    setup_project_logging, 
    format_results_display, 
    save_results_to_json, 
    save_results_to_csv,
    create_timestamp_filename,
    print_system_info,
    calculate_processing_time
)
from configs.config import (
    CLIP_MODEL_NAME, 
    DEFAULT_TOP_K, 
    PREDEFINED_CAPTIONS,
    setup_directories
)

class MultimodalCaptionMatcher:
    """
    Main class that orchestrates the complete multimodal AI caption matching pipeline.
    """
    
    def __init__(self, model_name: str = CLIP_MODEL_NAME, device: str = "auto"):
        """
        Initialize the complete multimodal AI system.
        
        Args:
            model_name (str): CLIP model identifier
            device (str): Device for model inference
        """
        self.model_name = model_name
        self.device = device
        self.logger = setup_project_logging()
        
        # Initialize components
        self.logger.info("Initializing Multimodal AI Caption Matcher...")
        self.image_processor = ImageProcessor(model_name)
        self.embedding_generator = EmbeddingGenerator(model_name, device)
        self.caption_matcher = CaptionMatcher()
        
        self.logger.info("System initialization completed successfully!")
    
    def process_single_image(self, image_path: str, 
                           custom_captions: Optional[List[str]] = None,
                           top_k: int = DEFAULT_TOP_K) -> Tuple[List[str], List[float]]:
        """
        Process a single image and find matching captions.
        
        Args:
            image_path (str): Path to the input image
            custom_captions (Optional[List[str]]): Custom caption list
            top_k (int): Number of top matches to return
            
        Returns:
            Tuple[List[str], List[float]]: Best matching captions and scores
        """
        start_time = time.time()
        
        try:
            # Step 1: Process image
            self.logger.info(f"Processing image: {image_path}")
            processed_image, processor = self.image_processor.process_image_pipeline(image_path)
            
            # Step 2: Generate image embeddings
            self.logger.info("Generating image embeddings...")
            image_embeddings, clip_model = self.embedding_generator.generate_image_embeddings(
                processed_image
            )
            
            # Step 3: Match with captions
            self.logger.info("Finding best caption matches...")
            best_captions, similarity_scores = self.caption_matcher.match_image_captions(
                image_embeddings, clip_model, processor, custom_captions, top_k
            )
            
            end_time = time.time()
            processing_info = calculate_processing_time(start_time, end_time)
            self.logger.info(f"Processing completed in {processing_info['duration_formatted']}")
            
            return best_captions, similarity_scores
            
        except Exception as e:
            self.logger.error(f"Error processing image {image_path}: {e}")
            raise
    
    def display_results(self, captions: List[str], scores: List[float], 
                       image_path: Optional[str] = None) -> None:
        """
        Display caption matching results in a formatted way.
        
        Args:
            captions (List[str]): Matched captions
            scores (List[float]): Similarity scores
            image_path (Optional[str]): Original image path
        """
        print("\n" + "="*60)
        if image_path:
            print(f"RESULTS FOR IMAGE: {Path(image_path).name}")
        print("="*60)
        
        formatted_results = format_results_display(captions, scores)
        print(formatted_results)
        print("="*60 + "\n")
    
    def save_results(self, captions: List[str], scores: List[float], 
                    image_path: str, output_dir: str = "results/outputs") -> None:
        """
        Save results to both JSON and CSV formats.
        
        Args:
            captions (List[str]): Matched captions
            scores (List[float]): Similarity scores
            image_path (str): Original image path
            output_dir (str): Output directory
        """
        # Prepare results data
        image_name = Path(image_path).stem
        results_data = {
            "image_path": str(image_path),
            "image_name": image_name,
            "model_used": self.model_name,
            "processing_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "top_matches": [
                {"rank": i+1, "caption": cap, "similarity_score": score}
                for i, (cap, score) in enumerate(zip(captions, scores))
            ]
        }
        
        # Create output paths
        output_path = Path(output_dir)
        json_filename = create_timestamp_filename(f"{image_name}_results", ".json")
        csv_filename = create_timestamp_filename(f"{image_name}_results", ".csv")
        
        # Save files
        save_results_to_json(results_data, output_path / json_filename)
        save_results_to_csv(captions, scores, output_path / csv_filename)
    
    def get_system_status(self) -> dict:
        """
        Get system status and configuration information.
        
        Returns:
            dict: System status information
        """
        model_info = self.embedding_generator.get_model_info()
        caption_stats = self.caption_matcher.get_database_stats()
        
        return {
            "model_info": model_info,
            "caption_database_stats": caption_stats,
            "components_status": {
                "image_processor": "Ready",
                "embedding_generator": "Ready", 
                "caption_matcher": "Ready"
            }
        }

def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(description="Multimodal AI Caption Matcher")
    parser.add_argument("--image_path", type=str, required=True, 
                       help="Path to input image")
    parser.add_argument("--top_k", type=int, default=DEFAULT_TOP_K,
                       help=f"Number of top matches (default: {DEFAULT_TOP_K})")
    parser.add_argument("--model", type=str, default=CLIP_MODEL_NAME,
                       help=f"CLIP model to use (default: {CLIP_MODEL_NAME})")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (auto/cuda/cpu)")
    parser.add_argument("--save_results", action="store_true",
                       help="Save results to files")
    parser.add_argument("--show_system_info", action="store_true",
                       help="Display system information")
    
    args = parser.parse_args()
    
    # Setup directories
    setup_directories()
    
    # Show system info if requested
    if args.show_system_info:
        print_system_info()
    
    try:
        # Initialize the system
        matcher = MultimodalCaptionMatcher(args.model, args.device)
        
        # Process the image
        best_captions, similarity_scores = matcher.process_single_image(
            args.image_path, top_k=args.top_k
        )
        
        # Display results
        matcher.display_results(best_captions, similarity_scores, args.image_path)
        
        # Save results if requested
        if args.save_results:
            matcher.save_results(best_captions, similarity_scores, args.image_path)
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
