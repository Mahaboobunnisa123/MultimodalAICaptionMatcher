"""
Utility functions for the Multimodal AI Caption Matcher project.

This module provides helper functions for logging, file operations, and result formatting.
"""

import logging
import json
import csv
from pathlib import Path
from typing import List, Dict, Any, Union
from datetime import datetime
from typing import Optional

def setup_project_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """
    Setup logging configuration for the project.
    
    Args:
        log_level (str): Logging level ("DEBUG", "INFO", "WARNING", "ERROR")
        log_file (Optional[str]): Path to log file (optional)
        
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logger
    logger = logging.getLogger("MultimodalAI")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear any existing handlers
    logger.handlers = []
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def save_results_to_json(results: Dict[str, Any], output_path: Union[str, Path]) -> None:
    """
    Save results dictionary to JSON file.
    
    Args:
        results (Dict[str, Any]): Results data to save
        output_path (Union[str, Path]): Output file path
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to: {output_file}")

def save_results_to_csv(captions: List[str], scores: List[float], 
                       output_path: Union[str, Path]) -> None:
    """
    Save caption matching results to CSV file.
    
    Args:
        captions (List[str]): List of matched captions
        scores (List[float]): List of similarity scores
        output_path (Union[str, Path]): Output CSV file path
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Rank', 'Caption', 'Similarity_Score'])
        
        for idx, (caption, score) in enumerate(zip(captions, scores), 1):
            writer.writerow([idx, caption, f"{score:.4f}"])
    
    print(f"Results saved to CSV: {output_file}")

def format_results_display(captions: List[str], scores: List[float], 
                          top_n: int = 5) -> str:
    """
    Format caption matching results for display.
    
    Args:
        captions (List[str]): List of matched captions
        scores (List[float]): List of similarity scores
        top_n (int): Number of top results to display
        
    Returns:
        str: Formatted results string
    """
    display_count = min(top_n, len(captions))
    result_lines = [f"Top {display_count} Caption Matches:"]
    result_lines.append("=" * 50)
    
    for idx in range(display_count):
        result_lines.append(
            f"{idx + 1}. {captions[idx]}"
            f"\n   Similarity Score: {scores[idx]:.4f}"
        )
    
    return "\n".join(result_lines)

def validate_file_path(file_path: Union[str, Path], 
                      must_exist: bool = True) -> Path:
    """
    Validate and convert file path.
    
    Args:
        file_path (Union[str, Path]): File path to validate
        must_exist (bool): Whether file must already exist
        
    Returns:
        Path: Validated Path object
        
    Raises:
        FileNotFoundError: If file doesn't exist and must_exist is True
        ValueError: If path is invalid
    """
    path_obj = Path(file_path)
    
    if must_exist and not path_obj.exists():
        raise FileNotFoundError(f"File not found: {path_obj}")
    
    return path_obj

def create_timestamp_filename(base_name: str, extension: str = ".json") -> str:
    """
    Create a filename with timestamp.
    
    Args:
        base_name (str): Base filename without extension
        extension (str): File extension
        
    Returns:
        str: Filename with timestamp
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base_name}_{timestamp}{extension}"

def get_file_size_mb(file_path: Union[str, Path]) -> float:
    """
    Get file size in megabytes.
    
    Args:
        file_path (Union[str, Path]): Path to file
        
    Returns:
        float: File size in MB
    """
    path_obj = Path(file_path)
    if path_obj.exists():
        size_bytes = path_obj.stat().st_size
        return size_bytes / (1024 * 1024)
    return 0.0

def ensure_directory_exists(directory_path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if it doesn't.
    
    Args:
        directory_path (Union[str, Path]): Directory path
        
    Returns:
        Path: Directory path object
    """
    dir_path = Path(directory_path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path

def clean_filename(filename: str) -> str:
    """
    Clean filename by removing invalid characters.
    
    Args:
        filename (str): Original filename
        
    Returns:
        str: Cleaned filename
    """
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    return filename

def calculate_processing_time(start_time: float, end_time: float) -> Dict[str, str]:
    """
    Calculate and format processing time.
    
    Args:
        start_time (float): Start timestamp
        end_time (float): End timestamp
        
    Returns:
        Dict[str, str]: Formatted time information
    """
    duration = end_time - start_time
    
    return {
        "duration_seconds": f"{duration:.2f}",
        "duration_formatted": f"{duration:.2f} seconds",
        "start_time": datetime.fromtimestamp(start_time).strftime("%Y-%m-%d %H:%M:%S"),
        "end_time": datetime.fromtimestamp(end_time).strftime("%Y-%m-%d %H:%M:%S")
    }

def print_system_info() -> None:
    """Print system and environment information."""
    import torch
    import sys
    
    print("=" * 50)
    print("SYSTEM INFORMATION")
    print("=" * 50)
    print(f"Python Version: {sys.version}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    print("=" * 50)