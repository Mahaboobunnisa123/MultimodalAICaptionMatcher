# Multimodal AI Caption Matcher

## Project Overview
An intelligent caption matching system that leverages the power of OpenAI's CLIP model to find the most relevant captions for input images. This project demonstrates multimodal AI capabilities by bridging computer vision and natural language processing.

## Problem Statement
Traditional image captioning systems often struggle with understanding context and generating semantically meaningful descriptions. This project addresses the challenge of accurately matching images with appropriate captions by using deep learning embeddings to understand both visual and textual content in a shared semantic space.

## Advantages
- **Multimodal Understanding**: Processes both visual and textual data simultaneously
- **High Accuracy**: Uses pre-trained CLIP model with proven performance on 400M image-text pairs
- **Scalable Architecture**: Modular design allows easy expansion and customization
- **Real-time Processing**: Efficient embedding generation and similarity matching
- **Extensible Caption Database**: Easy to add, modify, or expand caption collections
- **Cross-Modal Search**: Can search images using text descriptions and vice versa

## Key Features
- Image preprocessing and normalization
- CLIP-based feature extraction for images and text
- Cosine similarity calculation for semantic matching
- Top-N caption ranking system
- Configurable caption database
- Comprehensive logging and result tracking

## Installation and Setup

### Prerequisites
- Python 3.10+ (Recommended: 3.10 or 3.11 due to PyTorch compatibility)
- CUDA-compatible GPU (optional but recommended)

### Installation Steps
**Note: This project's code isn't added for some secuirity purposes. Please go through these instructions before getting into the project.**

1. Create project directory:
```bash
mkdir MultimodalAICaptionMatcher
cd MultimodalAICaptionMatcher
```

2. Create virtual environment:
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Check package installation:
```bash
pip list | findstr "torch transformers pillow scikit-learn"
```

## Usage

### Basic Usage
```python
from src.caption_matcher import MultimodalCaptionMatcher

# Initialize the matcher
matcher = MultimodalCaptionMatcher()

# Process an image
image_path = "data/sample_images/your_image.jpg"
top_captions, similarity_scores = matcher.match_captions(image_path)

# Display results
matcher.display_results(top_captions, similarity_scores)
```

### Running the Main Script
```bash
python main.py --image_path "data/sample_images/sample.jpg" --top_k 5
```

## Project Structure Details

- **src/**: Core source code modules
- **models/**: Cached pre-trained models and model artifacts
- **results/**: Output files, logs, and generated results
- **data/**: Input images and datasets
- **configs/**: Configuration files and parameters
- **notebooks/**: Jupyter notebooks for experimentation
- **tests/**: Unit tests and validation scripts

## Technical Architecture

### Core Components
1. **Image Processor**: Handles image loading, preprocessing, and normalization
2. **Embedding Generator**: Extracts semantic embeddings using CLIP model
3. **Caption Matcher**: Performs similarity calculations and ranking
4. **Utils**: Helper functions and utilities

### Model Information
- **Base Model**: OpenAI CLIP (clip-vit-base-patch32)
- **Vision Encoder**: Vision Transformer (ViT)
- **Text Encoder**: Transformer-based language model
- **Embedding Dimension**: 512-dimensional vectors

## Performance Considerations
- First run will download the CLIP model (~600MB)
- Processing time depends on image size and caption database size
- GPU acceleration significantly improves performance
- Memory usage scales with batch size and model size

## Troubleshooting

### Common Issues
1. **PyTorch Installation**: Use Python 3.10 or 3.11 for best compatibility
2. **CUDA Issues**: Ensure proper CUDA version compatibility
3. **Memory Errors**: Reduce batch size or use CPU inference
4. **Import Errors**: Verify all dependencies are installed correctly

### Package Verification Command
```bash
python -c "import torch, transformers, PIL, sklearn; print('All packages imported successfully')"
```

## Contributing
This project follows clean code principles and modular architecture. Please maintain code quality and add appropriate tests for new features.

## License
This project is for educational and research purposes.

## Acknowledgments
- OpenAI for the CLIP model
- Hugging Face for the Transformers library
- PyTorch community for the deep learning framework