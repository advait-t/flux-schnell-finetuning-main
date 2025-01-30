"""
FLUX Model Inference Script
--------------------------
This script loads a trained FLUX model with LoRA weights and generates an image from a text prompt.
It uses the Diffusers library and expects the LoRA weights to be in the specified output directory.
"""

import os
import sys
import torch
from collections import OrderedDict
from diffusers import AutoPipelineForText2Image

def setup_environment():
    """Add AI toolkit to Python path."""
    sys.path.append('./ai-toolkit')

def load_pipeline(model_id: str, weights_dir: str, lora_name: str) -> AutoPipelineForText2Image:
    """
    Load the FLUX pipeline with LoRA weights.
    
    Args:
        model_id (str): HuggingFace model ID for the base model
        weights_dir (str): Directory containing the LoRA weights
        lora_name (str): Name of the LoRA weights file (without extension)
    
    Returns:
        AutoPipelineForText2Image: Loaded pipeline ready for inference
    """
    # Initialize pipeline with base model
    pipeline = AutoPipelineForText2Image.from_pretrained(
        model_id,
        torch_dtype=torch.float16  # Use float16 for memory efficiency
    )
    
    # Move pipeline to GPU
    pipeline.to("cuda")
    
    # Load LoRA weights
    pipeline.load_lora_weights(
        weights_dir,
        weight_name=f"{lora_name}.safetensors"
    )
    
    return pipeline

def generate_image(pipeline: AutoPipelineForText2Image, prompt: str, output_path: str):
    """
    Generate and save an image using the pipeline.
    
    Args:
        pipeline (AutoPipelineForText2Image): Loaded FLUX pipeline
        prompt (str): Text prompt for image generation
        output_path (str): Path where the generated image will be saved
    """
    # Generate image
    image = pipeline(prompt).images[0]
    
    # Save the generated image
    image.save(output_path)

def main():
    """Main function to run the inference pipeline."""
    # Setup
    setup_environment()
    
    # Configuration
    weights_path = "./output/my_first_flux_lora_v1"
    model_id = "black-forest-labs/FLUX.1-schnell"
    lora_name = "my_first_flux_lora_v1"
    prompt = "Subject is an astronaut, in ocean"
    output_file = f"{weights_path}/output.png"
    
    # Load model and generate image
    pipeline = load_pipeline(model_id, weights_path, lora_name)
    generate_image(pipeline, prompt, output_file)

if __name__ == "__main__":
    main()
