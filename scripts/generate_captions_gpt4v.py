#!/usr/bin/env python3
"""
Generate captions for pathology images using GPT-4V (Vision).

This script creates vision-language pairs by generating detailed
medical descriptions for pathology images.
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import List, Dict, Optional
import base64
from tqdm import tqdm
import time

try:
    import openai
except ImportError:
    print("Error: openai package not installed")
    print("Install with: pip install openai")
    sys.exit(1)

from PIL import Image


PATHOLOGY_PROMPT = """Describe this pathology image in detail. Include:

1. **Tissue Type**: What tissue/organ is shown (e.g., breast, lung, colon)
2. **Staining**: Type of stain used (e.g., H&E, IHC)
3. **Magnification**: Approximate magnification level
4. **Cellular Features**: 
   - Cell types present
   - Cell morphology and arrangement
   - Nuclear features (size, shape, chromatin pattern)
   - Cytoplasmic features
5. **Architectural Patterns**: Tissue organization and structure
6. **Pathological Findings**: 
   - Normal vs abnormal features
   - Specific pathological changes
   - Diagnostic features
7. **Diagnostic Impression**: Most likely diagnosis or differential

Be specific and use medical terminology. Focus on observable features."""


def encode_image(image_path: Path) -> str:
    """Encode image to base64."""
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def resize_image_if_needed(image_path: Path, max_size: int = 2048) -> Path:
    """Resize image if it exceeds max_size to reduce API costs."""
    img = Image.open(image_path)
    
    if max(img.size) <= max_size:
        return image_path
    
    # Resize maintaining aspect ratio
    ratio = max_size / max(img.size)
    new_size = tuple(int(dim * ratio) for dim in img.size)
    img = img.resize(new_size, Image.Resampling.LANCZOS)
    
    # Save to temp file
    temp_path = image_path.parent / f"temp_{image_path.name}"
    img.save(temp_path)
    
    return temp_path


def generate_caption(
    image_path: Path,
    client: openai.OpenAI,
    model: str = "gpt-4o",
    max_tokens: int = 500,
    temperature: float = 0.3
) -> Dict[str, any]:
    """Generate caption for a single image using GPT-4V."""
    
    # Resize if needed
    processed_path = resize_image_if_needed(image_path)
    
    try:
        # Encode image
        image_data = encode_image(processed_path)
        
        # Call GPT-4V
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": PATHOLOGY_PROMPT
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_data}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        caption = response.choices[0].message.content
        
        return {
            "success": True,
            "caption": caption,
            "model": model,
            "tokens": response.usage.total_tokens,
            "cost": estimate_cost(response.usage.total_tokens, model)
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
    
    finally:
        # Clean up temp file
        if processed_path != image_path and processed_path.exists():
            processed_path.unlink()


def estimate_cost(tokens: int, model: str) -> float:
    """Estimate API cost based on tokens."""
    # GPT-4V pricing (as of 2024)
    if "gpt-4o" in model:
        # $2.50 per 1M input tokens, $10 per 1M output tokens
        # Approximate 80% input, 20% output
        input_tokens = int(tokens * 0.8)
        output_tokens = int(tokens * 0.2)
        cost = (input_tokens / 1_000_000 * 2.50) + (output_tokens / 1_000_000 * 10.0)
    else:
        # Fallback estimate
        cost = tokens / 1_000_000 * 5.0
    
    return cost


def process_dataset(
    image_dir: Path,
    output_dir: Path,
    api_key: str,
    model: str = "gpt-4o",
    max_images: Optional[int] = None,
    skip_existing: bool = True,
    batch_delay: float = 1.0
):
    """Process all images in a directory."""
    
    # Initialize OpenAI client
    client = openai.OpenAI(api_key=api_key)
    
    # Find all images
    image_extensions = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}
    image_paths = [
        p for p in image_dir.rglob('*')
        if p.suffix.lower() in image_extensions
    ]
    
    if max_images:
        image_paths = image_paths[:max_images]
    
    print(f"Found {len(image_paths)} images to process")
    
    # Create output directories
    captions_dir = output_dir / "captions"
    images_out_dir = output_dir / "images"
    captions_dir.mkdir(parents=True, exist_ok=True)
    images_out_dir.mkdir(parents=True, exist_ok=True)
    
    # Process images
    results = []
    total_cost = 0.0
    total_tokens = 0
    
    for image_path in tqdm(image_paths, desc="Generating captions"):
        # Check if already processed
        caption_path = captions_dir / f"{image_path.stem}.txt"
        if skip_existing and caption_path.exists():
            continue
        
        # Generate caption
        result = generate_caption(image_path, client, model)
        
        if result["success"]:
            # Save caption
            with open(caption_path, 'w', encoding='utf-8') as f:
                f.write(result["caption"])
            
            # Copy image to output
            import shutil
            shutil.copy2(image_path, images_out_dir / image_path.name)
            
            # Track stats
            total_cost += result["cost"]
            total_tokens += result["tokens"]
            
            results.append({
                "image": str(image_path),
                "caption_file": str(caption_path),
                "tokens": result["tokens"],
                "cost": result["cost"]
            })
        else:
            print(f"\n✗ Failed to process {image_path.name}: {result['error']}")
            results.append({
                "image": str(image_path),
                "error": result["error"]
            })
        
        # Rate limiting
        time.sleep(batch_delay)
    
    # Save results
    results_path = output_dir / "generation_results.json"
    with open(results_path, 'w') as f:
        json.dump({
            "total_images": len(image_paths),
            "successful": len([r for r in results if "caption_file" in r]),
            "failed": len([r for r in results if "error" in r]),
            "total_tokens": total_tokens,
            "total_cost": total_cost,
            "model": model,
            "results": results
        }, f, indent=2)
    
    print(f"\n{'='*60}")
    print("Generation Complete")
    print(f"{'='*60}")
    print(f"Total images: {len(image_paths)}")
    print(f"Successful: {len([r for r in results if 'caption_file' in r])}")
    print(f"Failed: {len([r for r in results if 'error' in r])}")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Estimated cost: ${total_cost:.2f}")
    print(f"\nCaptions saved to: {captions_dir}")
    print(f"Results saved to: {results_path}")


def create_vision_language_manifest(output_dir: Path):
    """Create a manifest file for vision-language training."""
    captions_dir = output_dir / "captions"
    images_dir = output_dir / "images"
    
    manifest = []
    
    for caption_file in captions_dir.glob("*.txt"):
        # Find corresponding image
        image_name = caption_file.stem
        
        # Try different extensions
        image_path = None
        for ext in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']:
            candidate = images_dir / f"{image_name}{ext}"
            if candidate.exists():
                image_path = candidate
                break
        
        if image_path:
            with open(caption_file, 'r', encoding='utf-8') as f:
                caption = f.read().strip()
            
            manifest.append({
                "image": str(image_path.relative_to(output_dir)),
                "caption": caption
            })
    
    manifest_path = output_dir / "vision_language_manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"✓ Created manifest with {len(manifest)} image-text pairs")
    print(f"  Location: {manifest_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate captions for pathology images using GPT-4V"
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        required=True,
        help="Directory containing pathology images"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for captions and processed images"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        help="OpenAI API key (or set OPENAI_API_KEY env var)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="Model to use (default: gpt-4o)"
    )
    parser.add_argument(
        "--max-images",
        type=int,
        help="Maximum number of images to process"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip images that already have captions"
    )
    parser.add_argument(
        "--batch-delay",
        type=float,
        default=1.0,
        help="Delay between API calls in seconds (default: 1.0)"
    )
    
    args = parser.parse_args()
    
    # Get API key
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OpenAI API key required")
        print("Provide via --api-key or set OPENAI_API_KEY environment variable")
        sys.exit(1)
    
    # Validate inputs
    if not args.image_dir.exists():
        print(f"Error: Image directory not found: {args.image_dir}")
        sys.exit(1)
    
    print("=" * 60)
    print("GPT-4V Caption Generator for Pathology Images")
    print("=" * 60)
    print(f"Image directory: {args.image_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Model: {args.model}")
    print(f"Max images: {args.max_images or 'unlimited'}")
    print("=" * 60)
    
    # Process dataset
    process_dataset(
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        api_key=api_key,
        model=args.model,
        max_images=args.max_images,
        skip_existing=args.skip_existing,
        batch_delay=args.batch_delay
    )
    
    # Create manifest
    create_vision_language_manifest(args.output_dir)
    
    print("\n✓ All done! Ready for vision-language training.")


if __name__ == "__main__":
    main()
