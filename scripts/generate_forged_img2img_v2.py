# scripts/generate_forged_img2img_v2.py
"""
Forgery Generation V2: Parameter Grid Exploration
Generates forgeries with varying denoising strengths, inference steps, and forgery counts
for comprehensive evaluation of face recognition vulnerabilities.
"""

import os, csv, json, random, argparse
from PIL import Image
import torch
from diffusers import StableDiffusionImg2ImgPipeline
from torchvision import transforms
import numpy as np
from tqdm import tqdm
from pathlib import Path
import itertools

def load_metadata(attr_file_path, identity_file_path=None):
    """
    Load CelebA attributes from list_attr_celeba.txt
    Optionally load identity mapping from list_identity_celeba.txt for organized directories
    Returns dict: filename -> {attribute_name: value, identity_id: id}
    """
    metadata = {}
    
    if not os.path.exists(attr_file_path):
        print(f"Warning: Attribute file not found at {attr_file_path}")
        return metadata
    
    # Load identity mapping if provided
    identity_map = {}
    if identity_file_path and os.path.exists(identity_file_path):
        with open(identity_file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    identity_map[parts[0]] = parts[1]
        print(f"Loaded identity mapping for {len(identity_map)} images")
    
    with open(attr_file_path, 'r') as f:
        lines = f.readlines()
        
        # First line is count, second line is attribute names
        num_images = int(lines[0].strip())
        attr_names = lines[1].strip().split()
        
        # Process each image
        for line in lines[2:]:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
                
            filename = parts[0]
            # Parse attributes (-1 means absent, 1 means present)
            attrs = {}
            for i, attr_name in enumerate(attr_names):
                if i + 1 < len(parts):
                    value = int(parts[i + 1])
                    if value == 1:
                        attrs[attr_name] = attr_name.replace('_', ' ')
                    else:
                        attrs[attr_name] = ""
            
            # Add identity if available
            if filename in identity_map:
                attrs['identity_id'] = identity_map[filename]
            
            metadata[filename] = attrs
    
    print(f"Loaded metadata for {len(metadata)} images")
    return metadata

def build_prompt(attrs, template):
    """Fill attribute tokens into template"""
    safe_attrs = {k: v for k, v in attrs.items()}
    
    try:
        return template.format(**safe_attrs)
    except KeyError as e:
        print(f"Warning: Template key {e} not found in attributes, using default prompt")
        return f"a {safe_attrs.get('quality', 'high-quality')} {safe_attrs.get('style', 'portrait photo')}"

def calculate_image_quality_metrics(image):
    """
    Calculate basic image quality metrics
    Returns dict with quality scores
    """
    import cv2
    
    # Convert PIL to numpy
    img_array = np.array(image)
    
    # Convert to grayscale for some metrics
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    
    # Calculate Laplacian variance (sharpness)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # Calculate brightness
    brightness = np.mean(img_array)
    
    # Calculate contrast
    contrast = np.std(img_array)
    
    return {
        "sharpness": float(laplacian_var),
        "brightness": float(brightness),
        "contrast": float(contrast)
    }

def generate_forgeries_with_params(pipe, real_img_path, prompt, out_dir, 
                                   num_forgeries, denoise, steps, cfg_scale, 
                                   device, param_suffix=""):
    """
    Generate forgeries with specific parameters
    
    Args:
        pipe: Stable Diffusion pipeline
        real_img_path: Path to real image
        prompt: Text prompt for generation
        out_dir: Output directory
        num_forgeries: Number of forgeries to generate
        denoise: Denoising strength (0-1)
        steps: Number of inference steps
        cfg_scale: Classifier-free guidance scale
        device: Device to use (cuda/cpu)
        param_suffix: Suffix for filenames to identify parameter set
    
    Returns:
        List of dicts with metadata for each generated image
    """
    img = Image.open(real_img_path).convert("RGB")
    results = []
    base_name = os.path.splitext(os.path.basename(real_img_path))[0]
    
    for i in range(num_forgeries):
        seed = random.randint(0, 2**30)
        
        # Ensure generator is on the correct device
        if device == "cuda":
            generator = torch.Generator(device="cuda").manual_seed(seed)
        else:
            generator = torch.Generator().manual_seed(seed)
        
        output = pipe(
            prompt=prompt, 
            image=img, 
            strength=denoise,
            guidance_scale=cfg_scale, 
            num_inference_steps=steps, 
            generator=generator
        )
        
        forged = output.images[0]
        
        # Create filename with parameter information
        fname = f"{base_name}_forg{param_suffix}_s{seed}.png"
        output_path = os.path.join(out_dir, fname)
        forged.save(output_path)
        
        # Calculate quality metrics
        quality_metrics = calculate_image_quality_metrics(forged)
        
        result = {
            "filename": fname,
            "seed": int(seed),
            "denoise_strength": denoise,
            "inference_steps": steps,
            "cfg_scale": cfg_scale,
            "quality_metrics": quality_metrics
        }
        results.append(result)
    
    return results

def create_visualization_grid(real_img_path, forged_paths, output_path, grid_size=(3, 3)):
    """
    Create a visualization grid showing real image vs forgeries
    """
    from PIL import Image, ImageDraw, ImageFont
    
    # Load real image
    real_img = Image.open(real_img_path).convert("RGB")
    
    # Resize to standard size
    img_size = (256, 256)
    real_img = real_img.resize(img_size)
    
    # Load and resize forged images
    forged_imgs = []
    for path in forged_paths[:grid_size[0] * grid_size[1] - 1]:  # -1 for real image
        img = Image.open(path).convert("RGB")
        img = img.resize(img_size)
        forged_imgs.append(img)
    
    # Create grid
    grid_width = grid_size[1] * img_size[0]
    grid_height = grid_size[0] * img_size[1]
    grid = Image.new('RGB', (grid_width, grid_height), color='white')
    
    # Place real image in top-left
    grid.paste(real_img, (0, 0))
    
    # Add label to real image
    draw = ImageDraw.Draw(grid)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except:
        font = ImageFont.load_default()
    draw.text((10, 10), "REAL", fill='red', font=font)
    
    # Place forged images
    for idx, forged in enumerate(forged_imgs):
        row = (idx + 1) // grid_size[1]
        col = (idx + 1) % grid_size[1]
        x = col * img_size[0]
        y = row * img_size[1]
        grid.paste(forged, (x, y))
        
        # Add label
        draw.text((x + 10, y + 10), f"FORGED {idx+1}", fill='blue', font=font)
    
    # Save grid
    grid.save(output_path)
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate forged images with parameter grid exploration")
    parser.add_argument("--model", required=True, help="Stable Diffusion model name or path")
    parser.add_argument("--in-dir", required=True, help="Input directory with real images")
    parser.add_argument("--meta", required=True, help="Path to list_attr_celeba.txt metadata file")
    parser.add_argument("--identity-meta", default=None, help="Path to list_identity_celeba.txt")
    parser.add_argument("--out-dir", required=True, help="Output directory for forged images")
    parser.add_argument("--template", default="a high-quality portrait photo of a person, {Male}, {Young}, {Black_Hair}, {Smiling}", 
                        help="Prompt template for generation")
    
    # Parameter grid options
    parser.add_argument("--denoise-grid", nargs='+', type=float, default=[0.2, 0.4, 0.6, 0.8],
                        help="Denoising strength values to explore")
    parser.add_argument("--steps-grid", nargs='+', type=int, default=[20, 40, 60],
                        help="Inference step counts to explore")
    parser.add_argument("--forgery-counts", nargs='+', type=int, default=[1, 3, 5, 9],
                        help="Number of forgeries per image per parameter set")
    parser.add_argument("--cfg-scale", type=float, default=9.0, 
                        help="Classifier-free guidance scale (fixed)")
    
    parser.add_argument("--limit", type=int, default=None, 
                        help="Limit number of identities to process")
    parser.add_argument("--create-visualizations", action='store_true',
                        help="Create side-by-side visualization grids")
    parser.add_argument("--viz-samples", type=int, default=5,
                        help="Number of identities to create visualizations for")
    
    args = parser.parse_args()

    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"\n{'='*80}")
    print("FORGERY GENERATION V2: PARAMETER GRID EXPLORATION")
    print(f"{'='*80}\n")
    
    # Print parameter grid
    print("Parameter Grid Configuration:")
    print(f"  Denoising strengths: {args.denoise_grid}")
    print(f"  Inference steps: {args.steps_grid}")
    print(f"  Forgery counts: {args.forgery_counts}")
    print(f"  CFG scale (fixed): {args.cfg_scale}")
    print(f"  Total parameter combinations: {len(args.denoise_grid) * len(args.steps_grid) * len(args.forgery_counts)}")
    print()
    
    # Load model
    print(f"Loading model: {args.model}")
    
    if device == "cuda":
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            args.model,
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False
        )
        pipe = pipe.to(device)
        try:
            pipe.enable_xformers_memory_efficient_attention()
            print("Enabled xformers memory efficient attention")
        except:
            print("xformers not available, using default attention")
    else:
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            args.model,
            torch_dtype=torch.float32,
            safety_checker=None,
            requires_safety_checker=False
        )
        pipe = pipe.to(device)
    
    print("Model loaded successfully\n")

    # Load metadata
    print(f"Loading metadata from: {args.meta}")
    meta = load_metadata(args.meta, args.identity_meta)
    
    # Create output directories
    os.makedirs(args.out_dir, exist_ok=True)
    viz_dir = os.path.join(args.out_dir, "visualizations")
    if args.create_visualizations:
        os.makedirs(viz_dir, exist_ok=True)

    # Process images and generate forgeries
    all_results = []
    processed = 0
    viz_created = 0
    
    # Create parameter combinations
    param_combinations = list(itertools.product(args.denoise_grid, args.steps_grid, args.forgery_counts))
    print(f"\nTotal parameter combinations: {len(param_combinations)}")
    print(f"Processing up to {args.limit if args.limit else 'all'} identities\n")
    
    for idx, (fname, attrs) in enumerate(tqdm(meta.items(), desc="Processing identities")):
        # Check limit
        if args.limit and processed >= args.limit:
            break
            
        # Try to find the image
        real_path = os.path.join(args.in_dir, fname)
        
        if not os.path.exists(real_path):
            alt = fname.replace('.jpg', '.png')
            real_path = os.path.join(args.in_dir, alt)
        
        if not os.path.exists(real_path):
            identity_id = attrs.get('identity_id')
            if identity_id:
                real_path = os.path.join(args.in_dir, identity_id, fname)
                if not os.path.exists(real_path):
                    alt = fname.replace('.jpg', '.png')
                    real_path = os.path.join(args.in_dir, identity_id, alt)
        
        if not os.path.exists(real_path):
            continue
        
        # Build prompt
        prompt = build_prompt(attrs, args.template)
        
        identity_results = {
            "real_image": fname,
            "real_path": real_path,
            "identity_id": attrs.get('identity_id', 'unknown'),
            "prompt": prompt,
            "parameter_sets": []
        }
        
        forged_paths_for_viz = []
        
        # Generate forgeries for each parameter combination
        for denoise, steps, num_forgeries in param_combinations:
            param_suffix = f"_d{int(denoise*100)}_s{steps}_n{num_forgeries}"
            
            forged_info = generate_forgeries_with_params(
                pipe, real_path, prompt, args.out_dir,
                num_forgeries, denoise, steps, args.cfg_scale,
                device, param_suffix
            )
            
            param_result = {
                "denoising_strength": denoise,
                "inference_steps": steps,
                "num_forgeries": num_forgeries,
                "forgeries": forged_info
            }
            identity_results["parameter_sets"].append(param_result)
            
            # Collect paths for visualization (use middle parameter set)
            if denoise == 0.4 and steps == 40:
                forged_paths_for_viz.extend([
                    os.path.join(args.out_dir, f["filename"]) for f in forged_info
                ])
        
        all_results.append(identity_results)
        processed += 1
        
        # Create visualization for first N identities
        if args.create_visualizations and viz_created < args.viz_samples and forged_paths_for_viz:
            viz_path = os.path.join(viz_dir, f"comparison_{fname.replace('.jpg', '.png')}")
            create_visualization_grid(real_path, forged_paths_for_viz, viz_path)
            viz_created += 1

    # Save comprehensive metadata
    metadata_path = os.path.join(args.out_dir, "metadata_v2.json")
    with open(metadata_path, "w") as f:
        json.dump({
            "experiment_config": {
                "model": args.model,
                "denoising_grid": args.denoise_grid,
                "steps_grid": args.steps_grid,
                "forgery_counts": args.forgery_counts,
                "cfg_scale": args.cfg_scale,
                "prompt_template": args.template
            },
            "summary": {
                "total_identities_processed": processed,
                "total_parameter_combinations": len(param_combinations),
                "total_forgeries_generated": sum(
                    sum(ps["num_forgeries"] for ps in r["parameter_sets"]) 
                    for r in all_results
                ),
                "visualizations_created": viz_created
            },
            "results": all_results
        }, f, indent=2)
    
    print(f"\n{'='*80}")
    print("GENERATION COMPLETE")
    print(f"{'='*80}")
    print(f"Processed identities: {processed}")
    print(f"Parameter combinations per identity: {len(param_combinations)}")
    print(f"Total forgeries generated: {sum(sum(ps['num_forgeries'] for ps in r['parameter_sets']) for r in all_results)}")
    print(f"Metadata saved to: {metadata_path}")
    if args.create_visualizations:
        print(f"Visualizations created: {viz_created} (saved to {viz_dir}/)")
    print(f"All forgeries saved to: {args.out_dir}")
    print(f"{'='*80}\n")
